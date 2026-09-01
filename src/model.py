# -*- coding: utf-8 -*-
"""
Модель Диксона-Коулза для UAE Pro League.

    lambda_home = exp(mu + gamma + atk[home] - def[away] + X*beta)
    lambda_away = exp(mu         + atk[away] - def[home] + X*beta)

Двухэтапная подгонка:
  Этап 1. atk/def оцениваются на СМЕСИ голов и перекалиброванного xG
          (xG менее шумный -> лучше оценивает силу команды).
  Этап 2. mu, gamma (и опциональные ковариаты) переоцениваются на РЕАЛЬНЫХ ГОЛАХ
          при фиксированных atk/def. Это гарантирует, что средний уровень
          результативности и преимущество поля откалиброваны под то,
          на что реально принимаются ставки.

Плюс: экспоненциальное затухание веса матчей, L2-сжатие к априору,
поправка Диксона-Коулза rho на низкие счета, априор «новичка лиги».
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAXG = 12  # матрица счёта 0..MAXG


def dc_tau(rho, lh, la):
    """Поправка Диксона-Коулза tau(x, y) для низких счетов."""
    t = np.ones((MAXG + 1, MAXG + 1))
    t[0, 0] = 1.0 - lh * la * rho
    t[0, 1] = 1.0 + lh * rho
    t[1, 0] = 1.0 + la * rho
    t[1, 1] = 1.0 - rho
    return np.clip(t, 1e-9, None)


def score_matrix(lh, la, rho=0.0):
    """Совместное распределение счёта P(x, y)."""
    ph = poisson.pmf(np.arange(MAXG + 1), lh)
    pa = poisson.pmf(np.arange(MAXG + 1), la)
    m = np.outer(ph, pa)
    if rho:
        m = m * dc_tau(rho, lh, la)
    return m / m.sum()


def detect_newcomers(df):
    """{сезон: множество команд, которых не было в предыдущем сезоне}"""
    seasons = sorted(df.season.unique())
    teams = {s: set(df[df.season == s].home) | set(df[df.season == s].away) for s in seasons}
    out = {}
    for i, s in enumerate(seasons):
        out[s] = set() if i == 0 else teams[s] - teams[seasons[i - 1]]
    return out


class DixonColes:
    def __init__(self, half_life=180.0, alpha=1.5, w_goals=0.35,
                 xg_scale=None, promoted_shift=0.25, covariates=(),
                 xg_cols=('h_xg', 'a_xg'), dispersion=None):
        self.half_life = half_life        # период полураспада веса матча, дней
        self.alpha = alpha                # сила L2-сжатия к априору
        self.w_goals = w_goals            # доля голов в целевой переменной (остальное xG)
        self.xg_scale = xg_scale          # перекалибровка xG (None -> оценить по данным)
        self.promoted_shift = promoted_shift
        self.covariates = list(covariates)
        self.xg_cols = tuple(xg_cols)
        # Пуассоновское ядро предполагает Var = lambda. Для xG это неверно:
        # измеренная дисперсия Пирсона около 0.47, то есть xG вдвое менее шумный,
        # чем голы. Без поправки правдоподобие занижает информативность xG примерно
        # в два раза, и L2-штраф оказывается вдвое сильнее, чем задумано.
        # dispersion=None -> оценить по данным (две итерации).
        self.dispersion = dispersion
        self.rho = 0.0
        self.beta = np.zeros(len(self.covariates))

    # -------------------------------------------------- целевая переменная
    def _target(self, d):
        hg = d.hg.values.astype(float)
        ag = d.ag.values.astype(float)
        hx = d[self.xg_cols[0]].values.astype(float) * self.xg_scale
        ax = d[self.xg_cols[1]].values.astype(float) * self.xg_scale
        have = ~(np.isnan(hx) | np.isnan(ax))
        w = np.where(have, self.w_goals, 1.0)
        yh = w * hg + (1 - w) * np.where(have, hx, hg)
        ya = w * ag + (1 - w) * np.where(have, ax, ag)
        return yh, ya

    # -------------------------------------------------- подгонка
    def fit(self, d, ref_ts=None, newcomers=()):
        d = d[d.played & d.hg.notna()].copy()
        if self.xg_scale is None:
            c0, c1 = self.xg_cols
            m = d.dropna(subset=[c0, c1])
            self.xg_scale = (float((m.hg + m.ag).sum() / (m[c0] + m[c1]).sum())
                             if len(m) and (m[c0] + m[c1]).sum() > 0 else 1.0)
        ref_ts = ref_ts if ref_ts is not None else d.ts.max()
        dt_days = np.maximum((ref_ts - d.ts.values) / 86400.0, 0.0)
        w = np.exp(-np.log(2) / self.half_life * dt_days)

        teams = sorted(set(d.home) | set(d.away))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d.home.map(self.idx).values
        ai = d.away.map(self.idx).values
        yh, ya = self._target(d)

        prior = np.zeros(2 * n)
        for t in newcomers:
            if t in self.idx:
                prior[self.idx[t]] = -self.promoted_shift
                prior[n + self.idx[t]] = -self.promoted_shift

        # --- этап 1: сила команд на смеси голов и xG
        def nll1(th):
            mu, gam, atk, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]
            eh = mu + gam + atk[hi] - dfn[ai]
            ea = mu + atk[ai] - dfn[hi]
            lh = np.exp(np.clip(eh, -6, 3))
            la = np.exp(np.clip(ea, -6, 3))
            return (inv_phi * (np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea)))
                    + self.alpha * np.sum((th[2:] - prior) ** 2))

        def g1(th):
            mu, gam, atk, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]
            eh = mu + gam + atk[hi] - dfn[ai]
            ea = mu + atk[ai] - dfn[hi]
            lh = np.exp(np.clip(eh, -6, 3))
            la = np.exp(np.clip(ea, -6, 3))
            rh, ra = inv_phi * w * (lh - yh), inv_phi * w * (la - ya)
            g = np.zeros_like(th)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            np.add.at(g, 2 + hi, rh)
            np.add.at(g, 2 + ai, ra)
            np.add.at(g, 2 + n + ai, -rh)
            np.add.at(g, 2 + n + hi, -ra)
            g[2:] += 2 * self.alpha * (th[2:] - prior)
            return g

        th0 = np.concatenate([[np.log(max(yh.mean(), 0.1)), 0.05], prior])
        inv_phi = 1.0
        r1 = minimize(nll1, th0, jac=g1, method='L-BFGS-B',
                      options=dict(maxiter=1000, ftol=1e-12))
        if self.dispersion is None:
            # дисперсия Пирсона целевой переменной вокруг подогнанных lambda
            mu0, gam0 = r1.x[0], r1.x[1]
            a0, d0 = r1.x[2:2 + n], r1.x[2 + n:]
            lh0 = np.exp(np.clip(mu0 + gam0 + a0[hi] - d0[ai], -6, 3))
            la0 = np.exp(np.clip(mu0 + a0[ai] - d0[hi], -6, 3))
            num = np.sum((yh - lh0) ** 2 / np.maximum(lh0, 1e-6)) +                   np.sum((ya - la0) ** 2 / np.maximum(la0, 1e-6))
            self.dispersion = float(np.clip(num / max(2 * len(d) - 2 * n - 2, 1), 0.15, 2.0))
        inv_phi = 1.0 / self.dispersion
        if abs(inv_phi - 1.0) > 1e-6:
            r1 = minimize(nll1, r1.x, jac=g1, method='L-BFGS-B',
                          options=dict(maxiter=1000, ftol=1e-12))
        self.atk, self.dfn = r1.x[2:2 + n], r1.x[2 + n:]

        # --- этап 2: уровень и преимущество поля на реальных голах
        base_h = self.atk[hi] - self.dfn[ai]
        base_a = self.atk[ai] - self.dfn[hi]
        gh = d.hg.values.astype(float)
        ga = d.ag.values.astype(float)
        if self.covariates:
            X = np.column_stack([d[c].astype(float).fillna(d[c].astype(float).median()).values
                                 for c in self.covariates])
            self._cov_mean = X.mean(axis=0)
            Xc = X - self._cov_mean
        else:
            X = np.zeros((len(d), 0))
            self._cov_mean = np.zeros(0)
            Xc = X
        k = X.shape[1]

        def nll2(th):
            mu, gam, beta = th[0], th[1], th[2:]
            adj = Xc @ beta if k else 0.0
            eh = mu + gam + base_h + adj
            ea = mu + base_a + adj
            lh = np.exp(np.clip(eh, -6, 3))
            la = np.exp(np.clip(ea, -6, 3))
            return (np.sum(w * (lh - gh * eh)) + np.sum(w * (la - ga * ea))
                    + 5.0 * float(np.sum(beta ** 2)))

        r2 = minimize(nll2, np.concatenate([[r1.x[0], r1.x[1]], np.zeros(k)]),
                      method='L-BFGS-B', options=dict(maxiter=600))
        self.mu, self.gamma, self.beta = r2.x[0], r2.x[1], r2.x[2:]
        self.eff_n = float(w.sum())
        # rho обнулён намеренно. Профиль правдоподобия по нему даёт SE 0.124
        # и 95% интервал [-0.20, +0.20] при оценке +0.030 -- параметр
        # неидентифицируем на 370 матчах. Клетки 0-0 и 1-1, ради которых
        # поправка вводится, лежат ровно на пуассоновском ожидании (obs/exp
        # 0.98 и 0.99); вверх rho тянут 0-1 и 1-0, то есть асимметрия
        # дом/гости, которую tau выразить не может. Снятие параметра стоит
        # -0.00038 RPS (t = -2.24 на 250 матчах) -- это упрощение, а не выигрыш.
        self.rho = 0.0
        return self

    def _fit_rho(self, d, hi, ai, w, Xc):
        adj = Xc @ self.beta if len(self.beta) else 0.0
        lh = np.exp(self.mu + self.gamma + self.atk[hi] - self.dfn[ai] + adj)
        la = np.exp(self.mu + self.atk[ai] - self.dfn[hi] + adj)
        hg = d.hg.values.astype(int)
        ag = d.ag.values.astype(int)

        def f(rho):
            t = np.ones(len(d))
            m = (hg == 0) & (ag == 0)
            t[m] = 1 - lh[m] * la[m] * rho
            m = (hg == 0) & (ag == 1)
            t[m] = 1 + lh[m] * rho
            m = (hg == 1) & (ag == 0)
            t[m] = 1 + la[m] * rho
            m = (hg == 1) & (ag == 1)
            t[m] = 1 - rho
            if (t <= 0).any():
                return 1e9
            return -float(np.sum(w * np.log(t)))

        grid = np.linspace(-0.30, 0.20, 101)
        vals = [f(r) for r in grid]
        return float(grid[int(np.argmin(vals))])

    # -------------------------------------------------- предсказание
    def lambdas(self, home, away, cov=None):
        i, j = self.idx.get(home), self.idx.get(away)
        if i is None or j is None:
            raise KeyError('нет в модели: ' + str(home if i is None else away))
        adj = 0.0
        if self.covariates:
            v = np.array([cov[c] for c in self.covariates], float) - self._cov_mean
            adj = float(v @ self.beta)
        lh = np.exp(self.mu + self.gamma + self.atk[i] - self.dfn[j] + adj)
        la = np.exp(self.mu + self.atk[j] - self.dfn[i] + adj)
        return float(lh), float(la)

    def matrix(self, home, away, cov=None):
        lh, la = self.lambdas(home, away, cov)
        return score_matrix(lh, la, self.rho), lh, la

    def ratings(self):
        df = pd.DataFrame({'team': self.teams, 'atk': self.atk, 'def': self.dfn})
        df['rating'] = df.atk + df['def']
        return df.sort_values('rating', ascending=False).reset_index(drop=True)
