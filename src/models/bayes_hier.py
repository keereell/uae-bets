# -*- coding: utf-8 -*-
"""
Иерархический Пуассон на xG с ОЦЕНКОЙ СЖАТИЯ по данным (эмпирический Байес).

    lambda_home = exp(mu + gamma + beta * (atk[home] - def[away]))
    lambda_away = exp(mu         + beta * (atk[away] - def[home]))
    atk_i, def_i ~ N(m_i, tau^2)          m_i = 0, у новичка лиги -promoted_shift

Чем отличается от действующей модели (src/model.py). Там сила команд
сжимается к априору штрафом alpha * sum(theta^2) с alpha = 0.30, взятым
«из теории». В терминах нормального априора это tau^2 = 1 / (2 alpha),
то есть tau ~ 1.29 на лог-шкале -- априор практически плоский: разброс
сил команд в лиге на самом деле порядка 0.2, и такой штраф почти ничего
не сжимает. Здесь tau -- параметр, который оценивается по train
максимизацией МАРГИНАЛЬНОГО правдоподобия (Лапласово приближение):

    log p(y | tau) ~ log p(y | th*) + log p(th* | tau) - 1/2 log det H(th*)

где th* -- мода апостериорного распределения при данном tau, H -- гессиан
минус-лог-апостериора в моде. Слагаемое с определителем -- «штраф за
сложность»: маленький tau делает априор узким (много log(1/tau)), но и
гессиан большим; оптимум там, где разброс оценок сил команд отвечает
их реальной информативности. Это обычный ML для случайных эффектов,
только без предположения нормальности наблюдений.

Двухэтапная схема, как в src/model.py, с одним дополнением:
  Этап 1. atk/def -- на xG с игры (h_xg_open/a_xg_open), квази-пуассоновское
          правдоподобие с дисперсией phi (оценка Пирсона: xG вдвое менее
          шумный, чем голы), априор N(m, tau^2), tau -- по маргинальному
          правдоподобию.
  Этап 2. mu, gamma И МАСШТАБ beta -- на РЕАЛЬНЫХ голах при фиксированных
          atk/def. Апостериорные средние сил -- это сжатый предиктор, и
          использовать его в регрессии на другую целевую переменную положено
          с собственным наклоном: beta говорит, насколько сильнее (или
          слабее) разница сил по xG с игры проявляется в голах. Голы
          включают пенальти и стандарты (треть всех голов лиги), которые
          xG с игры не видит, но которые тоже растут с классом команды.

ЧТО ПОКАЗАЛ ПРОТОКОЛ (walk-forward с 2025-02-01, n=264), честно:
  * tau по маргинальному правдоподобию устойчив: 0.18-0.25, медиана 0.21
    на всех 106 игровых днях. Оценка не шумная.
  * Но чистое сжатие (без beta) ХУЖЕ действующей модели: log-loss 0.96648
    против 0.96216; сжатие лишь уплощает прогноз, а фавориты в этой лиге
    и так недооценены. Перебор ФИКСИРОВАННОГО tau (только для диагностики)
    монотонен: 0.1 -> 1.007, 0.2 -> 0.968, 0.5 -> 0.953, 1.9 -> 0.951.
    Сжатие, оправданное правдоподобием xG, -- пересжатие для прогноза голов.
  * С beta на голах масштаб сил перестаёт зависеть от tau: при tau = 0.21
    beta ~ 1.29, при tau = 1.9 -- beta ~ 0.91, итоговый log-loss в обоих
    случаях ~0.9547. Именно наклон, откалиброванный по целевой переменной,
    и есть работающее «сжатие по данным»; сжатие по xG само по себе -- нет.
  Итог модели: log-loss 0.95473 (действующая 0.96216, открытие Bet365
  0.94692); против открытия +0.0105 +- 0.0106 (t = +0.99) -- на уровне
  рынка в пределах ошибки, но не лучше него.

Константы, взятые не из данных:
  half_life = 180 дней  -- по результатам проверки переноса (см. predict.py)
  promoted_shift = 0.25 -- мягкий априор «новичок слабее среднего»; тот же,
                           что в действующей модели
  xG с игры, без пенальти и стандартов -- как в действующей модели
  phi -- дисперсия Пирсона, оценивается на train, не гиперпараметр.

Команды, которых в train ещё нет (первый тур новичка), получают силу,
равную априорному среднему -- это ровно то, что байесовская модель и должна
делать; действующая модель в таких матчах выдаёт равномерный прогноз
(4 матча из 264 в окне оценки).

Слабые места. (1) tau оценивается по ~14 командам -- оценка дисперсии
случайного эффекта на 14 группах в принципе неточна, хотя здесь она
и оказалась стабильной. (2) Прогноз по моде апостериора (plug-in), а не по
апостериорному предиктивному распределению; при апостериорной SD сил
~0.1 разница пренебрежима (множитель exp(s^2/2) ~ 1.005). (3) Решение
добавить beta принято ПОСЛЕ прогона чистого варианта по протоколу; это
выбор из двух кандидатов по тестовому окну, хоть и с теоретическим
обоснованием, -- см. блок выше. (4) Рынок не обыгран: структурная
модель на xG упирается в потолок ~0.95 при любом сжатии.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from model import detect_newcomers, score_matrix
from markets import wdl

HALF_LIFE = 180.0
PROMOTED_SHIFT = 0.25
XG_COLS = ('h_xg_open', 'a_xg_open')
LOG_TAU_BOUNDS = (np.log(0.03), np.log(2.0))   # диапазон поиска tau на лог-шкале


class HierPoisson:
    def __init__(self, half_life=HALF_LIFE, promoted_shift=PROMOTED_SHIFT,
                 xg_cols=XG_COLS):
        self.half_life = half_life
        self.promoted_shift = promoted_shift
        self.xg_cols = tuple(xg_cols)
        self.tau = None
        self.phi = None
        self.beta = 1.0

    # ------------------------------------------------------------ данные
    def _design(self, hi, ai, n):
        """Матрица плана для стека [домашние наблюдения; гостевые]:
        столбцы [mu, gamma, atk_1..n, def_1..n]."""
        m = len(hi)
        p = 2 + 2 * n
        X = np.zeros((2 * m, p))
        r = np.arange(m)
        X[r, 0] = 1.0
        X[r, 1] = 1.0
        X[r, 2 + hi] = 1.0
        X[r, 2 + n + ai] = -1.0
        X[m + r, 0] = 1.0
        X[m + r, 2 + ai] = 1.0
        X[m + r, 2 + n + hi] = -1.0
        return X

    # ------------------------------------------------------------ этап 1
    def _map(self, X, y, c, prior, tau, th0):
        """Мода апостериора при данном tau: минимизируем
        sum c*(lam - y*eta) + sum (th_team - prior)^2 / (2 tau^2)."""
        prec = np.zeros(X.shape[1])
        prec[2:] = 1.0 / tau ** 2

        def f(th):
            eta = np.clip(X @ th, -6.0, 3.0)
            lam = np.exp(eta)
            d = th - prior
            return float(np.sum(c * (lam - y * eta)) + 0.5 * np.sum(prec * d * d))

        def g(th):
            eta = np.clip(X @ th, -6.0, 3.0)
            lam = np.exp(eta)
            return X.T @ (c * (lam - y)) + prec * (th - prior)

        r = minimize(f, th0, jac=g, method='L-BFGS-B',
                     options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
        return r.x, r.fun

    def _log_marginal(self, tau, X, y, c, prior, th0):
        """Лапласово приближение log p(y | tau) (с точностью до константы)."""
        th, nll_pen = self._map(X, y, c, prior, tau, th0)
        lam = np.exp(np.clip(X @ th, -6.0, 3.0))
        H = X.T @ (X * (c * lam)[:, None])
        k = X.shape[1] - 2
        H[np.arange(2, 2 + k), np.arange(2, 2 + k)] += 1.0 / tau ** 2
        sign, logdet = np.linalg.slogdet(H)
        if sign <= 0:
            return -np.inf, th
        # -NLL_pen уже содержит log-плотность априора без нормировки;
        # нормировка k гауссиан даёт -k log(tau); Лаплас -- -1/2 log det H.
        return float(-nll_pen - k * np.log(tau) - 0.5 * logdet), th

    def fit(self, d, ref_ts=None, newcomers=(), extra_teams=()):
        d = d[d.played & d.hg.notna()].copy()
        c0, c1 = self.xg_cols
        m = d.dropna(subset=[c0, c1])
        # xG перекалибровывается на уровень голов: xG с игры составляет ~2/3
        # всех голов, без масштаба квази-правдоподобие считало бы наблюдения
        # менее информативными, чем они есть
        self.xg_scale = (float((m.hg + m.ag).sum() / (m[c0] + m[c1]).sum())
                         if len(m) and (m[c0] + m[c1]).sum() > 0 else 1.0)
        ref_ts = ref_ts if ref_ts is not None else d.ts.max()
        dt = np.maximum((ref_ts - d.ts.values) / 86400.0, 0.0)
        w = np.exp(-np.log(2) / self.half_life * dt)

        teams = sorted((set(d.home) | set(d.away)) | set(extra_teams))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d.home.map(self.idx).values
        ai = d.away.map(self.idx).values

        hg = d.hg.values.astype(float)
        ag = d.ag.values.astype(float)
        hx = d[c0].values.astype(float) * self.xg_scale
        ax = d[c1].values.astype(float) * self.xg_scale
        have = ~(np.isnan(hx) | np.isnan(ax))
        yh = np.where(have, hx, hg)      # без xG -- голы (2 матча из 385)
        ya = np.where(have, ax, ag)

        X = self._design(hi, ai, n)
        y = np.concatenate([yh, ya])
        ww = np.concatenate([w, w])

        prior = np.zeros(2 + 2 * n)
        for t in newcomers:
            if t in self.idx:
                prior[2 + self.idx[t]] = -self.promoted_shift
                prior[2 + n + self.idx[t]] = -self.promoted_shift

        th0 = prior.copy()
        th0[0] = np.log(max(y.mean(), 0.1))
        th0[1] = 0.05

        # --- дисперсия Пирсона при предварительном tau; сама по себе
        # к tau малочувствительна, поэтому одной итерации достаточно
        th_pre, _ = self._map(X, y, ww, prior, 0.3, th0)
        lam = np.exp(np.clip(X @ th_pre, -6.0, 3.0))
        dof = max(len(y) - X.shape[1], 1)
        self.phi = float(np.clip(np.sum((y - lam) ** 2 / np.maximum(lam, 1e-6)) / dof,
                                 0.15, 2.0))
        c = ww / self.phi

        # --- tau по маргинальному правдоподобию: грубая сетка, затем Брент
        grid = np.linspace(LOG_TAU_BOUNDS[0], LOG_TAU_BOUNDS[1], 12)
        vals = [self._log_marginal(np.exp(lt), X, y, c, prior, th_pre)[0] for lt in grid]
        j = int(np.argmax(vals))
        lo = grid[max(j - 1, 0)]
        hi_ = grid[min(j + 1, len(grid) - 1)]
        if hi_ - lo < 1e-6:
            best_lt = grid[j]
        else:
            r = minimize_scalar(lambda lt: -self._log_marginal(np.exp(lt), X, y, c, prior, th_pre)[0],
                                bounds=(lo, hi_), method='bounded',
                                options=dict(xatol=1e-3))
            best_lt = r.x if -r.fun >= vals[j] else grid[j]
        self.tau = float(np.exp(best_lt))
        self.lml, th1 = self._log_marginal(self.tau, X, y, c, prior, th_pre)
        self.atk = th1[2:2 + n].copy()
        self.dfn = th1[2 + n:].copy()

        # --- этап 2: уровень, преимущество поля и масштаб сил на реальных голах
        base_h = self.atk[hi] - self.dfn[ai]
        base_a = self.atk[ai] - self.dfn[hi]

        def nll2(t):
            eh = np.clip(t[0] + t[1] + t[2] * base_h, -6.0, 3.0)
            ea = np.clip(t[0] + t[2] * base_a, -6.0, 3.0)
            lh, la = np.exp(eh), np.exp(ea)
            return float(np.sum(w * (lh - hg * eh)) + np.sum(w * (la - ag * ea)))

        def g2(t):
            eh = np.clip(t[0] + t[1] + t[2] * base_h, -6.0, 3.0)
            ea = np.clip(t[0] + t[2] * base_a, -6.0, 3.0)
            rh = w * (np.exp(eh) - hg)
            ra = w * (np.exp(ea) - ag)
            return np.array([rh.sum() + ra.sum(), rh.sum(),
                             float(rh @ base_h + ra @ base_a)])

        r2 = minimize(nll2, np.array([th1[0], th1[1], 1.0]), jac=g2, method='L-BFGS-B',
                      options=dict(maxiter=500))
        self.mu, self.gamma, self.beta = float(r2.x[0]), float(r2.x[1]), float(r2.x[2])
        self.eff_n = float(w.sum())
        return self

    # ------------------------------------------------------------ прогноз
    def lambdas(self, home, away):
        i, j = self.idx[home], self.idx[away]
        lh = np.exp(self.mu + self.gamma + self.beta * (self.atk[i] - self.dfn[j]))
        la = np.exp(self.mu + self.beta * (self.atk[j] - self.dfn[i]))
        return float(lh), float(la)

    def probs(self, home, away):
        lh, la = self.lambdas(home, away)
        p = wdl(score_matrix(lh, la, 0.0))
        return np.array([p['H'], p['D'], p['A']], float)


def predict(train, test):
    """Интерфейс единого протокола: (train, test) -> (n_test, 3) [П1, X, П2]."""
    nc = detect_newcomers(pd.concat([train, test]))
    new_here = set()
    for s in test.season.unique():
        new_here |= nc.get(s, set())
    # команды из test, которых в train ещё нет, входят в модель с силой,
    # равной априорному среднему (0 или -promoted_shift для новичка)
    extra = set(test.home) | set(test.away)
    m = HierPoisson().fit(train, ref_ts=float(test.ts.min()),
                          newcomers=new_here, extra_teams=extra)
    return np.vstack([m.probs(r.home, r.away) for _, r in test.iterrows()])
