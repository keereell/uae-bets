# -*- coding: utf-8 -*-
"""
Диксон-Коулз на xG с РЫНОЧНЫМ АПРИОРОМ.

Базовая модель (src/model.py) сжимает силы команд к нулю: штраф
alpha*||theta||^2 говорит «все команды одинаковые, пока xG не докажет
обратное». Это плохой априор, когда есть котировки: рейтинг на котировках
бьёт рейтинг на голах (Wunderlich & Memmert 2018). Здесь рынок входит в
модель ДО подгонки, а не смесью после неё, в двух ролях.

  0. Рыночные силы. Из открывающей линии 1X2 Bet365 каждого матча train
     (степенной де-виг, как в протоколе) восстанавливаются пуассоновские
     (lh, la): два уравнения на два неизвестных, решение точное. На этих
     лямбдах подгоняется обычный Диксон-Коулз (полураспад 180, xg_scale=1)
     -> theta_m = (atk_m, def_m): сила команд глазами рынка, сглаженная
     по времени. Новичкам отдельный априор не нужен -- рынок их оценил.

  1. xG-подгонка с априором N(theta_m, tau^2): штраф kappa*||theta-theta_m||^2,
     kappa = 1/(2 tau^2). tau -- насколько сила по xG расходится с рыночной --
     оценивается ЭМПИРИЧЕСКИМ БАЙЕСОМ на train: разброс (theta_xG - theta_m)
     по командам минус шум оценки theta_xG (диагональ информации Фишера).
     tau^2 обрезано в [0.05^2, 0.5^2], чтобы одна выборка не увела kappa
     в бесконечность или в ноль. Факт по этой лиге: разброс 0.028 МЕНЬШЕ
     шума 0.041, tau упирается в пол -- xG не несёт информации о силе
     команд сверх рыночного рейтинга, theta почти равно theta_m.

  2. Открывающая линия ДНЯ ПРОГНОЗА -- разрешённый признак -- это измерение
     log-lambda ЭТОГО матча: сила команд плюс матчевый эффект (составы,
     мотивация, то, что рейтинг по определению не видит) плюс ошибка линии.
     Постериор log-lambda при нормальном приближении:
         log lam = f + beta * (l - f),   f -- прогноз рейтинга, l -- линия,
     beta = V/(V + s^2), V -- дисперсия матчевого эффекта, s^2 -- ошибка линии.
     beta = 0 -- чистая модель, beta = 1 -- чистое открытие. Раньше линия
     входила псевдонаблюдением в общую подгонку, где её давил тот же штраф
     kappa, что и xG, -- это ошибка структуры: линия измеряет не силу.

     Ни V, ни s^2 из теории не выводятся: ошибка открытия по движению
     открытие->закрытие не идентифицируется (зависит от того, какую долю
     ошибки закрытие исправляет), а наклон регрессии голов на линию
     нестабилен между сезонами. Поэтому beta подбирается ВНУТРИ predict
     на train: внутренний walk-forward по игровым дням train даёт для
     каждого матча train ВНЕВЫБОРОЧНЫЙ прогноз рейтинга f (подгонка только
     на матчах до его дня), и beta = argmin log-loss 1X2 смеси на этих
     матчах. Один параметр на отрезке [0, 1], без сетки по тесту. Подгонки
     по дням кэшируются: они зависят только от матчей до дня, и train
     любого более позднего дня их содержит.

Этап 2 -- как в базе: уровень mu и преимущество поля gamma переоцениваются
на РЕАЛЬНЫХ голах при фиксированных atk/def.

Слабые места. (а) beta оценён по ~250-370 матчам train с SE порядка 0.15:
при beta -> 1 модель вырождается в открытие, и это честный итог, если
рейтинг+xG не добавляют к линии ничего. (б) Матч, где хотя бы одна команда
не встречалась в train, получает де-вигнутую открывающую линию. (в) Если
у матча в test линии нет -- чистый прогноз рейтинга (beta не применить).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DixonColes, score_matrix          # noqa: E402
from markets import wdl, DEVIG                      # noqa: E402

HALF_LIFE = 180.0            # полураспад веса матча, дней (как в базе)
ALPHA_MARKET = 0.30          # L2 рыночного рейтинга к среднему (как в базе; при phi
                             # рыночных лямбд 0.15 штраф практически не действует)
ALPHA_DIAG = 0.05            # слабое L2 диагностической xG-подгонки для EB
PROMOTED_SHIFT = 0.25        # априор новичка там, где рынка нет
XG_COLS = ('h_xg_open', 'a_xg_open')
TAU_MIN, TAU_MAX = 0.05, 0.50
MIN_HIST = 80                # минимум матчей истории для внутреннего walk-forward
DEVIG_METHOD = 'power'

_LAMBDA_CACHE = {}           # game_id -> (lh, la): чистая функция от линии
_FIT_CACHE = {}              # (день, число матчей до него) -> подгонка на этот день


# ---------------------------------------------------------------- рыночные лямбды
def implied_lambdas(odds):
    """Пуассоновские (lh, la) при rho=0, точно воспроизводящие де-вигнутый 1X2."""
    q = np.asarray(DEVIG[DEVIG_METHOD](odds), float)
    if np.any(np.isnan(q)):
        return None

    def resid(x):
        p = wdl(score_matrix(np.exp(x[0]), np.exp(x[1]), 0.0))
        return [p['H'] - q[0], p['D'] - q[1]]

    r = least_squares(resid, [0.4, 0.1], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if r.cost > 1e-8:
        return None
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def _line_lambdas(df):
    """Массивы lh/la: рыночные лямбды по открывающей линии (NaN, если линии нет)."""
    lh, la = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for k, r in enumerate(df.itertuples()):
        o = (r.open_H, r.open_D, r.open_A)
        if any(pd.isna(x) for x in o):
            continue
        key = int(r.game_id)
        if key not in _LAMBDA_CACHE:
            _LAMBDA_CACHE[key] = implied_lambdas(o)
        v = _LAMBDA_CACHE[key]
        if v is not None:
            lh[k], la[k] = v
    return lh, la


def _line_probs(row):
    o = (row.open_H, row.open_D, row.open_A)
    if any(pd.isna(x) for x in o):
        return np.array([1 / 3, 1 / 3, 1 / 3])
    q = np.asarray(DEVIG[DEVIG_METHOD](o), float)
    return q if not np.any(np.isnan(q)) else np.array([1 / 3, 1 / 3, 1 / 3])


def _wdl(lh, la):
    p = wdl(score_matrix(lh, la, 0.0))
    return np.array([p['H'], p['D'], p['A']])


# ---------------------------------------------------------------- подгонка на день
def _fit_day(hist, ref_ts):
    """
    Рейтинг на момент ref_ts по матчам hist (все строго раньше).
    -> dict(idx, atk, dfn, mu, gamma, diag).
    """
    d = hist[hist.played & hist.hg.notna()].copy()
    d['h_lm'], d['a_lm'] = _line_lambdas(d)

    # --- 0. рыночный рейтинг: DC на рыночных лямбдах train
    mk = DixonColes(half_life=HALF_LIFE, alpha=ALPHA_MARKET, w_goals=0.0, xg_scale=1.0,
                    promoted_shift=PROMOTED_SHIFT, xg_cols=('h_lm', 'a_lm')).fit(d, ref_ts=ref_ts)

    # --- диагностическая xG-подгонка со слабым L2: theta_xG, дисперсия phi, xg_scale
    dg = DixonColes(half_life=HALF_LIFE, alpha=ALPHA_DIAG, w_goals=0.0,
                    promoted_shift=PROMOTED_SHIFT, xg_cols=XG_COLS).fit(d, ref_ts=ref_ts)
    teams, idx = dg.teams, dg.idx
    n = len(teams)
    phi = float(dg.dispersion)
    inv_phi = 1.0 / phi

    hi = d.home.map(idx).values
    ai = d.away.map(idx).values
    dt_days = np.maximum((ref_ts - d.ts.values) / 86400.0, 0.0)
    w = np.exp(-np.log(2) / HALF_LIFE * dt_days)
    yh, ya = dg._target(d)

    # рыночный априор в «калибре» xG-подгонки: центрируем atk и def по отдельности
    # (общий сдвиг всех atk или всех def -- калибровочная свобода, её съедает mu)
    theta_m = np.zeros(2 * n)
    has_m = np.zeros(2 * n, bool)
    for t, i in idx.items():
        j = mk.idx.get(t)
        if j is not None:
            theta_m[i], theta_m[n + i] = mk.atk[j], mk.dfn[j]
            has_m[i] = has_m[n + i] = True
    for sl in (slice(0, n), slice(n, 2 * n)):
        m = has_m[sl]
        if m.any():
            theta_m[sl][m] -= theta_m[sl][m].mean()
    theta_x = np.concatenate([dg.atk - dg.atk.mean(), dg.dfn - dg.dfn.mean()])

    # --- 1. эмпирический Байес: tau^2 = разброс (theta_xG - theta_m) минус шум оценки
    lh0 = np.exp(np.clip(dg.mu + dg.gamma + dg.atk[hi] - dg.dfn[ai], -6, 3))
    la0 = np.exp(np.clip(dg.mu + dg.atk[ai] - dg.dfn[hi], -6, 3))
    info = np.zeros(2 * n)
    np.add.at(info, hi, inv_phi * w * lh0)          # atk хозяев
    np.add.at(info, ai, inv_phi * w * la0)          # atk гостей
    np.add.at(info, n + ai, inv_phi * w * lh0)      # def гостей
    np.add.at(info, n + hi, inv_phi * w * la0)      # def хозяев
    noise = 1.0 / (info + 2 * ALPHA_DIAG)
    dd = (theta_x - theta_m)[has_m]
    tau2 = float(np.mean(dd ** 2) - np.mean(noise[has_m])) if has_m.any() else TAU_MAX ** 2
    tau2 = float(np.clip(tau2, TAU_MIN ** 2, TAU_MAX ** 2))
    kappa = 1.0 / (2.0 * tau2)

    # штраф: к рыночной силе с kappa; там, где рынка нет -- к нулю с alpha базы
    pen = np.where(has_m, kappa, ALPHA_MARKET)
    prior = np.where(has_m, theta_m, 0.0)

    def nll(th):
        mu, gam, atk, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]
        eh = mu + gam + atk[hi] - dfn[ai]
        ea = mu + atk[ai] - dfn[hi]
        lh = np.exp(np.clip(eh, -6, 3))
        la = np.exp(np.clip(ea, -6, 3))
        return (inv_phi * (np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea)))
                + np.sum(pen * (th[2:] - prior) ** 2))

    def grad(th):
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
        g[2:] += 2 * pen * (th[2:] - prior)
        return g

    th0 = np.concatenate([[dg.mu, dg.gamma], prior])
    r1 = minimize(nll, th0, jac=grad, method='L-BFGS-B',
                  options=dict(maxiter=2000, ftol=1e-12))
    atk, dfn = r1.x[2:2 + n], r1.x[2 + n:]

    # --- этап 2: mu, gamma на реальных голах train при фиксированных силах
    base_h = atk[hi] - dfn[ai]
    base_a = atk[ai] - dfn[hi]
    gh = d.hg.values.astype(float)
    ga = d.ag.values.astype(float)

    def nll2(th):
        eh = th[0] + th[1] + base_h
        ea = th[0] + base_a
        lh = np.exp(np.clip(eh, -6, 3))
        la = np.exp(np.clip(ea, -6, 3))
        return np.sum(w * (lh - gh * eh)) + np.sum(w * (la - ga * ea))

    r2 = minimize(nll2, np.array([r1.x[0], r1.x[1]]), method='L-BFGS-B',
                  options=dict(maxiter=600))
    return dict(mu=float(r2.x[0]), gamma=float(r2.x[1]), atk=atk, dfn=dfn, idx=idx,
                diag=dict(tau=float(np.sqrt(tau2)), kappa=float(kappa), phi=phi,
                          var_d=float(np.mean(dd ** 2)) if has_m.any() else np.nan,
                          noise=float(np.mean(noise[has_m])) if has_m.any() else np.nan,
                          eff_n=float(w.sum()) / max(n, 1) * 2))


def _fit_cached(hist, ref_ts, day):
    key = (str(day), int(len(hist)))
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = _fit_day(hist, ref_ts)
    return _FIT_CACHE[key]


def _rating_loglam(fit, home, away):
    """(log lh, log la) по рейтингу или None, если команды в подгонке нет."""
    i, j = fit['idx'].get(home), fit['idx'].get(away)
    if i is None or j is None:
        return None
    return (fit['mu'] + fit['gamma'] + fit['atk'][i] - fit['dfn'][j],
            fit['mu'] + fit['atk'][j] - fit['dfn'][i])


# ---------------------------------------------------------------- вес линии
def _fit_beta(train):
    """
    beta = argmin log-loss 1X2 смеси log lam = f + beta*(l - f) по матчам train,
    где f -- вневыборочный прогноз рейтинга (внутренний walk-forward по дням).
    -> (beta, число матчей, log-loss при beta=0, при beta=1, при beta)
    """
    d = train[train.played & train.hg.notna()].sort_values('ts')
    lh_l, la_l = _line_lambdas(d)
    F, L, O = [], [], []
    for day in sorted(d.date.unique()):
        hist = d[d.date < day]
        if len(hist) < MIN_HIST:
            continue
        today = d[d.date == day]
        fit = _fit_cached(hist, float(today.ts.min()), day)
        pos = np.where((d.date == day).values)[0]
        for k, r in zip(pos, today.itertuples()):
            if np.isnan(lh_l[k]):
                continue
            f = _rating_loglam(fit, r.home, r.away)
            if f is None:
                continue
            F.append(f)
            L.append((np.log(lh_l[k]), np.log(la_l[k])))
            O.append(0 if r.hg > r.ag else (1 if r.hg == r.ag else 2))
    if len(O) < 30:
        return 1.0, len(O), np.nan, np.nan, np.nan
    F, L, O = np.array(F), np.array(L), np.array(O, int)

    def ll(beta):
        s = 0.0
        for f, l, o in zip(F, L, O):
            e = f + beta * (l - f)
            p = _wdl(float(np.exp(e[0])), float(np.exp(e[1])))
            s -= np.log(max(p[o], 1e-9))
        return s / len(O)

    r = minimize_scalar(ll, bounds=(0.0, 1.0), method='bounded', options=dict(xatol=1e-3))
    return float(r.x), len(O), float(ll(0.0)), float(ll(1.0)), float(r.fun)


LAST_DIAG = {}               # диагностика последней подгонки (tau, kappa, beta, ...)


def predict(train, test):
    day = str(test.date.min())
    ref_ts = float(test.ts.min())
    fit = _fit_cached(train, ref_ts, day)
    beta, n_beta, ll0, ll1, llb = _fit_beta(train)
    LAST_DIAG.update(fit['diag'])
    LAST_DIAG.update(beta=beta, n_beta=n_beta, ll_model=ll0, ll_line=ll1, ll_blend=llb)

    tl_h, tl_a = _line_lambdas(test)
    out = []
    for k, (_, r) in enumerate(test.iterrows()):
        f = _rating_loglam(fit, r.home, r.away)
        if f is None:
            out.append(_line_probs(r))          # команды не было в train: только линия
            continue
        eh, ea = f
        if not np.isnan(tl_h[k]):
            eh = eh + beta * (np.log(tl_h[k]) - eh)
            ea = ea + beta * (np.log(tl_a[k]) - ea)
        out.append(_wdl(float(np.exp(eh)), float(np.exp(ea))))
    return np.array(out, float)
