# -*- coding: utf-8 -*-
"""
ELO НА КОТИРОВКАХ (Betting Odds Rating System, Wunderlich & Memmert 2018).

Идея статьи: обычный Эло учится на результате S ∈ {1, ½, 0}, который на
одном матче -- почти шум. Рынок же перед матчем уже выдал ожидание
E_odds = p(П1) + ½·p(X) (открытие Bet365 без маржи), и оно куда точнее
одного результата. Поэтому обновление рейтинга делается по РЫНОЧНОМУ
ожиданию, а не по факту:

    d      = R_home - R_away + h                  (h -- преимущество поля)
    E_elo  = 1 / (1 + 10^(-d/400))
    R_home += K·(S - E_elo),   R_away -= K·(S - E_elo)

где S -- «наблюдение». Два варианта, оба сравниваются ВНУТРИ predict на train:
    ORS:     S = E_odds                          (чистая статья)
    ORS+xG:  S = ½·E_odds + ½·E_xg               (E_xg -- ожидание из xG:
             П1 + ½·X по Пуассону с λ = xG с игры × масштаб голы/xG)
Смысл второго: рынок медленно признаёт перемены в игре, xG их видит раньше
(Mead 2023), но xG одного матча шумный -- поэтому пополам, а не целиком.
Вес ½ фиксирован, не подбирается: два варианта, а не сетка. На всех 106
игровых днях окна оценки train выбирает ORS+xG; чистая статья на этой лиге
хуже рынка (по протоколу 0.967 на открытии, 0.954 на закрытии).

Цены для ОБУЧЕНИЯ рейтинга -- закрывающие (odds_*), они в train известны и
точнее открытия (log-loss 0.917 против 0.947); где закрытия нет -- открытие.
В test закрытия нет, и оно там не нужно: прогноз идёт от рейтингов, а
открытие test используется лишь для команды, которой в рейтинге ещё нет.

Прогноз -- порядковый логит по разнице рейтингов (три параметра: наклон и
два порога), подогнанный на train по ПРЕДМАТЧЕВЫМ разницам, т.е. так же,
как он будет применён к test. Наклон свободный -- именно он может
исправить известную на этой лиге недооценку сильных фаворитов.

Константы.
  * Шкала 400 -- стандарт Эло, чтобы K читался привычно.
  * K подбирается ВНУТРИ predict на train: минимум log-loss порядкового
    логита по предматчевым разницам (одномерный, гладкий, с внутренним
    минимумом около 60-120 -- проверено на срезах train). Критерий «как быстро
    рейтинг догоняет рынок» без результатов был бы чище, но он монотонно
    просит K -> ∞ (рейтинг = вчерашняя котировка), т.е. не отделяет сигнал
    от шума котировок; поэтому результаты в критерии всё же нужны.
  * Каждому варианту -- свой K, вариант выбирается по тому же log-loss
    на train. Наклон логита при этом свободный, так что сравнение честное:
    у вариантов одинаковое число параметров.
  * Новичок лиги стартует со среднего итогового рейтинга выбывших команд
    (их место он и занял); если таких нет -- с 0.
  * Команда, которой в рейтинге ещё нет (первый тур новичка), получает
    рейтинг из открытия того же матча -- открытие в test разрешено.

Слабое место: рейтинг -- это сглаженный рынок, и обыграть открытие он
может только там, где рынок систематически ошибается, а не за счёт новой
информации. По протоколу он ровно на уровне открытия (Δlog-loss -0.003,
t=-0.3) и хуже закрытия; наклон логита на train получается ~1.3 от Эло,
т.е. рейтинги «растягиваются», но в окне оценки бакет p>0.75 у модели
слегка перепредсказан (80% против 78%) -- недооценку фаворитов этот
подход не лечит.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from markets import devig_power

SCALE = 400.0          # шкала Эло
K_LO, K_HI = 2.0, 200.0
W_XG = 0.5             # доля xG-ожидания в наблюдении варианта ORS+xG
MAXG = 10
XG_COLS = (('h_xg_open', 'a_xg_open'), ('h_xg', 'a_xg'))   # приоритет столбцов xG
ODDS_PRE = ('odds', 'open')   # цены для обучения рейтинга: закрытие, иначе открытие
LAST = {}              # диагностика последнего вызова predict: вариант, K, наклон


# --------------------------------------------------------------- вспомогательное
def _expect(d):
    """Ожидаемый результат Эло по разнице рейтингов."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(d, float) / SCALE))


def _to_points(e):
    """Обратное: ожидание -> разница рейтингов."""
    e = float(np.clip(e, 1e-4, 1 - 1e-4))
    return SCALE * np.log10(e / (1.0 - e))


def _odds_expect(row, pres=None):
    """Ожидание хозяев из цен: первый доступный префикс из pres."""
    for pre in (ODDS_PRE if pres is None else pres):
        if f'{pre}_H' not in row.index:
            continue
        o = [row[f'{pre}_H'], row[f'{pre}_D'], row[f'{pre}_A']]
        if any(pd.isna(x) for x in o) or any(x <= 1.0 for x in o):
            continue
        q = devig_power(o)
        if np.any(np.isnan(q)):
            continue
        return float(q[0] + 0.5 * q[1])
    return np.nan


def _xg_expect(lh, la):
    """П1 + ½·X по независимому Пуассону."""
    if not (np.isfinite(lh) and np.isfinite(la)):
        return np.nan
    ph = poisson.pmf(np.arange(MAXG + 1), max(lh, 1e-3))
    pa = poisson.pmf(np.arange(MAXG + 1), max(la, 1e-3))
    M = np.outer(ph, pa)
    M /= M.sum()
    x = np.arange(MAXG + 1)[:, None]
    y = np.arange(MAXG + 1)[None, :]
    return float(M[x > y].sum() + 0.5 * M[x == y].sum())


def _prepare(train):
    """Предрасчёт наблюдений по каждому матчу train (в хронологии)."""
    d = train[train.hg.notna()].sort_values('ts').reset_index(drop=True)
    # масштаб голы / xG (xG с игры не содержит стандартов и пенальти)
    hx = ax = None
    for ch, ca in XG_COLS:
        if ch in d.columns and ca in d.columns:
            hx = d[ch].astype(float).values if hx is None else np.where(np.isnan(hx), d[ch].astype(float).values, hx)
            ax = d[ca].astype(float).values if ax is None else np.where(np.isnan(ax), d[ca].astype(float).values, ax)
    if hx is None:
        hx = ax = np.full(len(d), np.nan)
    have = ~(np.isnan(hx) | np.isnan(ax))
    tot_xg = float((hx[have] + ax[have]).sum())
    scale = float((d.hg.values[have] + d.ag.values[have]).sum() / tot_xg) if tot_xg > 0 else 1.0

    s_res = np.where(d.hg > d.ag, 1.0, np.where(d.hg == d.ag, 0.5, 0.0))
    s_odds = np.array([_odds_expect(r) for _, r in d.iterrows()], float)
    s_xg = np.array([_xg_expect(scale * a, scale * b) if ok else np.nan
                     for a, b, ok in zip(hx, ax, have)], float)
    return d, s_res, s_odds, s_xg


def _run(d, S, K, h):
    """
    Прогон рейтингов по train в хронологии. S -- наблюдение по матчу
    (NaN -> матч пропускается в обновлении, но предматчевая разница считается).
    -> (предматчевые разницы d_pre с учётом h, словарь рейтингов)
    """
    R = {}
    seen_season = {}          # команда -> последний сезон, в котором играла
    cur_season = None
    dpre = np.full(len(d), np.nan)
    homes, aways, seasons = d.home.values, d.away.values, d.season.values
    for i in range(len(d)):
        s = seasons[i]
        if s != cur_season:
            # старт сезона: новички получают средний рейтинг выбывших
            if cur_season is not None:
                teams_now = set(d.home.values[seasons == s]) | set(d.away.values[seasons == s])
                gone = [t for t, ss in seen_season.items() if ss == cur_season and t not in teams_now]
                init = float(np.mean([R[t] for t in gone])) if gone else 0.0
                for t in teams_now:
                    if t not in R:
                        R[t] = init
            cur_season = s
        a, b = homes[i], aways[i]
        R.setdefault(a, 0.0)
        R.setdefault(b, 0.0)
        seen_season[a] = s
        seen_season[b] = s
        dd = R[a] - R[b] + h
        dpre[i] = dd
        if np.isfinite(S[i]):
            delta = K * (S[i] - _expect(dd))
            R[a] += delta
            R[b] -= delta
    return dpre, R


def _fit_ordered_logit(dpre, out):
    """
    Порядковый логит: P(П2) = σ(c1 - a·d), P(П1) = 1 - σ(c2 - a·d), X -- между.
    -> params (a, c1, gap), где c2 = c1 + exp(gap).
    """

    def probs(th, d):
        a, c1, gap = th
        c2 = c1 + np.exp(gap)
        z1 = 1.0 / (1.0 + np.exp(-(c1 - a * d)))
        z2 = 1.0 / (1.0 + np.exp(-(c2 - a * d)))
        P = np.column_stack([1.0 - z2, z2 - z1, z1])
        return np.clip(P, 1e-9, 1.0)

    def nll(th):
        P = probs(th, dpre)
        # крошечный гребень на наклон -- только чтобы Nelder-Mead не уплывал
        return -float(np.sum(np.log(P[np.arange(len(out)), out]))) + 1e-3 * th[0] ** 2

    # старт: наклон = ln(10)/400 (то есть в точности Эло), пороги под долю ничьих
    th0 = np.array([np.log(10.0) / SCALE, -0.5, np.log(1.0)])
    r = minimize(nll, th0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
    return r.x, probs


# ---------------------------------------------------------------------- модель
def predict(train, test):
    d, s_res, s_odds, s_xg = _prepare(train)
    out = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2)).astype(int)

    # наблюдения двух вариантов; где нет открытия -- берём результат
    # (12 матчей в базе), где нет xG -- только рыночную часть
    S_ors = np.where(np.isfinite(s_odds), s_odds, s_res)
    S_mix = np.where(np.isfinite(s_xg), (1 - W_XG) * S_ors + W_XG * s_xg, S_ors)

    # преимущество поля в очках Эло: среднее рыночное ожидание хозяев
    ok = np.isfinite(s_odds)
    h = _to_points(float(s_odds[ok].mean())) if ok.sum() >= 20 else 0.0

    warm = min(30, len(d) // 4)          # первые матчи: рейтинги ещё нули
    m = np.arange(len(d)) >= warm

    def fit_for(S, K):
        dpre, R = _run(d, S, K, h)
        th, probs = _fit_ordered_logit(dpre[m], out[m])
        ll = -float(np.mean(np.log(probs(th, dpre[m])[np.arange(m.sum()), out[m]])))
        return ll, th, probs, R

    # два варианта, каждому свой K по log-loss train, выбор варианта -- по нему же
    best = None
    for name, S in (('ors', S_ors), ('ors_xg', S_mix)):
        K = float(minimize_scalar(lambda k: fit_for(S, k)[0], bounds=(K_LO, K_HI),
                                  method='bounded', options=dict(xatol=1.0)).x)
        ll, th, probs, R = fit_for(S, K)
        if best is None or ll < best[0]:
            best = (ll, name, K, th, probs, R)
    ll, name, K, th, probs, R = best
    LAST.update(variant=name, K=K, h=h, slope=float(th[0] / (np.log(10.0) / SCALE)), ll_train=ll)

    P = []
    for _, r in test.iterrows():
        if r.home in R and r.away in R:
            dd = R[r.home] - R[r.away] + h
        else:
            # команды ещё нет в рейтинге: разница из открытия этого же матча
            e = _odds_expect(r, ('open',))
            dd = _to_points(e) if np.isfinite(e) else h
        P.append(probs(th, np.array([dd]))[0])
    P = np.asarray(P, float)
    return P / P.sum(axis=1, keepdims=True)
