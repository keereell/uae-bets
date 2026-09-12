# -*- coding: utf-8 -*-
"""
Рынок + калибровка: открывающая цена Bet365 без маржи, перекалиброванная
по прошлым матчам.

    q  = степенной де-виг открытия (markets.DEVIG['power'])
    p_k  ∝  q_k^T * exp(a_k),   a_H = 0,   k ∈ {П1, X, П2}

Три параметра: температура T (T > 1 -- рынок недооценивает фаворитов,
T < 1 -- переоценивает) и два сдвига классов a_X, a_П2 (рынок системно
недодаёт ничьей или гостям). При T = 1, a = 0 модель совпадает с сырым
открытием -- это нулевая точка, к которой параметры сжимаются
L2-штрафом lam * (ln T)^2 + lam * (a_X^2 + a_П2^2).

Сила сжатия lam НЕ фиксируется: она выбирается ВНУТРИ predict по K-fold
кросс-валидации на train, сетка от «почти без штрафа» до «калибровки нет»
(lam = 1e6 -> T = 1, a = 0). Так модель сама решает, есть ли в прошлых
матчах устойчивое смещение рынка, которое стоит переносить на будущее.

ЧТО ПОКАЗАЛ ПОДХОД (walk-forward с 2025-02-01, n = 264). Кросс-валидация
на каждом игровом дне выбирает lam = 1e6, то есть ОТКАЗ от калибровки:
результат совпадает с сырым открытием (log-loss 0.94692). Любая фиксированная
сила калибровки хуже рынка (lam = 3..30: +0.0015..+0.0037 log-loss), то же
для изотонной регрессии по классам (+0.0013) и для логит-регрессии по
классам. Причина: смещение рынка на этой лиге НЕ устойчиво во времени --
по сезону 2024/25 T = 0.91 (рынок переуверен), по 2025/26 T = 1.30 (рынок
недоуверен); калибровка, выученная на прошлом сезоне, в новом действует
в обратную сторону. «Зацепка» с сильными фаворитами (кэф до 1.35, факт 90%
при 80% по рынку) держится на 21 матче и в train до момента прогноза
не видна. Взвешивание train по давности (полураспад 180 дн.) не помогает:
+0.0012..+0.0048 log-loss.

Слабое место: на старте окна в train всего ~120 матчей с ценами, и любая
калибровка с числом параметров > 0 добавляет больше дисперсии, чем снимает
смещения. Ситуация может измениться, когда наберётся 2-3 сезона с одним
и тем же знаком смещения -- тогда CV сама включит калибровку.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from markets import DEVIG

LAM_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1e6)   # 1e6 == калибровки нет
K_FOLDS = 8
EPS = 1e-9

# диагностика последней подгонки: выбранный lam, T, a, размер train
LAST = {}


def _devig(df):
    """(n, 3) вероятностей из открытия; NaN, если цен нет."""
    out = np.full((len(df), 3), np.nan)
    for i, (_, r) in enumerate(df.iterrows()):
        o = [r.open_H, r.open_D, r.open_A]
        if any(pd.isna(x) for x in o):
            continue
        q = np.asarray(DEVIG['power'](o), float)
        if not np.any(np.isnan(q)):
            out[i] = q
    return out


def _outcome(df):
    return np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))


def _apply(q, T, a):
    """p_k ∝ q_k^T * exp(a_k)."""
    z = T * np.log(np.clip(q, EPS, 1.0)) + a
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _fit(q, o, lam):
    """MAP-оценка (ln T, a_X, a_П2) с L2-штрафом lam к нулю (= сырой рынок)."""
    L = np.log(np.clip(q, EPS, 1.0))
    rows = np.arange(len(o))

    def unpack(th):
        return float(np.exp(th[0])), np.array([0.0, th[1], th[2]])

    def nll(th):
        T, a = unpack(th)
        z = T * L + a
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -lp[rows, o].sum() + lam * float(th @ th)

    r = minimize(nll, np.zeros(3), method='L-BFGS-B')
    return unpack(r.x)


def _choose_lam(q, o):
    """K-fold CV на train: складки чередуются по времени, чтобы каждая
    покрывала все сезоны. Минимизируем суммарный out-of-fold log-loss."""
    n = len(o)
    idx = np.arange(n) % K_FOLDS
    rows = np.arange(n)
    best = None
    for lam in LAM_GRID:
        P = np.zeros((n, 3))
        for k in range(K_FOLDS):
            tr, te = idx != k, idx == k
            T, a = _fit(q[tr], o[tr], lam)
            P[te] = _apply(q[te], T, a)
        ll = float(-np.log(np.clip(P[rows, o], EPS, 1.0)).sum())
        if best is None or ll < best[0]:
            best = (ll, lam)
    return best[1]


def predict(train, test):
    tr = train.dropna(subset=['open_H', 'open_D', 'open_A', 'hg', 'ag'])
    q_tr = _devig(tr)
    ok = ~np.isnan(q_tr[:, 0])
    q_tr, o_tr = q_tr[ok], _outcome(tr)[ok]

    q_te = _devig(test)
    no_price = np.isnan(q_te[:, 0])
    q_te = np.nan_to_num(q_te, nan=1.0 / 3)

    if len(o_tr) < 3 * K_FOLDS:
        T, a, lam = 1.0, np.zeros(3), np.inf        # мало данных -- сырой рынок
    else:
        lam = _choose_lam(q_tr, o_tr)
        T, a = _fit(q_tr, o_tr, lam)
    LAST.update(lam=lam, T=T, a=a, n_train=int(len(o_tr)))

    P = _apply(q_te, T, a)
    P[no_price] = 1.0 / 3                          # без открытия -- нет прогноза
    return P
