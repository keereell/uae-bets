# -*- coding: utf-8 -*-
"""
ЛОГИТ-БЛЕНД МОДЕЛИ И РЫНКА.

Честная смесь двух прогнозов -- Диксона-Коулза на xG (действующая модель
проекта, eval_protocol.predict_dixon_coles) и открытия Bet365 без маржи:

    p_k = softmax_k( w1 * log p_dc,k  +  w2 * log p_open,k  +  b_k )

Для трёх исходов «логит» -- это log-вероятность: softmax(log p) = p, поэтому
w1 = 0, w2 = 1, b = 0 воспроизводит рынок ровно, w1 = 1, w2 = 0 -- модель.
Общий вес на все три класса (а не по классу) -- чтобы w2 > 1 умел
ЗАОСТРЯТЬ рынок: на этой лиге все методы недооценивают сильных фаворитов,
и w2 > 1 -- ровно тот параметр, который это чинит, не трогая остального.
Смещения b_k ловят систематику по классам (например, лишнюю ничью).

ОТКУДА БЕРУТСЯ ВЕСА. Только из train, внутри predict:
  1. Внутренний walk-forward по игровым дням train: для каждого дня d
     Диксон-Коулз подгоняется на матчах строго ДО d и прогнозирует d.
     Так получаются ПРЕДМАТЧЕВЫЕ пары (p_dc, p_open, исход) -- те же,
     что будут в бою. Подгонять веса по in-sample прогнозу DC нельзя:
     модель на своих же матчах переуверена, и вес ушёл бы к ней.
  2. На парах -- мультиномиальный логит с 5 параметрами (w1, w2, b_H, b_D,
     b_A), L-BFGS-B с аналитическим градиентом, без sklearn.

Прогноз DC для дня d зависит только от матчей до d, все они лежат в train
любого более позднего дня протокола -- поэтому внутренние прогнозы
кэшируются между вызовами predict. Это ускорение, а не утечка.

КОНСТАНТЫ (из теории, не по сетке).
  * INNER_MIN_TRAIN = 60 -- минимум матчей для внутренней подгонки DC.
    Протокольные 120 здесь невозможны: на первый день оценки в train
    ~120 матчей, и пар не было бы вовсе. 60 -- это ~4 тура, ниже DC
    ещё не модель. Ранние пары честно отражают слабую раннюю DC, поэтому
    вес модели на старте занижен -- это плата за честность.
  * RIDGE = 2.0 -- L2-штраф к априору «чистый рынок» (w1=0, w2=1, b=0):
    литература (Pitcan 2026) говорит, что вес структурной модели рядом
    с ценой около нуля, и пока пар мало (первые недели -- 30-40 штук),
    смесь должна опираться на рынок. По силе это эквивалент ~5-10
    матчей информации, на 100+ парах штраф почти незаметен.
  * P_FLOOR = 1e-3 -- пол вероятности перед логарифмом.

СЛАБОЕ МЕСТО. Пар мало (десятки, к концу -- две с половиной сотни), а
подгоняются 5 параметров; SE веса модели порядка 0.2-0.3. Если w1 выйдет
~0 или отрицательным -- это и есть результат: xG-модель не добавляет
к открытию ничего, что рынок не знал бы. Обыграть открытие смесь может
только за счёт заострения (w1 + w2 > 1) и смещений по классам, т.е.
за счёт систематических ошибок рынка, а не новой информации.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eval_protocol import predict_dixon_coles, _market, outcome

INNER_MIN_TRAIN = 60
RIDGE = 2.0
P_FLOOR = 1e-3
PRIOR = np.array([0.0, 1.0, 0.0, 0.0, 0.0])      # w1, w2, b_H, b_D, b_A

# кэш внутренних прогнозов: день -> DataFrame(game_id, dc_*, op_*, out)
_DC_CACHE = {}
# история подгонок для отчёта: список dict(date, n_pairs, w1, w2, b_H, b_D, b_A)
FITS = []


# --------------------------------------------------------------- внутренний walk-forward
def _inner_day(train, day):
    """Предматчевые пары для игрового дня day из train (с кэшем)."""
    if day in _DC_CACHE:
        return _DC_CACHE[day]
    sub_train = train[train.date < day]
    sub_test = train[train.date == day]
    if len(sub_train) < INNER_MIN_TRAIN:
        return None
    try:
        P = np.asarray(predict_dixon_coles(sub_train.copy(), sub_test.copy()), float)
    except Exception as e:                     # день пропускается, не кэшируется
        print(f'  logit_blend: внутренняя DC упала на {day}: {e}', file=sys.stderr)
        return None
    rows = []
    outs = outcome(sub_test.hg.values, sub_test.ag.values)
    for (_, r), p, o in zip(sub_test.iterrows(), P, outs):
        q = _market(r, 'open')
        if q is None:
            continue
        rows.append(dict(game_id=r.game_id, dc_H=p[0], dc_D=p[1], dc_A=p[2],
                         op_H=q[0], op_D=q[1], op_A=q[2], out=int(o)))
    df = pd.DataFrame(rows)
    _DC_CACHE[day] = df
    return df


def _pairs(train):
    parts = []
    for day in sorted(train.date.unique()):
        df = _inner_day(train, day)
        if df is not None and len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------- мультиномиальный логит
def _logp(P):
    return np.log(np.clip(np.asarray(P, float), P_FLOOR, 1.0))


def _scores(theta, Ldc, Lop):
    w1, w2, b = theta[0], theta[1], theta[2:]
    return w1 * Ldc + w2 * Lop + b[None, :]


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_blend(Ldc, Lop, y):
    """
    theta = (w1, w2, b_H, b_D, b_A): минимум
        -sum log softmax(w1*Ldc + w2*Lop + b)[y] + RIDGE * |theta - PRIOR|^2
    """
    Y = np.eye(3)[y]

    def f(th):
        P = _softmax(_scores(th, Ldc, Lop))
        nll = -np.sum(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)))
        return nll + RIDGE * float(np.sum((th - PRIOR) ** 2))

    def g(th):
        P = _softmax(_scores(th, Ldc, Lop))
        R = P - Y                                    # (n, 3)
        gr = np.empty(5)
        gr[0] = np.sum(R * Ldc)
        gr[1] = np.sum(R * Lop)
        gr[2:] = R.sum(axis=0)
        return gr + 2 * RIDGE * (th - PRIOR)

    r = minimize(f, PRIOR.copy(), jac=g, method='L-BFGS-B',
                 options=dict(maxiter=500, ftol=1e-12))
    return r.x


# --------------------------------------------------------------- predict
def predict(train, test):
    # 1. пары (p_dc, p_open, исход) по внутреннему walk-forward на train
    pairs = _pairs(train)
    if len(pairs) >= 10:
        Ldc = _logp(pairs[['dc_H', 'dc_D', 'dc_A']].values)
        Lop = _logp(pairs[['op_H', 'op_D', 'op_A']].values)
        theta = fit_blend(Ldc, Lop, pairs.out.values.astype(int))
    else:
        theta = PRIOR.copy()                        # пар нет -> чистый рынок
    FITS.append(dict(date=str(test.date.min()), n_pairs=int(len(pairs)),
                     w1=theta[0], w2=theta[1], b_H=theta[2], b_D=theta[3], b_A=theta[4]))

    # 2. прогнозы компонентов на test и смесь
    Pdc = np.asarray(predict_dixon_coles(train, test), float)
    out = np.empty((len(test), 3))
    for i, (_, r) in enumerate(test.iterrows()):
        q = _market(r, 'open')
        if q is None:                               # без открытия -- только модель
            out[i] = Pdc[i]
            continue
        z = _scores(theta, _logp(Pdc[i][None, :]), _logp(q[None, :]))
        out[i] = _softmax(z)[0]
    return out
