# -*- coding: utf-8 -*-
"""
ДИКСОН-КОУЛЗ + КАЛИБРОВОЧНЫЙ СЛОЙ.

Самая дешёвая проверка гипотезы «модель права по ранжированию, но
разкалибрована»: берём действующую DC на xG (eval_protocol.predict_dixon_coles)
как есть и поверх неё учим монотонное (или почти монотонное) преобразование
вероятностей. Ничего нового о командах слой не знает -- он лишь исправляет
систематику: на этой лиге все методы НЕДООЦЕНИВАЮТ сильных фаворитов, и
если DC этим страдает, заострение p^(1/T) с T < 1 должно помочь.

Два калибратора, оба на log-вероятностях DC (для трёх исходов
softmax(log p) = p, поэтому «логит» здесь -- это log p):

  temp   -- температурное масштабирование, ОДИН параметр s = 1/T:
                p_k  ∝  p_dc,k ^ s
            s > 1 заостряет, s < 1 сглаживает. Монотонно, ранжирование
            не трогает. Именно этот параметр отвечает на вопрос
            «недооценивает ли DC фаворитов».
  logit  -- мультиномиальная логит-регрессия по классам (калибровка
            Дирихле, Kull et al. 2019):
                p_k  ∝  exp( sum_j W[k, j] * log p_dc,j  +  b_k )
            W 3x3 и b -- 12 чисел, из них идентифицируемо 8. Умеет
            двигать классы по-разному (лишняя/недостающая ничья,
            асимметрия дом/гости), но на сотнях пар это уже риск переподгонки,
            поэтому W сжимается к единичной матрице, b -- к нулю.

ОТКУДА ПАРЫ (p_dc, исход). Только из train, внутри predict, внутренним
walk-forward по игровым дням train: DC подгоняется на матчах строго ДО дня
d и прогнозирует d. По in-sample прогнозу DC калибровать нельзя -- на своих
же матчах модель переуверена, и температура вышла бы «сглаживающей», хотя в
бою нужно обратное. Прогноз DC на день d зависит только от матчей до d, они
лежат в train любого более позднего дня протокола -- значит, внутренние
прогнозы можно кэшировать между вызовами predict. Это ускорение, не утечка.

ВЫБОР ВАРИАНТА. Тоже только по train и тоже вне выборки: по тем же парам
считается ПРЕКВЕНЦИАЛЬНЫЙ log-loss каждого варианта (raw / temp / logit) --
для каждого внутреннего дня калибратор подгоняется на парах ДО этого дня и
оценивается на его парах. Берём вариант с наименьшей суммой. In-sample
сравнение здесь бессмысленно: 8-параметрический логит всегда «выиграл» бы у
одного параметра. Преквенциальные оценки для дня d зависят только от пар
до d -- кэшируются так же, как прогнозы DC.

КОНСТАНТЫ (из теории, не по сетке).
  * INNER_MIN_TRAIN = 60 -- минимум матчей для внутренней DC (как в
    logit_blend): протокольные 120 оставили бы первый день оценки без пар.
  * MIN_PAIRS = 30 -- меньше пар температуру не оценить (SE > 0.3), возвращаем
    сырую DC. Для логита порог MIN_PAIRS_LOGIT = 80: 8 параметров на 30 парах
    -- шум.
  * RIDGE = 2.0 -- L2-штраф к априору «без калибровки» (W = I, b = 0),
    по силе ~5 матчей информации; на 200+ парах почти незаметен.
  * Диапазон s ∈ [0.25, 4]: вне его калибровка означала бы, что DC не модель.
  * P_FLOOR = 1e-3 -- пол вероятности перед логарифмом.

СЛАБОЕ МЕСТО. Калибровка не добавляет информации: если DC отстаёт от рынка
по ранжированию, а не по масштабу, слой ничего не исправит -- и это тоже
результат. Второе: пар мало (на старте 30-40, к концу ~300), температура на
старте шумная, а внутренняя DC ранних дней сама обучена на 60-100 матчах и
калибрована иначе, чем боевая на 300 -- пары не совсем «те же», что в бою.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from eval_protocol import predict_dixon_coles, outcome

INNER_MIN_TRAIN = 60
MIN_PAIRS = 30
MIN_PAIRS_LOGIT = 80
RIDGE = 2.0
S_BOUNDS = (0.25, 4.0)
P_FLOOR = 1e-3
VARIANTS = ('raw', 'temp', 'logit')

# кэш внутренних прогнозов DC: день -> DataFrame(game_id, date, dc_H, dc_D, dc_A, out)
_DC_CACHE = {}
# кэш прогноза боевой DC на день test (train одинаков для всех вариантов в процессе)
_OUT_CACHE = {}
# кэш преквенциальных оценок: день -> dict(raw=sum ll, temp=..., logit=..., n=...)
_PREQ_CACHE = {}
# история для отчёта: dict(date, n_pairs, s, chosen, preq_raw, preq_temp, preq_logit)
FITS = []


# --------------------------------------------------------------- внутренний walk-forward
def _inner_day(train, day):
    """Предматчевые пары (p_dc, исход) для игрового дня day из train (с кэшем)."""
    if day in _DC_CACHE:
        return _DC_CACHE[day]
    sub_train = train[train.date < day]
    sub_test = train[train.date == day]
    if len(sub_train) < INNER_MIN_TRAIN:
        return None
    try:
        P = np.asarray(predict_dixon_coles(sub_train.copy(), sub_test.copy()), float)
    except Exception as e:                     # день пропускается, не кэшируется
        print(f'  dc_calibrated: внутренняя DC упала на {day}: {e}', file=sys.stderr)
        return None
    outs = outcome(sub_test.hg.values, sub_test.ag.values)
    df = pd.DataFrame(dict(game_id=sub_test.game_id.values, date=day,
                           dc_H=P[:, 0], dc_D=P[:, 1], dc_A=P[:, 2], out=outs.astype(int)))
    _DC_CACHE[day] = df
    return df


def _pairs(train):
    parts = []
    for day in sorted(train.date.unique()):
        df = _inner_day(train, day)
        if df is not None and len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------- калибраторы
def _logp(P):
    return np.log(np.clip(np.asarray(P, float), P_FLOOR, 1.0))


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _nll(P, y):
    return -np.sum(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)))


def apply_temp(L, s):
    """p ∝ p^s, L = log p."""
    return _softmax(s * L)


def fit_temp(L, y):
    """Один параметр s = 1/T, минимум log-loss на отрезке S_BOUNDS."""
    r = minimize_scalar(lambda s: _nll(apply_temp(L, s), y), bounds=S_BOUNDS, method='bounded',
                        options=dict(xatol=1e-5))
    return float(r.x)


def apply_logit(L, theta):
    W, b = theta[:9].reshape(3, 3), theta[9:]
    return _softmax(L @ W.T + b[None, :])


_PRIOR_LOGIT = np.concatenate([np.eye(3).ravel(), np.zeros(3)])     # W = I, b = 0


def fit_logit(L, y):
    """
    theta = (W 3x3, b 3): минимум
        -sum log softmax(W log p + b)[y] + RIDGE * (|W - I|^2 + |b|^2),
    L-BFGS-B с аналитическим градиентом.
    """
    Y = np.eye(3)[y]

    def f(th):
        return _nll(apply_logit(L, th), y) + RIDGE * float(np.sum((th - _PRIOR_LOGIT) ** 2))

    def g(th):
        R = apply_logit(L, th) - Y                   # (n, 3)
        gr = np.empty(12)
        gr[:9] = (R.T @ L).ravel()                   # dW[k, j] = sum_i R[i, k] L[i, j]
        gr[9:] = R.sum(axis=0)
        return gr + 2 * RIDGE * (th - _PRIOR_LOGIT)

    r = minimize(f, _PRIOR_LOGIT.copy(), jac=g, method='L-BFGS-B',
                 options=dict(maxiter=500, ftol=1e-12))
    return r.x


def _fit_all(pairs):
    """Подгонка обоих калибраторов на всех парах; None -- если пар мало."""
    L = _logp(pairs[['dc_H', 'dc_D', 'dc_A']].values)
    y = pairs.out.values.astype(int)
    s = fit_temp(L, y) if len(pairs) >= MIN_PAIRS else None
    theta = fit_logit(L, y) if len(pairs) >= MIN_PAIRS_LOGIT else None
    return s, theta


def _calibrate(P, variant, s, theta):
    """Применить вариант к матрице вероятностей P; при нехватке пар -- деградация к простому."""
    L = _logp(P)
    if variant == 'logit' and theta is not None:
        return apply_logit(L, theta)
    if variant in ('logit', 'temp') and s is not None:
        return apply_temp(L, s)
    return np.asarray(P, float)


# --------------------------------------------------------------- преквенциальный выбор
def _prequential(pairs):
    """
    Сумма log-loss каждого варианта по парам, где калибратор для дня d
    подогнан только на парах до d. -> dict(raw, temp, logit, n).
    """
    tot = dict(raw=0.0, temp=0.0, logit=0.0, n=0)
    days = sorted(pairs.date.unique())
    for day in days:
        if day not in _PREQ_CACHE:
            before = pairs[pairs.date < day]
            if len(before) < MIN_PAIRS:
                continue
            here = pairs[pairs.date == day]
            P = here[['dc_H', 'dc_D', 'dc_A']].values
            y = here.out.values.astype(int)
            s, theta = _fit_all(before)
            _PREQ_CACHE[day] = dict(
                raw=_nll(np.clip(P, P_FLOOR, 1.0) / np.clip(P, P_FLOOR, 1.0).sum(axis=1, keepdims=True), y),
                temp=_nll(_calibrate(P, 'temp', s, theta), y),
                logit=_nll(_calibrate(P, 'logit', s, theta), y),
                n=len(here))
        c = _PREQ_CACHE[day]
        for k in VARIANTS:
            tot[k] += c[k]
        tot['n'] += c['n']
    return tot


# --------------------------------------------------------------- predict
def _predict_variant(train, test, variant):
    day = str(test.date.min())
    if day not in _OUT_CACHE:
        _OUT_CACHE[day] = np.asarray(predict_dixon_coles(train, test), float)
    Pdc = _OUT_CACHE[day]

    pairs = _pairs(train)
    if len(pairs) < MIN_PAIRS:
        FITS.append(dict(date=day, n_pairs=int(len(pairs)), s=np.nan, chosen='raw',
                         preq_raw=np.nan, preq_temp=np.nan, preq_logit=np.nan, preq_n=0))
        return Pdc
    s, theta = _fit_all(pairs)

    chosen = variant
    preq = dict(raw=np.nan, temp=np.nan, logit=np.nan, n=0)
    if variant == 'best':
        preq = _prequential(pairs)
        if preq['n'] > 0:
            chosen = min(VARIANTS, key=lambda k: preq[k])
        else:
            chosen = 'temp'                          # оценок ещё нет -- один параметр безопаснее
    FITS.append(dict(date=day, n_pairs=int(len(pairs)), s=s, chosen=chosen,
                     preq_raw=preq['raw'], preq_temp=preq['temp'], preq_logit=preq['logit'],
                     preq_n=preq['n'], theta=theta))
    return _calibrate(Pdc, chosen, s, theta)


def predict_temp(train, test):
    """DC + температурное масштабирование."""
    return _predict_variant(train, test, 'temp')


def predict_logit(train, test):
    """DC + мультиномиальный логит по классам (при нехватке пар -- температура)."""
    return _predict_variant(train, test, 'logit')


def predict(train, test):
    """DC + лучший по преквенциальному log-loss на train вариант (raw / temp / logit)."""
    return _predict_variant(train, test, 'best')


def fits_summary():
    """Сводка подгонок за прогон: температура по дням и частота выбора вариантов."""
    F = pd.DataFrame(FITS)
    if F.empty:
        return 'нет подгонок'
    L = [f'подгонок {len(F)}, выбор вариантов: ' + ', '.join(
        f'{k}={int((F.chosen == k).sum())}' for k in VARIANTS)]
    S = F.dropna(subset=['s'])
    if len(S):
        L.append(f's = 1/T: первая {S.s.iloc[0]:.3f} (пар {int(S.n_pairs.iloc[0])}), '
                 f'медиана {S.s.median():.3f}, последняя {S.s.iloc[-1]:.3f} (пар {int(S.n_pairs.iloc[-1])})')
    Q = F.dropna(subset=['preq_raw'])
    if len(Q):
        q = Q.iloc[-1]
        L.append(f'преквенциальный log-loss на train в последний день (n={int(q.preq_n)}): '
                 f'raw {q.preq_raw / q.preq_n:.5f}  temp {q.preq_temp / q.preq_n:.5f}  '
                 f'logit {q.preq_logit / q.preq_n:.5f}')
    return '\n'.join(L)
