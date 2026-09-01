# -*- coding: utf-8 -*-
"""
Перевод матрицы счёта в вероятности рынков + снятие маржи букмекера.
"""
import numpy as np
from scipy.optimize import brentq

# =====================================================================
#                      РЫНКИ ИЗ МАТРИЦЫ СЧЁТА
# =====================================================================
def _grid(M):
    n = M.shape[0]
    x = np.arange(n)[:, None] * np.ones((1, n))
    y = np.ones((n, 1)) * np.arange(n)[None, :]
    return x, y


def wdl(M):
    x, y = _grid(M)
    return dict(H=float(M[x > y].sum()), D=float(M[x == y].sum()), A=float(M[x < y].sum()))


def double_chance(M):
    p = wdl(M)
    return {'1X': p['H'] + p['D'], '12': p['H'] + p['A'], 'X2': p['D'] + p['A']}


def totals(M, line):
    """Тотал матча. Возвращает (P(больше), P(меньше), P(пуш))."""
    x, y = _grid(M)
    t = x + y
    over = float(M[t > line].sum())
    under = float(M[t < line].sum())
    push = float(M[t == line].sum())
    return over, under, push


def team_total(M, side, line):
    x, y = _grid(M)
    t = x if side == 'home' else y
    return float(M[t > line].sum()), float(M[t < line].sum()), float(M[t == line].sum())


def btts(M):
    x, y = _grid(M)
    yes = float(M[(x > 0) & (y > 0)].sum())
    return yes, 1 - yes


def odd_even(M):
    x, y = _grid(M)
    t = x + y
    odd = float(M[t % 2 == 1].sum())
    return odd, 1 - odd


def _ah_simple(M, line, side='home'):
    """Азиатская фора для целой/половинной линии. -> (win, push, lose)."""
    x, y = _grid(M)
    d = (x - y) if side == 'home' else (y - x)
    v = d + line
    return float(M[v > 0].sum()), float(M[v == 0].sum()), float(M[v < 0].sum())


def asian_handicap(M, line, side='home'):
    """Азиатская фора, включая четвертные линии (-0.25, -0.75, ...)."""
    frac = round(abs(line * 4)) % 4
    if frac in (1, 3):                       # четвертная линия -> две половинки
        lo, hi = line - 0.25, line + 0.25
        a = _ah_simple(M, lo, side)
        b = _ah_simple(M, hi, side)
        return tuple(0.5 * (np.array(a) + np.array(b)))
    return _ah_simple(M, line, side)


def asian_total(M, line, over=True):
    """Азиатский тотал, включая четвертные линии. -> (win, push, lose)."""
    frac = round(abs(line * 4)) % 4
    if frac in (1, 3):
        parts = [asian_total(M, line - 0.25, over), asian_total(M, line + 0.25, over)]
        return tuple(0.5 * (np.array(parts[0]) + np.array(parts[1])))
    o, u, p = totals(M, line)
    return (o, p, u) if over else (u, p, o)


def fair_odds_asian(win, push, lose):
    """Справедливый кэф для ставки с возвратом: d = 1 + lose/win."""
    return float('inf') if win <= 0 else 1.0 + lose / win


def correct_score(M, top=15):
    n = M.shape[0]
    out = [((i, j), float(M[i, j])) for i in range(n) for j in range(n)]
    out.sort(key=lambda z: -z[1])
    return out[:top]


def interval_goals(M, lo, hi):
    """P(количество голов в матче в диапазоне [lo, hi])."""
    x, y = _grid(M)
    t = x + y
    return float(M[(t >= lo) & (t <= hi)].sum())


# =====================================================================
#                        СНЯТИЕ МАРЖИ
# =====================================================================
def devig_multiplicative(odds):
    inv = np.array([1.0 / o for o in odds])
    return inv / inv.sum()


def devig_additive(odds):
    inv = np.array([1.0 / o for o in odds])
    return np.clip(inv - (inv.sum() - 1.0) / len(inv), 1e-9, None)


def devig_power(odds):
    """p_i = (1/o_i)^k, k подбирается так, чтобы sum(p)=1."""
    inv = np.array([1.0 / o for o in odds])
    f = lambda k: np.sum(inv ** k) - 1.0
    try:
        k = brentq(f, 0.5, 3.0)
    except ValueError:
        return devig_multiplicative(odds)
    return inv ** k


def devig_shin(odds):
    """
    Модель Шина: доля инсайдеров z.
    p_i = ( sqrt(z^2 + 4(1-z) * inv_i^2 / S) - z ) / (2(1-z)),  S = sum(inv)
    z подбирается так, чтобы sum(p_i) = 1.
    """
    inv = np.array([1.0 / o for o in odds])
    S = inv.sum()
    if S <= 1.0:
        return inv / S

    def total(z):
        p = (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / S) - z) / (2 * (1 - z))
        return p.sum() - 1.0

    try:
        z = brentq(total, 1e-9, 0.6)
    except ValueError:
        return devig_multiplicative(odds)
    p = (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / S) - z) / (2 * (1 - z))
    return p / p.sum()


def devig_odds_ratio(odds):
    """Метод отношения шансов (Cheung): OR = p/(1-p) / (q/(1-q))."""
    inv = np.array([1.0 / o for o in odds])

    def total(c):
        p = inv / (c + inv - c * inv)
        return p.sum() - 1.0

    try:
        c = brentq(total, 0.01, 100.0)
    except ValueError:
        return devig_multiplicative(odds)
    p = inv / (c + inv - c * inv)
    return p / p.sum()


DEVIG = dict(mult=devig_multiplicative, add=devig_additive, power=devig_power,
             shin=devig_shin, oddsratio=devig_odds_ratio)


def margin(odds):
    return sum(1.0 / o for o in odds) - 1.0


# =====================================================================
#                        ОЦЕНКА ВАЛУЯ
# =====================================================================
def edge(p_model, price):
    """Матожидание на 1 единицу ставки."""
    return p_model * price - 1.0


def kelly(p_model, price, frac=0.25, cap=0.02, p_push=0.0):
    """
    Доля банка по Келли с учётом возврата.

    Для ставки с исходами (выигрыш w, возврат p, проигрыш l) оптимум
    w*log(1+f*b) + l*log(1-f) достигается при f = (w*b - l) / (b*(w+l)),
    что равно обычной формуле Келли от УСЛОВНОЙ вероятности w/(w+l).
    Без этой поправки ставка на азиатскую или целую линию занижается
    в разы: при w=0.40, push=0.20, кэфе 2.60 верный Келли 0.1875,
    а формула без возврата даёт 0.0250.
    """
    b = price - 1.0
    if b <= 0:
        return 0.0
    p = p_model / max(1.0 - p_push, 1e-9) if p_push else p_model
    p = float(np.clip(p, 0.0, 1.0))
    f = (p * b - (1 - p)) / b
    return float(np.clip(f * frac, 0.0, cap))
