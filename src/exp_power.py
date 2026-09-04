# -*- coding: utf-8 -*-
"""
МОЩНОСТЬ ПРОВЕРКИ ФАКТОРОВ.

«Фактор ничего не дал» и «выборка не увидела бы и слона» -- разные утверждения,
и путать их нельзя. exp_factors.py показывает первое (прирост точности прогноза
неотличим от нуля). Здесь считается второе: какой РАЗМЕР эффекта эта выборка
вообще способна отличить от нуля.

Метод. Берём силу команд из уже подогнанной модели как смещение (offset) и
поверх него гоняем пуассоновскую регрессию числа голов на признак. Коэффициент
при признаке -- это ровно тот остаточный эффект, которого нет в рейтинге по xG.
Стандартная ошибка берётся из наблюдённой информационной матрицы, поэтому
доверительный интервал честный, а не выведенный из разности RPS.

    python src/exp_power.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from model import DixonColes, detect_newcomers
from predict import best_params
from exp_factors import load

OUT = os.path.join(ROOT, 'data', 'exp_power.csv')

# (имя, колонка хозяев, колонка гостей, единица измерения, шаг для пересчёта)
TEAM_FACTORS = [
    ('отдых',            'rest_h',   'rest_a',   'день',      1.0),
    ('плотность',        'cong_h',   'cong_a',   'матч/21д',  1.0),
    ('переезд',          'travel_h', 'travel_a', '100 км',    1.0),
    ('обычная 11',       'core_h',   'core_a',   '10% состава', 0.1),
    ('ротация',          'rot_h',    'rot_a',    'новое лицо', 1.0),
]
# признаки матча: действуют на обе команды сразу, то есть на тотал
MATCH_FACTORS = [
    ('жара',      'heat',    '10 °C'),
    ('температура', 'temp',  '10 °C'),
    ('вечер',     'evening', 'вечерний матч'),
    ('Рамадан',   'ramadan', 'матч в пост'),
]


def poisson_fit(y, X, offset):
    """
    Ньютон-Рафсон для пуассоновской регрессии со смещением.
    -> (beta, se). Своя реализация: statsmodels в окружении нет,
    а задача на 10 строк.
    """
    b = np.zeros(X.shape[1])
    for _ in range(60):
        eta = offset + X @ b
        mu = np.exp(np.clip(eta, -8, 4))
        g = X.T @ (y - mu)
        H = (X * mu[:, None]).T @ X
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = offset + X @ b
    mu = np.exp(np.clip(eta, -8, 4))
    H = (X * mu[:, None]).T @ X
    cov = np.linalg.pinv(H)
    return b, np.sqrt(np.diag(cov))


def main():
    d = load()
    p = dict(best_params())
    played = d[d.played.fillna(False) & d.hg.notna()].copy()
    nc = detect_newcomers(d)
    allnew = set()
    for s in nc.values():
        allnew |= s
    m = DixonColes(**p).fit(played, newcomers=allnew)

    # ожидаемые голы по силе команд -- то, что модель уже знает
    lh, la = [], []
    ok = []
    for _, r in played.iterrows():
        try:
            a, b = m.lambdas(r.home, r.away)
        except KeyError:
            a = b = np.nan
        lh.append(a); la.append(b); ok.append(not np.isnan(a))
    played['lh'], played['la'] = lh, la
    played = played[np.array(ok)]
    print(f'матчей в оценке: {len(played)}')
    print(f'база модели: ожидаемый тотал {played.lh.mean() + played.la.mean():.2f}, '
          f'фактический {(played.hg + played.ag).mean():.2f}')
    print()

    rows = []

    # ---- командные признаки: каждая команда со своим значением -----------
    # Наблюдение = команда в матче, поэтому строк вдвое больше.
    for name, ch, ca, unit, step in TEAM_FACTORS:
        sub = played.dropna(subset=[ch, ca])
        if len(sub) < 50:
            print(f'{name}: мало данных ({len(sub)})'); continue
        y = np.concatenate([sub.hg.values, sub.ag.values]).astype(float)
        off = np.log(np.concatenate([sub.lh.values, sub.la.values]))
        x = np.concatenate([sub[ch].values, sub[ca].values]).astype(float)
        x = x - x.mean()
        X = np.column_stack([np.ones(len(y)), x])
        b, se = poisson_fit(y, X, off)
        # эффект в голах за матч на один шаг признака
        base = float(np.exp(off).mean())
        eff = base * (np.exp(b[1] * step) - 1.0)
        lo = base * (np.exp((b[1] - 1.96 * se[1]) * step) - 1.0)
        hi = base * (np.exp((b[1] + 1.96 * se[1]) * step) - 1.0)
        rows.append(dict(фактор=name, вид='команда', ед=unit, n=len(y),
                         эффект=eff, низ=min(lo, hi), верх=max(lo, hi),
                         t=b[1] / se[1]))

    # ---- признаки матча: обе команды сразу, эффект на тотал --------------
    for name, col, unit in MATCH_FACTORS:
        sub = played.dropna(subset=[col])
        if len(sub) < 50:
            print(f'{name}: мало данных ({len(sub)})'); continue
        y = np.concatenate([sub.hg.values, sub.ag.values]).astype(float)
        off = np.log(np.concatenate([sub.lh.values, sub.la.values]))
        v = sub[col].values.astype(float)
        x = np.concatenate([v, v])
        x = x - x.mean()
        X = np.column_stack([np.ones(len(y)), x])
        b, se = poisson_fit(y, X, off)
        base = float(np.exp(off).mean()) * 2.0        # тотал = обе команды
        eff = base * (np.exp(b[1]) - 1.0)
        lo = base * (np.exp(b[1] - 1.96 * se[1]) - 1.0)
        hi = base * (np.exp(b[1] + 1.96 * se[1]) - 1.0)
        rows.append(dict(фактор=name, вид='тотал', ед=unit, n=len(y),
                         эффект=eff, низ=min(lo, hi), верх=max(lo, hi),
                         t=b[1] / se[1]))

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('%-13s %-9s %-14s %5s %9s %20s %7s' %
          ('фактор', 'вид', 'на единицу', 'n', 'эффект', '95%% интервал', 't'))
    for _, r in R.iterrows():
        print('%-13s %-9s %-14s %5d %+9.4f  [%+7.4f .. %+7.4f] %+6.2f' %
              (r['фактор'], r['вид'], r['ед'], r['n'], r['эффект'],
               r['низ'], r['верх'], r['t']))
    print()
    print('Эффект -- изменение числа голов на ОДИН шаг признака.')
    print('Ширина интервала и есть предел различимости: всё, что уже него,')
    print('эта выборка отличить от нуля не может в принципе.')
    print(f'сохранено: {OUT}')


if __name__ == '__main__':
    main()
