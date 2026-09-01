# -*- coding: utf-8 -*-
"""
Гипотеза: объём ударов стабильнее, чем их качество.

xG за матч = (число ударов) × (средний xG на удар). Первый множитель — устойчивая
характеристика команды, второй сильно шумит на малой выборке. Значит, качество
ударов стоит сжимать к среднему по лиге, а объём оставлять как есть:

    xg_смешанный = (1-w) * xg_с_игры  +  w * (удары_с_игры * средний_xG_на_удар_по_лиге)

w = 0   -> обычный xG с игры
w = 1   -> чистый объём ударов, качество полностью сжато к среднему
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, summarize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT,'data','matches.csv'))
# удары с игры = все удары минус пенальти
for s in ('h','a'):
    df[f'{s}_shots_open'] = (df[f'{s}_shots_n'].fillna(0) - df[f'{s}_pens'].fillna(0)).clip(lower=0)
tot_xg = (df.h_xg_open.sum() + df.a_xg_open.sum())
tot_sh = (df.h_shots_open.sum() + df.a_shots_open.sum())
xgps = tot_xg / max(tot_sh, 1)
print(f'средний xG на удар с игры по лиге: {xgps:.4f} '
      f'(всего {tot_sh:.0f} ударов, {tot_xg:.1f} xG)')

base = dict(half_life=150.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25)
rows=[]
for w in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
    for s in ('h','a'):
        df[f'{s}_mix'] = (1-w)*df[f'{s}_xg_open'] + w*(df[f'{s}_shots_open']*xgps)
    p = dict(base); p['xg_cols'] = ('h_mix','a_mix')
    r = walk_forward(df, p)
    su = summarize(r)
    rows.append(dict(w_объём=w, N=su['n'], RPS=round(su['rps'],5), logloss=round(su['logloss'],5)))
    print(rows[-1])
R = pd.DataFrame(rows).sort_values('RPS')
print()
print(R.to_string(index=False))
print(f"\nлучший вес объёма: {R.iloc[0]['w_объём']}, RPS {R.iloc[0]['RPS']:.5f} "
      f"против {R[R.w_объём==0.0].iloc[0]['RPS']:.5f} при чистом xG")
