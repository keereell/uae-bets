# -*- coding: utf-8 -*-
"""
ЧЕСТНАЯ ВАЛИДАЦИЯ: подбор гиперпараметров и отчётная метрика на РАЗНЫХ отрезках.

Проблема исходной схемы: 240 конфигураций сравнивались на тех же 250 матчах,
на которых потом приводился итоговый RPS. Минимум по 240 вариантам смещён вниз
даже при полном отсутствии сигнала.

Здесь:
  ПОДБОР   на матчах 2025-02-01 .. 2025-12-31
  ОТЧЁТ    на матчах 2026-01-01 .. сегодня (эти данные при подборе не видели)
"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, summarize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT,'data','matches.csv'))
TUNE = ('2025-02-01', '2025-12-31')
TEST = ('2026-01-01', None)

grid = dict(half_life=[110,150,220], alpha=[0.15,0.30,0.60],
            w_goals=[0.0,0.2], promoted_shift=[0.0,0.25])
XG = [('h_xg_open','a_xg_open'), ('h_npxg','a_npxg'), ('h_xg','a_xg')]

rows=[]
for combo in itertools.product(*grid.values()):
    for xg in XG:
        p = dict(zip(grid.keys(), combo)); p['xg_cols']=xg
        r = walk_forward(df, p, start_date=TUNE[0], end_date=TUNE[1])
        s = summarize(r)
        rows.append({**{k:v for k,v in p.items() if k!='xg_cols'},
                     'xg': xg[0].replace('h_',''), 'n_tune': s['n'], 'rps_tune': s['rps']})
T = pd.DataFrame(rows).sort_values('rps_tune')
print('=== ПОДБОР (окно 2025-02-01 .. 2025-12-31) ===')
print(T.head(8).round(5).to_string(index=False))

best = T.iloc[0]
bp = dict(half_life=float(best.half_life), alpha=float(best.alpha),
          w_goals=float(best.w_goals), promoted_shift=float(best.promoted_shift),
          xg_cols=(f'h_{best.xg}', f'a_{best.xg}'))
print('\nвыбрано:', bp)

print('\n=== ОТЧЁТ на невиданных данных (2026-01-01 и позже) ===')
r = walk_forward(df, bp, start_date=TEST[0])
s = summarize(r)
print(f"  матчей: {s['n']}")
print(f"  модель : RPS {s['rps']:.5f}  log-loss {s['logloss']:.5f}")
if 'rps_book' in s:
    print(f"  Bet365 : RPS {s['rps_book']:.5f}  log-loss {s['logloss_book']:.5f}  (N={s['n_book']})")
    print(f"  разница: {s['rps']-s['rps_book']:+.5f} не в пользу модели"
          if s['rps']>s['rps_book'] else "  модель лучше линии")

# для сравнения -- метрика тех же параметров на окне подбора
r2 = walk_forward(df, bp, start_date=TUNE[0], end_date=TUNE[1])
s2 = summarize(r2)
print(f"\n  на окне подбора эти же параметры давали RPS {s2['rps']:.5f} (N={s2['n']})")
print(f"  оптимистическое смещение подбора: {s2['rps']-s['rps']:+.5f}")
