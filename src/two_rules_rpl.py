# -*- coding: utf-8 -*-
"""
Два правила отбора на одних и тех же матчах РПЛ.

ПРАВИЛО 1 (как в роликах): ставим, если вероятность модели выше подразумеваемой
                           из кэфа букмекера на 3-12 п.п.
ПРАВИЛО 2 (проверка острой линией): дополнительно требуем, чтобы кэф букмекера
                           был выше справедливого по закрытию Pinnacle.

Мягкая контора -- средняя цена рынка (AvgC*), острая -- Pinnacle (PSC*).
Обе цены закрывающие, из football-data.co.uk.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

R = pd.read_csv(os.path.join(ROOT, 'data', 'rpl', 'walkforward.csv'))
R = R.dropna(subset=['PSCH', 'PSCD', 'PSCA', 'AvgCH', 'AvgCD', 'AvgCA'])
P = R[['pH', 'pD', 'pA']].values          # вероятности модели
B = R[['AvgCH', 'AvgCD', 'AvgCA']].values  # цена мягкой конторы
S = R[['PSCH', 'PSCD', 'PSCA']].values     # цена Pinnacle
y = R.out.values.astype(int)
fair = np.array([DEVIG['power'](list(r)) for r in S])   # справедливые вероятности
print(f'матчей РПЛ с обеими линиями: {len(R)}')
print(f'маржа мягкой конторы {100*((1/B).sum(1).mean()-1):.2f}%, '
      f'Pinnacle {100*((1/S).sum(1).mean()-1):.2f}%')

edge_pp = P - 1.0 / B                       # перевес в п.п., как в роликах
ev_pin = fair * B - 1.0                     # выгодно ли против острой линии

def run(mask, name):
    n = int(mask.sum())
    if n == 0:
        print(f'{name:52s} ставок 0')
        return
    won = (np.arange(3)[None, :] == y[:, None]) & mask
    ret = np.where(won[mask], B[mask] - 1.0, -1.0)
    se = ret.std(ddof=1) / np.sqrt(n)
    print(f'{name:52s} ставок {n:4d} | на матч {n/len(R):.2f} | '
          f'ROI {100*ret.mean():+6.2f}% ± {100*1.96*se:.1f}% | ср.кэф {B[mask].mean():.2f}')

r1 = (edge_pp >= 0.03) & (edge_pp <= 0.12) & (B >= 1.35) & (B <= 7.0)
r2 = r1 & (ev_pin > 0.01)
run(r1, 'ПРАВИЛО 1: только перевес модели (как в роликах)')
run(r2, 'ПРАВИЛО 2: + проверка справедливой ценой Pinnacle')
print()
print(f'Правило 1 отбирает {100*r1.sum()/(3*len(R)):.1f}% всех исходов, '
      f'правило 2 — {100*r2.sum()/(3*len(R)):.1f}%.')
print(f'В туре из 8 матчей это примерно {8*r1.sum()/len(R):.1f} ставки против '
      f'{8*r2.sum()/len(R):.1f}.')
