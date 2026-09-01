# -*- coding: utf-8 -*-
"""
ТЕСТ МЕТОДА НА РПЛ — той самой лиге, что в ролике.

Данные: football-data.co.uk, RUS.csv — 3400+ матчей РПЛ с 2012/13.
В каждой строке есть ЗАКРЫВАЮЩИЕ коэффициенты:
    PSCH/PSCD/PSCA  — Pinnacle (острая контора, эталон)
    MaxCH/MaxCD/MaxCA — максимум по всем букмекерам (лучшая доступная цена)
    AvgCH/AvgCD/AvgCA — среднее по рынку

Тест 1. Работает ли вообще метод «мягкая контора против острой»:
        считаем справедливые вероятности по Pinnacle, ставим там, где
        лучшая цена рынка выше справедливой, меряем фактический ROI.

Тест 2. Насколько часто такой валуй встречается — то есть должна ли
        нормальная модель находить ставки «каждый тур».
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG

pd.set_option('display.width', 220)
SCRATCH = os.environ.get('SCRATCH', '.')


def load(path):
    d = pd.read_csv(path, encoding='utf-8-sig')
    need = ['PSCH', 'PSCD', 'PSCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'Res']
    d = d.dropna(subset=need)
    d = d[(d[['PSCH', 'PSCD', 'PSCA']] > 1.0).all(axis=1)]
    return d.reset_index(drop=True)


def run(d, devig='power', price_cols=('MaxCH', 'MaxCD', 'MaxCA'), label='лучшая цена рынка'):
    P = d[['PSCH', 'PSCD', 'PSCA']].values
    B = d[list(price_cols)].values
    y = d['Res'].map({'H': 0, 'D': 1, 'A': 2}).values

    fair = np.array([DEVIG[devig](list(row)) for row in P])
    ev = fair * B - 1.0

    pin_margin = (1 / P).sum(axis=1) - 1
    soft_margin = (1 / B).sum(axis=1) - 1
    print(f'\nматчей: {len(d)} | маржа Pinnacle {100*pin_margin.mean():.2f}% | '
          f'маржа «{label}» {100*soft_margin.mean():.2f}%')

    rows = []
    for th in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        m = ev > th
        n = int(m.sum())
        if n == 0:
            rows.append(dict(порог=th, ставок=0, доля='0%', ожидание=np.nan, ROI=np.nan)); continue
        won = (np.arange(3)[None, :] == y[:, None]) & m
        pnl = float((B[won] - 1).sum() - (m.sum() - won.sum()))
        rows.append(dict(порог=f'{th:.0%}', ставок=n, доля=f'{n/(3*len(d)):.1%}',
                         ожидание=f'{100*ev[m].mean():+.2f}%',
                         ROI=f'{100*pnl/n:+.2f}%',
                         ст_ошибка=f'±{100*1.96*np.sqrt((B[m].mean()-1))/np.sqrt(n):.2f}%'))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, 'RUS.csv')
    d = load(path)
    print('=' * 100)
    print(f'ЛИГА: {d.League.iloc[0]}, {d.Country.iloc[0]} | сезоны '
          f'{d.Season.min()} — {d.Season.max()}')
    print('=' * 100)

    print('\n### ТЕСТ 1. Ставим ЛУЧШУЮ цену рынка там, где она выше справедливой по Pinnacle')
    print(run(d, 'power', ('MaxCH', 'MaxCD', 'MaxCA'), 'максимум по рынку').to_string(index=False))

    print('\n### ТЕСТ 2. То же, но ставим СРЕДНЮЮ цену рынка (реалистичнее для одной конторы)')
    print(run(d, 'power', ('AvgCH', 'AvgCD', 'AvgCA'), 'среднее по рынку').to_string(index=False))

    print('\n### ТЕСТ 3. Чувствительность к методу снятия маржи (порог 0%, лучшая цена)')
    P = d[['PSCH', 'PSCD', 'PSCA']].values
    B = d[['MaxCH', 'MaxCD', 'MaxCA']].values
    y = d['Res'].map({'H': 0, 'D': 1, 'A': 2}).values
    out = []
    for nm in ('mult', 'add', 'power', 'shin', 'oddsratio'):
        fair = np.array([DEVIG[nm](list(r)) for r in P])
        ev = fair * B - 1.0
        m = ev > 0
        n = int(m.sum())
        won = (np.arange(3)[None, :] == y[:, None]) & m
        pnl = float((B[won] - 1).sum() - (m.sum() - won.sum()))
        out.append(dict(метод=nm, ставок=n, доля_исходов=f'{n/(3*len(d)):.1%}',
                        ожидание=f'{100*ev[m].mean():+.2f}%', ROI=f'{100*pnl/n:+.2f}%'))
    print(pd.DataFrame(out).to_string(index=False))

    print('\n### ТЕСТ 4. Насколько часто вообще бывает валуй (порог 2%, лучшая цена)')
    fair = np.array([DEVIG['power'](list(r)) for r in P])
    ev = fair * B - 1.0
    m = ev > 0.02
    per_match = m.sum(axis=1)
    print(f'  матчей с хотя бы одним валуйным исходом: {(per_match>0).mean():.1%}')
    print(f'  среднее число валуйных исходов на матч: {per_match.mean():.2f}')
    print(f'  средний перевес там, где он есть: {100*ev[m].mean():+.2f}%')
    print(f'  распределение по исходам: 1={m[:,0].sum()}  X={m[:,1].sum()}  2={m[:,2].sum()}')
