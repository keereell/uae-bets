# -*- coding: utf-8 -*-
"""
Разбор положительного ROI на РПЛ: сигнал это или дисперсия.

Проверки:
  A. Из чего состоит прибыль — распределение P&L, вклад топ-ставок
  B. Бутстрап доверительного интервала ROI
  C. Разбивка по диапазонам коэффициентов и по исходам (1 / X / 2)
  D. Плацебо: та же стратегия, но вероятности модели перемешаны между матчами
  E. Сравнение «максимум по рынку» против «среднее по рынку» —
     сколько прибыли от модели, а сколько от поиска лучшей цены
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)
rng = np.random.default_rng(7)


def bets(P, B, y, th):
    m = (P * B - 1.0) > th
    won = (np.arange(3)[None, :] == y[:, None]) & m
    ret = np.where(won[m], B[m] - 1.0, -1.0)
    return ret, B[m], m


def main():
    R = pd.read_csv(os.path.join(ROOT, 'data', 'rpl', 'walkforward.csv'))
    P = R[['pH', 'pD', 'pA']].values
    y = R.out.values.astype(int)
    MAXB = R[['MaxCH', 'MaxCD', 'MaxCA']].values
    AVGB = R[['AvgCH', 'AvgCD', 'AvgCA']].values
    ok = ~np.isnan(MAXB).any(axis=1) & ~np.isnan(AVGB).any(axis=1)
    P, y, MAXB, AVGB = P[ok], y[ok], MAXB[ok], AVGB[ok]
    print(f'матчей: {len(P)}')

    TH = 0.05
    ret, odds, mask = bets(P, MAXB, y, TH)
    n = len(ret)
    print(f'\n=== A. ИЗ ЧЕГО СОСТОИТ ПРИБЫЛЬ (порог {TH:.0%}, максимум по рынку) ===')
    print(f'  ставок {n}, ROI {100*ret.mean():+.2f}%, суммарно {ret.sum():+.1f} ед.')
    print(f'  выиграло {int((ret>0).sum())} ({100*(ret>0).mean():.1f}%), средний кэф {odds.mean():.2f}')
    top = np.sort(ret)[::-1]
    for k in (1, 3, 5, 10):
        print(f'  топ-{k:2d} ставок дают {top[:k].sum():+.1f} ед. '
              f'= {100*top[:k].sum()/max(ret.sum(),1e-9):.0f}% всей прибыли')
    med = np.median(ret)
    print(f'  медианная ставка: {med:+.2f} ед. (то есть половина ставок проигрывает целиком)')

    print(f'\n=== B. БУТСТРАП ROI (10 000 пересэмплирований) ===')
    boot = np.array([rng.choice(ret, size=n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'  ROI {100*ret.mean():+.2f}%   95% ДИ [{100*lo:+.1f}%, {100*hi:+.1f}%]   '
          f'P(ROI>0) = {(boot>0).mean():.1%}')

    print(f'\n=== C. РАЗБИВКА ПО КЭФАМ И ИСХОДАМ ===')
    rows = []
    for lo_, hi_ in ((1.0, 2.5), (2.5, 4.0), (4.0, 7.0), (7.0, 99)):
        sel = (odds >= lo_) & (odds < hi_)
        if sel.sum() == 0:
            continue
        rows.append(dict(диапазон=f'{lo_}–{hi_}', ставок=int(sel.sum()),
                         попаданий=int((ret[sel] > 0).sum()),
                         ROI=f'{100*ret[sel].mean():+.1f}%',
                         вклад=f'{ret[sel].sum():+.1f}'))
    print(pd.DataFrame(rows).to_string(index=False))
    idx = np.tile(np.arange(3), (len(P), 1))[mask]
    rows = []
    for j, nm in enumerate(('1 (хозяева)', 'X (ничья)', '2 (гости)')):
        sel = idx == j
        if sel.sum() == 0:
            continue
        rows.append(dict(исход=nm, ставок=int(sel.sum()),
                         ROI=f'{100*ret[sel].mean():+.1f}%', вклад=f'{ret[sel].sum():+.1f}'))
    print(pd.DataFrame(rows).to_string(index=False))

    print(f'\n=== D. ПЛАЦЕБО: вероятности модели перемешаны между матчами ===')
    res = []
    for _ in range(400):
        Pp = P[rng.permutation(len(P))]
        r2, _, _ = bets(Pp, MAXB, y, TH)
        if len(r2):
            res.append(r2.mean())
    res = np.array(res)
    print(f'  ROI перемешанной модели: медиана {100*np.median(res):+.2f}%, '
          f'95% диапазон [{100*np.percentile(res,2.5):+.1f}%, {100*np.percentile(res,97.5):+.1f}%]')
    print(f'  доля перемешиваний, которые бьют настоящую модель: '
          f'{(res >= ret.mean()).mean():.1%}')

    print(f'\n=== E. СКОЛЬКО ОТ МОДЕЛИ, А СКОЛЬКО ОТ ПОИСКА ЛУЧШЕЙ ЦЕНЫ ===')
    for nm, B in (('максимум по рынку', MAXB), ('среднее по рынку', AVGB)):
        r2, o2, _ = bets(P, B, y, TH)
        mg = (1 / B).sum(axis=1).mean() - 1
        print(f'  {nm:20s}: маржа {100*mg:+.2f}%  ставок {len(r2):4d}  '
              f'ROI {100*r2.mean():+.2f}%  ср.кэф {o2.mean():.2f}')
    prem = (MAXB / AVGB - 1).mean()
    print(f'  надбавка лучшей цены над средней: {100*prem:+.2f}% к коэффициенту')

    print(f'\n=== F. ТО ЖЕ САМОЕ БЕЗ ЛОНГШОТОВ (кэф <= 4.0) ===')
    for nm, B in (('максимум по рынку', MAXB), ('среднее по рынку', AVGB)):
        m = ((P * B - 1.0) > TH) & (B <= 4.0)
        won = (np.arange(3)[None, :] == y[:, None]) & m
        r3 = np.where(won[m], B[m] - 1.0, -1.0)
        if len(r3) == 0:
            continue
        se = r3.std(ddof=1) / np.sqrt(len(r3))
        print(f'  {nm:20s}: ставок {len(r3):4d}  ROI {100*r3.mean():+.2f}% '
              f'± {100*1.96*se:.1f}%  ср.кэф {B[m].mean():.2f}')


if __name__ == '__main__':
    main()
