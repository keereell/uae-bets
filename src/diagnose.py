# -*- coding: utf-8 -*-
"""
Диагностика: несёт ли модель информацию, которой нет в коэффициентах?

Три теста:
  A. Калибровка по бакетам: там, где модель расходится с рынком,
     кто оказывается прав по факту?
  B. Логит-смешивание: p ~ softmax(w*log p_model + (1-w)*log p_market).
     Если оптимальный w близок к 0 — модель не добавляет ничего к рынку.
  C. ROI стратегии «ставим при перевесе > порога» против Bet365,
     с разбивкой по направлению ставки.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, rps
from predict import best_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    wf = walk_forward(df, best_params())
    b = wf.dropna(subset=['qH']).copy()
    print(f'матчей с коэффициентами Bet365: {len(b)}')

    P = b[['pH', 'pD', 'pA']].values
    Q = b[['qH', 'qD', 'qA']].values
    O = b[['oH', 'oD', 'oA']].values
    y = b['out'].values.astype(int)
    Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1

    # ---------------- A. калибровка расхождений
    print('\n=== A. КТО ПРАВ ТАМ, ГДЕ МОДЕЛЬ РАСХОДИТСЯ С РЫНКОМ ===')
    d = (P - Q).ravel()
    q = Q.ravel(); p = P.ravel(); act = Y.ravel()
    bins = pd.cut(d, [-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1])
    t = pd.DataFrame(dict(d=d, q=q, p=p, act=act, bin=bins))
    g = t.groupby('bin', observed=True).agg(n=('act', 'size'), рынок=('q', 'mean'),
                                            модель=('p', 'mean'), факт=('act', 'mean'))
    g['кто_ближе'] = np.where((g['факт'] - g['рынок']).abs() < (g['факт'] - g['модель']).abs(),
                              'рынок', 'модель')
    print(g.round(4).to_string())

    # ---------------- B. логит-смешивание
    print('\n=== B. СКОЛЬКО ВЕСА ЗАСЛУЖИВАЕТ МОДЕЛЬ РЯДОМ С РЫНКОМ ===')

    def blend_ll(w):
        z = w * np.log(np.clip(P, 1e-9, 1)) + (1 - w) * np.log(np.clip(Q, 1e-9, 1))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); pb = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(pb[np.arange(len(y)), y], 1e-12, 1)))

    r = minimize_scalar(blend_ll, bounds=(-0.5, 1.5), method='bounded')
    w_opt = float(r.x)
    print(f'  оптимальный вес модели w = {w_opt:.3f}')
    print(f'  log-loss: только рынок {blend_ll(0):.4f} | только модель {blend_ll(1):.4f} | '
          f'смесь {blend_ll(w_opt):.4f}')
    for w in (0.0, 0.15, 0.25, 0.5, 0.75, 1.0):
        print(f'    w={w:.2f} -> log-loss {blend_ll(w):.4f}')

    # ---------------- C. ROI
    print('\n=== C. ROI СТРАТЕГИИ ПРОТИВ Bet365 ===')
    rows = []
    for th in (0.0, 0.03, 0.05, 0.08, 0.12, 0.20):
        for lo, hi, lab in ((1.0, 99, 'все кэфы'), (1.0, 2.5, 'кэф<2.5'), (2.5, 5.0, 'кэф 2.5-5'),
                            (5.0, 99, 'кэф>5')):
            n, pnl = 0, 0.0
            for i in range(len(P)):
                for j in range(3):
                    price = O[i, j]
                    if not (lo <= price < hi):
                        continue
                    if P[i, j] * price - 1 > th:
                        n += 1
                        pnl += (price - 1) if y[i] == j else -1
            rows.append(dict(порог=th, диапазон=lab, ставок=n,
                             ROI=(pnl / n if n else np.nan), PnL=round(pnl, 1)))
    R = pd.DataFrame(rows)
    print(R.pivot(index='порог', columns='диапазон', values=['ставок', 'ROI']).round(3).to_string())

    # ---------------- D. проверка: систематическое смещение по фаворитам
    print('\n=== D. ПРОВЕРКА СМЕЩЕНИЯ ПО СИЛЕ ФАВОРИТА ===')
    fav_q = Q.max(axis=1)
    fav_i = Q.argmax(axis=1)
    fav_p = P[np.arange(len(P)), fav_i]
    hit = (y == fav_i).astype(float)
    bb = pd.cut(fav_q, [0.3, 0.45, 0.55, 0.65, 0.75, 1.0])
    tt = pd.DataFrame(dict(q=fav_q, p=fav_p, hit=hit, b=bb))
    print(tt.groupby('b', observed=True).agg(n=('hit', 'size'), рынок=('q', 'mean'),
                                             модель=('p', 'mean'), факт=('hit', 'mean')).round(3).to_string())
    print('\n  (если «факт» стабильно выше «модели» — модель недооценивает фаворитов)')


if __name__ == '__main__':
    main()
