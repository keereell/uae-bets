# -*- coding: utf-8 -*-
"""
«Фантомный валуй»: воспроизведение популярной методики и её проверка.

Методика (то, что делают в роликах про ИИ-прогнозы):
    1) построить модель, получить вероятность p
    2) снять маржу с линии букмекера, получить q
    3) если p - q > порога (обычно 5-10 п.п.) — объявить это валуем

Проверка: прогоняем ровно эту методику на 250 матчах ВНЕ ВЫБОРКИ,
где есть реальные коэффициенты Bet365, и считаем фактический ROI.
Ставки на исход 1X2, как в ролике.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward
from predict import best_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    wf = walk_forward(df, best_params())
    b = wf.dropna(subset=['qH']).copy()
    P = b[['pH', 'pD', 'pA']].values
    Q = b[['qH', 'qD', 'qA']].values
    O = b[['oH', 'oD', 'oA']].values
    y = b['out'].values.astype(int)
    print(f'матчей вне выборки с коэффициентами: {len(b)}')

    print('\n=== СТРАТЕГИЯ «ПЕРЕВЕС В ПРОЦЕНТНЫХ ПУНКТАХ» (как в ролике) ===')
    print('порог   ставок   ROI      прибыль   ср.кэф   попаданий')
    res = []
    for th in (0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        n, pnl, wins, sump = 0, 0.0, 0, 0.0
        for i in range(len(P)):
            for j in range(3):
                if P[i, j] - Q[i, j] > th:
                    n += 1
                    sump += O[i, j]
                    if y[i] == j:
                        pnl += O[i, j] - 1
                        wins += 1
                    else:
                        pnl -= 1
        roi = pnl / n if n else np.nan
        res.append(dict(порог=th, ставок=n, ROI=roi, прибыль=pnl))
        print(f'{th*100:4.0f}%   {n:6d}   {roi:+.3f}   {pnl:+8.2f}   '
              f'{(sump/n if n else 0):.2f}     {wins}/{n}')

    print('\n  Для сравнения — сколько нужно, чтобы отличить реальный перевес 3% от нуля:')
    for edge in (0.03, 0.05, 0.10):
        for odds in (2.0, 3.5):
            sd = np.sqrt(odds - 1)          # ст.откл. прибыли на 1 ставку (грубо)
            n_need = (1.96 * sd / edge) ** 2
            print(f'    перевес {edge*100:.0f}% при среднем кэфе {odds}: нужно ~{n_need:,.0f} ставок')

    print('\n=== ПОЧЕМУ ПЕРЕВЕС «РАЗВАЛИВАЕТСЯ ПРИ СМЕНЕ УСАДКИ» ===')
    from markets import DEVIG
    print('  один и тот же коэффициент, разные методы снятия маржи:')
    ex = [13.0, 7.40, 1.15]
    print(f'  пример: линия 1X2 = {ex} (Калба — Аль-Айн, БЕТСИТИ)')
    for nm, f in DEVIG.items():
        q = f(ex)
        print(f'    {nm:10s}: 1={q[0]:.4f}  X={q[1]:.4f}  2={q[2]:.4f}   '
              f'справедливый кэф на 1 = {1/q[0]:.2f}')
    print('\n  Разница между методами на лонгшоте — до 2 п.п. вероятности,')
    print('  то есть больше, чем весь «перевес», который обычно объявляют валуем.')

    R = pd.DataFrame(res)
    R.to_csv(os.path.join(ROOT, 'data', 'phantom_backtest.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
