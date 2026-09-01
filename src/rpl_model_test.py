# -*- coding: utf-8 -*-
"""
ПРЯМАЯ ПРОВЕРКА УТВЕРЖДЕНИЯ ИЗ РОЛИКА НА САМОЙ РПЛ.

Та же самая xG-модель Диксона-Коулза, что и для ОАЭ, обучается на РПЛ
и проверяется вперёд по времени против реальных ЗАКРЫВАЮЩИХ коэффициентов
(football-data.co.uk): Pinnacle, максимум по рынку, среднее по рынку.

Отвечаем на три вопроса:
  1. Точнее ли модель, чем линия?
  2. Добавляет ли она информацию сверх линии?
  3. Сколько бы заработала стратегия «ставим при перевесе модели > порога»
     при ставке по лучшей цене рынка и по средней цене рынка?
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers
from markets import wdl, DEVIG
from backtest import rps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 230)

PARAMS = dict(half_life=150.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25)


def walk_forward_rpl(train_df, eval_df, params, start_date, min_train=150):
    """Обучаем на train_df (все матчи РПЛ), оцениваем на eval_df (те, где есть кэфы)."""
    nc = detect_newcomers(train_df)
    played = train_df[train_df.played & train_df.hg.notna()].sort_values('ts').reset_index(drop=True)
    ev = eval_df[eval_df.date >= start_date].sort_values('ts')
    rows = []
    for day in sorted(ev.date.unique()):
        train = played[played.date < day]
        if len(train) < min_train:
            continue
        test = ev[ev.date == day]
        try:
            m = DixonColes(**params).fit(train, ref_ts=float(test.ts.min()),
                                         newcomers=nc.get(int(test.season.iloc[0]), set()))
        except Exception as e:
            print('fit fail', day, e)
            continue
        for _, r in test.iterrows():
            if r.home not in m.idx or r.away not in m.idx:
                continue
            M, lh, la = m.matrix(r.home, r.away)
            p = wdl(M)
            out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            rows.append(dict(date=r.date, home=r.home, away=r.away, out=out, lh=lh, la=la,
                             pH=p['H'], pD=p['D'], pA=p['A'],
                             PSCH=r.get('PSCH'), PSCD=r.get('PSCD'), PSCA=r.get('PSCA'),
                             MaxCH=r.MaxCH, MaxCD=r.MaxCD, MaxCA=r.MaxCA,
                             AvgCH=r.AvgCH, AvgCD=r.AvgCD, AvgCA=r.AvgCA))
    return pd.DataFrame(rows)


def strategy(R, price_cols, label, devig='power'):
    P = R[['pH', 'pD', 'pA']].values
    B = R[list(price_cols)].values
    y = R['out'].values.astype(int)
    ok = ~np.isnan(B).any(axis=1)
    P, B, y = P[ok], B[ok], y[ok]
    out = []
    for th in (0.0, 0.03, 0.05, 0.07, 0.10, 0.15):
        m = (P * B - 1.0) > th
        n = int(m.sum())
        if n == 0:
            out.append(dict(порог=f'{th:.0%}', ставок=0, ROI='—', ст_ошибка='—'))
            continue
        won = (np.arange(3)[None, :] == y[:, None]) & m
        ret = np.where(won[m], B[m] - 1.0, -1.0)
        roi = ret.mean()
        se = ret.std(ddof=1) / np.sqrt(n)
        out.append(dict(порог=f'{th:.0%}', ставок=n, ср_кэф=f'{B[m].mean():.2f}',
                        ROI=f'{100*roi:+.2f}%', ст_ошибка=f'±{100*1.96*se:.2f}%'))
    df = pd.DataFrame(out)
    print(f'\n  ставим по цене «{label}» (матчей {len(P)}):')
    print(df.to_string(index=False))


def main():
    full = pd.read_csv(os.path.join(ROOT, 'data', 'rpl', 'matches.csv'))
    odds = pd.read_csv(os.path.join(ROOT, 'data', 'rpl', 'matches_odds.csv'))
    odds = odds[odds.played & odds.hg.notna()]
    print(f'РПЛ: всего матчей {len(full)}, из них с коэффициентами {len(odds)}, '
          f'с xG {int(full.h_xg.notna().sum())}')

    start = sorted(full[full.played].date)[int(len(full) * 0.45)]
    R = walk_forward_rpl(full, odds, PARAMS, start)
    print(f'\nоценено вперёд по времени: {len(R)} матчей, с {start}')
    R.to_csv(os.path.join(ROOT, 'data', 'rpl', 'walkforward.csv'), index=False, encoding='utf-8-sig')

    # ---------- 1. точность
    print('\n' + '=' * 100)
    print('1. ТОЧНОСТЬ: МОДЕЛЬ ПРОТИВ ЛИНИИ')
    print('=' * 100)
    P = R[['pH', 'pD', 'pA']].values
    y = R.out.values.astype(int)
    m_rps = np.mean([rps(P[i], y[i]) for i in range(len(P))])
    m_ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
    print(f'  модель          : RPS {m_rps:.4f}   log-loss {m_ll:.4f}   (N={len(P)})')
    for cols, nm in ((('AvgCH', 'AvgCD', 'AvgCA'), 'среднее по рынку'),
                     (('PSCH', 'PSCD', 'PSCA'), 'Pinnacle')):
        sub = R.dropna(subset=list(cols))
        if len(sub) < 30:
            print(f'  {nm:16s}: мало данных ({len(sub)})')
            continue
        Q = np.array([DEVIG['power'](list(r)) for r in sub[list(cols)].values])
        yy = sub.out.values.astype(int)
        b_rps = np.mean([rps(Q[i], yy[i]) for i in range(len(Q))])
        b_ll = -np.mean(np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-12, 1)))
        PP = sub[['pH', 'pD', 'pA']].values
        mm_rps = np.mean([rps(PP[i], yy[i]) for i in range(len(PP))])
        print(f'  {nm:16s}: RPS {b_rps:.4f}   log-loss {b_ll:.4f}   (N={len(sub)}, '
              f'модель на той же выборке RPS {mm_rps:.4f})')

    # ---------- 2. добавляет ли модель информацию
    print('\n' + '=' * 100)
    print('2. ДОБАВЛЯЕТ ЛИ МОДЕЛЬ ИНФОРМАЦИЮ СВЕРХ ЛИНИИ')
    print('=' * 100)
    sub = R.dropna(subset=['AvgCH', 'AvgCD', 'AvgCA'])
    Pp = sub[['pH', 'pD', 'pA']].values
    Qq = np.array([DEVIG['power'](list(r)) for r in sub[['AvgCH', 'AvgCD', 'AvgCA']].values])
    yy = sub.out.values.astype(int)

    def blend_ll(w):
        z = w * np.log(np.clip(Pp, 1e-9, 1)) + (1 - w) * np.log(np.clip(Qq, 1e-9, 1))
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        pb = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(pb[np.arange(len(yy)), yy], 1e-12, 1)))

    r = minimize_scalar(blend_ll, bounds=(-0.5, 1.5), method='bounded')
    print(f'  оптимальный вес модели рядом с рынком: {r.x:+.3f}')
    print(f'  log-loss: только рынок {blend_ll(0):.4f} | смесь {blend_ll(r.x):.4f} | '
          f'только модель {blend_ll(1):.4f}')
    if r.x > 0.05:
        print('  --> МОДЕЛЬ ДОБАВЛЯЕТ ИНФОРМАЦИЮ')
    else:
        print('  --> модель не добавляет информации сверх линии')

    # ---------- 3. ROI
    print('\n' + '=' * 100)
    print('3. ROI СТРАТЕГИИ «СТАВИМ ПРИ ПЕРЕВЕСЕ МОДЕЛИ»')
    print('=' * 100)
    strategy(R, ('MaxCH', 'MaxCD', 'MaxCA'), 'максимум по рынку (~40 контор)')
    strategy(R, ('AvgCH', 'AvgCD', 'AvgCA'), 'среднее по рынку (одна обычная контора)')
    sub = R.dropna(subset=['PSCH', 'PSCD', 'PSCA'])
    if len(sub) > 50:
        strategy(sub, ('PSCH', 'PSCD', 'PSCA'), 'Pinnacle')


if __name__ == '__main__':
    main()
