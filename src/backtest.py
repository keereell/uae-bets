# -*- coding: utf-8 -*-
"""
Честная валидация «вперёд по времени» (walk-forward):
для каждого игрового дня модель обучается ТОЛЬКО на матчах строго до него.

Метрики:
  * log-loss на исходе 1X2
  * RPS (ranked probability score) — стандарт для футбола
  * то же самое для коэффициентов Bet365 (после снятия маржи) — как эталон
  * ROI простой стратегии «ставим при перевесе > порога»
"""
import sys, os, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers
from markets import wdl, DEVIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rps(p, outcome):
    """p = [pH, pD, pA]; outcome in {0,1,2}."""
    obs = np.zeros(3); obs[outcome] = 1
    cp, co, s = 0.0, 0.0, 0.0
    for i in range(2):
        cp += p[i]; co += obs[i]
        s += (cp - co) ** 2
    return s / 2.0


def walk_forward(df, params, start_date='2025-02-01', end_date=None, min_train=120, devig='shin'):
    nc = detect_newcomers(df)
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    eval_mask = played.date >= start_date
    if end_date:
        eval_mask &= played.date <= end_date
    days = sorted(played.loc[eval_mask, 'date'].unique())
    rows = []
    for day in days:
        train = played[played.date < day]
        if len(train) < min_train:
            continue
        test = played[played.date == day]
        ref_ts = test.ts.min()
        new_here = set()
        for s in test.season.unique():
            new_here |= nc.get(s, set())
        try:
            m = DixonColes(**params).fit(train, ref_ts=ref_ts, newcomers=new_here)
        except Exception as e:
            print('fit fail', day, e); continue
        for _, r in test.iterrows():
            if r.home not in m.idx or r.away not in m.idx:
                continue
            # cov=r: строка матча целиком. Модель возьмёт из неё только
            # те колонки, что перечислены в covariates/covariates_team;
            # без ковариат аргумент игнорируется.
            M, lh, la = m.matrix(r.home, r.away, cov=r)
            p = wdl(M)
            out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            pm = np.array([p['H'], p['D'], p['A']])
            row = dict(date=r.date, season=r.season, home=r.home, away=r.away,
                       hg=r.hg, ag=r.ag, out=out, lh=lh, la=la,
                       pH=pm[0], pD=pm[1], pA=pm[2],
                       ll=-np.log(max(pm[out], 1e-12)), rps=rps(pm, out))
            if not any(pd.isna([r.odds_H, r.odds_D, r.odds_A])):
                o = [r.odds_H, r.odds_D, r.odds_A]
                q = DEVIG[devig](o)
                row.update(oH=o[0], oD=o[1], oA=o[2],
                           qH=q[0], qD=q[1], qA=q[2],
                           ll_book=-np.log(max(q[out], 1e-12)), rps_book=rps(q, out))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(res, name=''):
    r = res.dropna(subset=['ll'])
    b = res.dropna(subset=['ll_book']) if 'll_book' in res else res.iloc[:0]
    s = dict(name=name, n=len(r), logloss=r.ll.mean(), rps=r.rps.mean())
    if len(b):
        s.update(n_book=len(b), logloss_book=b.ll_book.mean(), rps_book=b.rps_book.mean(),
                 logloss_model_on_book_subset=b.ll.mean(), rps_model_on_book_subset=b.rps.mean())
    return s


def roi(res, thresholds=(0.02, 0.05, 0.08, 0.12), max_odds=8.0, min_odds=1.3):
    """ROI ставок против Bet365 при разных порогах перевеса."""
    b = res.dropna(subset=['oH']).copy()
    out = []
    for th in thresholds:
        bets, pnl, staked = 0, 0.0, 0.0
        for _, r in b.iterrows():
            for i, (pc, oc) in enumerate((('pH', 'oH'), ('pD', 'oD'), ('pA', 'oA'))):
                price = r[oc]
                if not (min_odds <= price <= max_odds):
                    continue
                e = r[pc] * price - 1
                if e > th:
                    bets += 1; staked += 1
                    pnl += (price - 1) if r['out'] == i else -1
        out.append(dict(threshold=th, bets=bets,
                        roi=(pnl / staked if staked else np.nan), pnl=pnl))
    return pd.DataFrame(out)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    grid = dict(
        half_life=[90, 150, 220, 320, 500],
        alpha=[0.7, 1.5, 3.0, 6.0],
        w_goals=[0.0, 0.35, 0.65, 1.0],
    )
    keys = list(grid)
    res_rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        r = walk_forward(df, params)
        s = summarize(r, str(params))
        res_rows.append({**params, **{k: v for k, v in s.items() if k != 'name'}})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res_rows[-1].items()})
    out = pd.DataFrame(res_rows).sort_values('rps')
    out.to_csv(os.path.join(ROOT, 'data', 'tuning.csv'), index=False, encoding='utf-8-sig')
    print('\n=== ЛУЧШИЕ ПО RPS ===')
    print(out.head(12).round(4).to_string(index=False))
