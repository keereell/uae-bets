# -*- coding: utf-8 -*-
"""
EXPERIMENT: does a BLEND of several attacking signals beat open-play xG alone
as the stage-1 fitting target?

Every component is first rescaled so that, ON THE TRAINING SLICE ONLY, its league
total equals the league goal total.  That puts shots, big chances, xGOT and xG on
one common "goals" scale, so a weighted blend is meaningful and the model's own
xg_scale recalibration becomes a no-op (we set xg_scale=1.0 and do the scaling
ourselves inside _target, which is called with the training frame only -> no
look-ahead).

Also tested: fitting ATTACK and DEFENCE on different signals, by running stage 1
twice (once per signal), keeping atk from the attack fit and dfn from the defence
fit, then redoing stage 2 (mu, gamma on real goals) and rho.

Run:  python src/exp_blend_target.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from model import DixonColes, detect_newcomers          # noqa: E402
from markets import wdl, DEVIG                          # noqa: E402
from backtest import rps                                # noqa: E402

BASE_PARAMS = dict(half_life=180.0, alpha=0.30, w_goals=0.0,
                   promoted_shift=0.25, dispersion=None)

WINDOWS = [('A', '2025-02-01', '2025-12-31'),
           ('B', '2026-01-01', None)]


# ----------------------------------------------------------------- components
def comp_cols(df, name):
    """Return (home_series, away_series) of the raw signal, aligned to df."""
    if name == 'xg_open':
        return df['h_xg_open'], df['a_xg_open']
    if name == 'xgot_sum':
        return df['h_xgot_sum'], df['a_xgot_sum']
    if name == 'npxg':
        return df['h_npxg'], df['a_npxg']
    if name == 'xg_all':
        return df['h_xg_all'], df['a_xg_all']
    if name == 'bigch':
        return df['h_bigch'], df['a_bigch']
    if name == 'sot':
        return df['h_sot'], df['a_sot']
    if name == 'vol':
        # non-penalty shot VOLUME; the per-shot xG multiplier is irrelevant
        # because the component is rescaled to the league goal total anyway.
        return df['h_shots_n'] - df['h_pens'], df['a_shots_n'] - df['a_pens']
    if name == 'oppvol':   # opponent non-penalty shot volume, home-perspective
        return df['a_shots_n'] - df['a_pens'], df['h_shots_n'] - df['h_pens']
    raise KeyError(name)


class BlendDC(DixonColes):
    """DixonColes whose stage-1 target is a weighted blend of rescaled signals."""

    def __init__(self, spec=(('xg_open', 1.0),), mode='arith', **kw):
        kw.setdefault('xg_scale', 1.0)        # disable the parent's rescaling
        kw.setdefault('xg_cols', ('h_xg_open', 'a_xg_open'))
        super().__init__(**kw)
        self.spec = list(spec)
        self.mode = mode

    def _target(self, d):
        hg = d.hg.values.astype(float)
        ag = d.ag.values.astype(float)
        gtot = hg + ag

        parts_h, parts_a, ws = [], [], []
        have = np.ones(len(d), bool)
        for name, w in self.spec:
            ch, ca = comp_cols(d, name)
            ch = ch.values.astype(float)
            ca = ca.values.astype(float)
            ok = ~(np.isnan(ch) | np.isnan(ca))
            denom = np.nansum(ch[ok]) + np.nansum(ca[ok])
            sc = float(gtot[ok].sum() / denom) if denom > 0 else 1.0
            parts_h.append(ch * sc)
            parts_a.append(ca * sc)
            ws.append(float(w))
            have &= ok
        ws = np.array(ws) / np.sum(ws)

        if self.mode == 'geom':
            eps = 0.05
            lh = np.zeros(len(d))
            la = np.zeros(len(d))
            for w, ph, pa in zip(ws, parts_h, parts_a):
                lh += w * np.log(np.maximum(ph, 0.0) + eps)
                la += w * np.log(np.maximum(pa, 0.0) + eps)
            bh, ba = np.exp(lh), np.exp(la)
        else:
            bh = np.zeros(len(d))
            ba = np.zeros(len(d))
            for w, ph, pa in zip(ws, parts_h, parts_a):
                bh += w * ph
                ba += w * pa

        # final rescale of the blend back onto the goals scale
        den = np.nansum(bh[have]) + np.nansum(ba[have])
        if den > 0:
            k = float(gtot[have].sum() / den)
            bh, ba = bh * k, ba * k

        wg = np.where(have, self.w_goals, 1.0)
        yh = wg * hg + (1 - wg) * np.where(have, bh, hg)
        ya = wg * ag + (1 - wg) * np.where(have, ba, ag)
        return yh, ya


# ------------------------------------------------- attack/defence split model
def _stage2_and_rho(m, d, ref_ts):
    """Re-run stage 2 (mu, gamma on real goals) + rho for a model whose
    atk/dfn have just been replaced.  Mirrors DixonColes.fit, no covariates."""
    hi = d.home.map(m.idx).values
    ai = d.away.map(m.idx).values
    dt = np.maximum((ref_ts - d.ts.values) / 86400.0, 0.0)
    w = np.exp(-np.log(2) / m.half_life * dt)
    base_h = m.atk[hi] - m.dfn[ai]
    base_a = m.atk[ai] - m.dfn[hi]
    gh = d.hg.values.astype(float)
    ga = d.ag.values.astype(float)

    def nll2(th):
        mu, gam = th
        lh = np.exp(np.clip(mu + gam + base_h, -6, 3))
        la = np.exp(np.clip(mu + base_a, -6, 3))
        return np.sum(w * (lh - gh * (mu + gam + base_h))) + \
               np.sum(w * (la - ga * (mu + base_a)))

    r2 = minimize(nll2, np.array([m.mu, m.gamma]), method='L-BFGS-B',
                  options=dict(maxiter=600))
    m.mu, m.gamma = float(r2.x[0]), float(r2.x[1])
    m.beta = np.zeros(0)
    m.covariates = []
    m.rho = m._fit_rho(d, hi, ai, w, np.zeros((len(d), 0)))
    return m


class SplitDC:
    """Fits stage 1 twice: attack strengths come from `atk_spec`,
    defence strengths from `def_spec`.  Then stage 2 on real goals."""

    def __init__(self, atk_spec, def_spec, mode='arith', **kw):
        self.atk_spec, self.def_spec, self.mode, self.kw = atk_spec, def_spec, mode, kw

    def fit(self, d, ref_ts=None, newcomers=()):
        ma = BlendDC(spec=self.atk_spec, mode=self.mode, **self.kw).fit(
            d, ref_ts=ref_ts, newcomers=newcomers)
        md = BlendDC(spec=self.def_spec, mode=self.mode, **self.kw).fit(
            d, ref_ts=ref_ts, newcomers=newcomers)
        assert ma.teams == md.teams
        ma.dfn = md.dfn
        dd = d[d.played & d.hg.notna()].copy()
        ref = ref_ts if ref_ts is not None else dd.ts.max()
        return _stage2_and_rho(ma, dd, ref)


# ----------------------------------------------------------------- backtest
def walk_forward(df, make_model, start_date, end_date=None, min_train=120, devig='shin'):
    """Identical loop to backtest.walk_forward, but takes a model factory."""
    nc = detect_newcomers(df)
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    mask = played.date >= start_date
    if end_date:
        mask &= played.date <= end_date
    days = sorted(played.loc[mask, 'date'].unique())
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
            m = make_model().fit(train, ref_ts=ref_ts, newcomers=new_here)
        except Exception as e:                                  # pragma: no cover
            print('fit fail', day, e)
            continue
        for _, r in test.iterrows():
            if r.home not in m.idx or r.away not in m.idx:
                continue
            M, lh, la = m.matrix(r.home, r.away)
            p = wdl(M)
            out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            pm = np.array([p['H'], p['D'], p['A']])
            row = dict(date=r.date, home=r.home, away=r.away, out=out,
                       ll=-np.log(max(pm[out], 1e-12)), rps=rps(pm, out))
            if not any(pd.isna([r.odds_H, r.odds_D, r.odds_A])):
                q = DEVIG[devig]([r.odds_H, r.odds_D, r.odds_A])
                row['rps_book'] = rps(q, out)
            rows.append(row)
    return pd.DataFrame(rows)


def paired(res, base):
    """Paired difference of per-match RPS vs the baseline run."""
    k = ['date', 'home', 'away']
    mg = res[k + ['rps']].merge(base[k + ['rps']], on=k, suffixes=('', '_b'))
    d = (mg.rps - mg.rps_b).values
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, rps=res.rps.mean(), d=d.mean(),
                ci=1.96 * se, t=(d.mean() / se if se else np.nan))


# ----------------------------------------------------------------- variants
def mk_blend(spec, mode='arith'):
    return lambda: BlendDC(spec=spec, mode=mode, **BASE_PARAMS)


def mk_split(a, dspec):
    return lambda: SplitDC(a, dspec, **BASE_PARAMS)


VARIANTS = [
    ('a  xg_open (baseline)',          mk_blend([('xg_open', 1.0)])),
    ('b  .5 open + .5 xgot_sum',       mk_blend([('xg_open', .5), ('xgot_sum', .5)])),
    ('c  .5 open + .5 volume',         mk_blend([('xg_open', .5), ('vol', .5)])),
    ('d  .4 open+.3 xgot+.3 vol',      mk_blend([('xg_open', .4), ('xgot_sum', .3), ('vol', .3)])),
    ('e  .5 open + .5 bigch',          mk_blend([('xg_open', .5), ('bigch', .5)])),
    ('b* geom .5 open/.5 xgot_sum',    mk_blend([('xg_open', .5), ('xgot_sum', .5)], 'geom')),
    ('c* geom .5 open/.5 volume',      mk_blend([('xg_open', .5), ('vol', .5)], 'geom')),
    ('d* geom .4/.3/.3',               mk_blend([('xg_open', .4), ('xgot_sum', .3), ('vol', .3)], 'geom')),
    ('g  .5 open + .5 sot',            mk_blend([('xg_open', .5), ('sot', .5)])),
    ('h  npxg alone',                  mk_blend([('npxg', 1.0)])),
    # attack / defence on different signals
    ('s1 atk=open, def=oppvol',        mk_split([('xg_open', 1.0)], [('oppvol', 1.0)])),
    ('s2 atk=open, def=.5open/.5oppvol', mk_split([('xg_open', 1.0)],
                                                  [('xg_open', .5), ('oppvol', .5)])),
    ('s3 atk=.5open/.5vol, def=open',  mk_split([('xg_open', .5), ('vol', .5)],
                                                [('xg_open', 1.0)])),
    ('s0 split control (both=open)',   mk_split([('xg_open', 1.0)], [('xg_open', 1.0)])),
]

GRID_W = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]


def run_all(df, variants):
    out = {}
    for wname, s, e in WINDOWS:
        base = None
        for label, mk in variants:
            t0 = time.time()
            res = walk_forward(df, mk, s, e)
            if base is None:
                base = res
            st = paired(res, base)
            st.update(window=wname, variant=label, secs=round(time.time() - t0, 1))
            out.setdefault(wname, []).append(st)
            print(f"[{wname}] {label:36s} n={st['n']:4d} RPS={st['rps']:.5f} "
                  f"d={st['d']:+.5f} +-{st['ci']:.5f} t={st['t']:+.2f} "
                  f"({st['secs']}s)")
    return {k: pd.DataFrame(v) for k, v in out.items()}


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))

    print('=== PART 1: fixed blends ===')
    t1 = run_all(df, VARIANTS)

    print('\n=== PART 2: weight grid, target = (1-w)*xg_open + w*OTHER ===')
    grids = {}
    for other in ('xgot_sum', 'vol', 'bigch'):
        vs = [('a  xg_open (baseline)', mk_blend([('xg_open', 1.0)]))]
        for w in GRID_W:
            if w == 0.0:
                continue
            spec = [('xg_open', 1 - w), (other, w)] if w < 1.0 else [(other, 1.0)]
            vs.append((f'w={w:.2f} {other}', mk_blend(spec)))
        print(f'\n--- {other} ---')
        grids[other] = run_all(df, vs)

    # PART 3 -- a blend changes the variance of the target, hence the fitted
    # dispersion, hence the EFFECTIVE shrinkage (xg_open 0.55 -> volume 0.30).
    # So a blend could be losing only because alpha=0.30 no longer suits it.
    # Re-check with alpha tuned: pick alpha on ONE window, report on the OTHER.
    print('\n=== PART 3: alpha x target, tuned on one window, read on the other ===')
    vs3 = [('a  xg_open (baseline)', mk_blend([('xg_open', 1.0)]))]
    for tgt, spec in [('open', [('xg_open', 1.0)]),
                      ('.5open+.5vol', [('xg_open', .5), ('vol', .5)]),
                      ('.5open+.5xgot', [('xg_open', .5), ('xgot_sum', .5)])]:
        for al in (0.15, 0.30, 0.60, 1.20):
            kw = dict(BASE_PARAMS, alpha=al)
            vs3.append((f'{tgt} a={al}',
                        (lambda s=spec, k=kw: BlendDC(spec=s, **k))))
    t3 = run_all(df, vs3)

    outdir = os.path.join(ROOT, 'data')
    rows = []
    for tag, tab in ([('fixed', t1)] + [(f'grid_{k}', v) for k, v in grids.items()]
                     + [('alpha', t3)]):
        for wname, t in tab.items():
            t = t.copy()
            t['block'] = tag
            rows.append(t)
    allt = pd.concat(rows, ignore_index=True)
    allt.to_csv(os.path.join(outdir, 'exp_blend_target.csv'),
                index=False, encoding='utf-8-sig')

    print('\n=== SUMMARY: paired d vs xg_open baseline, both windows ===')
    piv = allt.pivot_table(index=['block', 'variant'], columns='window',
                           values=['rps', 'd', 't'])
    print(piv.round(5).to_string())
    print('\nwrote', os.path.join(outdir, 'exp_blend_target.csv'))


if __name__ == '__main__':
    main()
