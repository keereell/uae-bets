# -*- coding: utf-8 -*-
"""
exp_bootstrap_uncertainty.py
============================
Quantify the Dixon-Coles model's OWN uncertainty and use it.

  (a) Bootstrap ensemble: B refits per walk-forward day, resampling the TRAINING
      matches with replacement (matches, not residuals). Gives, for every
      upcoming fixture, a posterior sample of P(H)/P(D)/P(A), of the expected
      total, and of any market probability (Over 2.5 used as the example).
  (b) Bagging: does the mean of the bootstrap predictive distributions beat the
      single point-estimate model on RPS? Paired SE, both windows.
  (c) Calibration / reliability tables (H, D, A x window A, window B).
  (d) Temperature p ~ p^T renormalised, fitted on window A, reported on window B.
  (e) Fraction of fixtures whose bootstrap 90% interval for P(H) is wider than
      10 percentage points.

NO LOOK-AHEAD: for each matchday the model (and every bootstrap replicate) is
fitted only on matches strictly before that day, exactly as backtest.walk_forward
does. The bootstrap resamples only inside that training window.

Run:
    python src/exp_bootstrap_uncertainty.py            # uses cache if present
    python src/exp_bootstrap_uncertainty.py --rebuild  # refit everything
    python src/exp_bootstrap_uncertainty.py --B 200 --rebuild
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from model import DixonColes, detect_newcomers          # noqa: E402
from markets import wdl, totals, DEVIG                   # noqa: E402
from backtest import rps                                 # noqa: E402

BASE_PARAMS = dict(half_life=180.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25,
                   xg_cols=('h_xg_open', 'a_xg_open'), dispersion=None)

START = '2025-02-01'
SPLIT = '2026-01-01'          # window A: START..2025-12-31 ; window B: SPLIT..
A_END = '2025-12-31'
MIN_TRAIN = 120
OVER_LINE = 2.5

CACHE_DIR = os.environ.get(
    'UAE_SCRATCH',
    r'C:\Users\Dell\AppData\Local\Temp\claude\C--Users-Dell-Desktop-Projects-UAE'
    r'\c87180a1-1dad-4ee3-8141-a6192ff1fe9e\scratchpad')

_DF = None          # per-process globals (multiprocessing)
_NC = None
_PARAMS = None


# ----------------------------------------------------------------- worker
def _init(csv_path, params):
    global _DF, _NC, _PARAMS
    df = pd.read_csv(csv_path)
    _DF = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    _NC = detect_newcomers(df)
    _PARAMS = params


def _fit(train, ref_ts, newcomers):
    return DixonColes(**_PARAMS).fit(train, ref_ts=ref_ts, newcomers=newcomers)


def day_job(args):
    """One matchday: point fit + B bootstrap refits. Returns list of row dicts."""
    day, B, seed = args
    played, nc = _DF, _NC
    train = played[played.date < day]
    if len(train) < MIN_TRAIN:
        return []
    test = played[played.date == day]
    ref_ts = test.ts.min()
    new_here = set()
    for s in test.season.unique():
        new_here |= nc.get(s, set())

    try:
        m0 = _fit(train, ref_ts, new_here)
    except Exception as e:
        print('point fit fail', day, e)
        return []

    fixtures = [r for _, r in test.iterrows()
                if r.home in m0.idx and r.away in m0.idx]
    if not fixtures:
        return []

    # ---- bootstrap replicates (resample training MATCHES with replacement)
    rng = np.random.default_rng(seed)
    n = len(train)
    tr_idx = np.arange(n)
    draws = {k: [] for k in ('pH', 'pD', 'pA', 'tot', 'pOver')}
    n_ok, n_fail = 0, 0
    for b in range(B):
        samp = train.iloc[rng.choice(tr_idx, size=n, replace=True)]
        try:
            mb = _fit(samp, ref_ts, new_here)
        except Exception:
            n_fail += 1
            continue
        if any((r.home not in mb.idx or r.away not in mb.idx) for r in fixtures):
            n_fail += 1                     # a team never got drawn: discard draw
            continue
        row_pH, row_pD, row_pA, row_tot, row_ov = [], [], [], [], []
        for r in fixtures:
            M, lh, la = mb.matrix(r.home, r.away)
            p = wdl(M)
            ov, un, pu = totals(M, OVER_LINE)
            row_pH.append(p['H'])
            row_pD.append(p['D'])
            row_pA.append(p['A'])
            row_tot.append(lh + la)
            row_ov.append(ov)
        draws['pH'].append(row_pH)
        draws['pD'].append(row_pD)
        draws['pA'].append(row_pA)
        draws['tot'].append(row_tot)
        draws['pOver'].append(row_ov)
        n_ok += 1
    D = {k: np.array(v) for k, v in draws.items()}      # (n_ok, n_fixtures)

    rows = []
    for j, r in enumerate(fixtures):
        M0, lh0, la0 = m0.matrix(r.home, r.away)
        p0 = wdl(M0)
        ov0, un0, pu0 = totals(M0, OVER_LINE)
        out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
        pm = np.array([p0['H'], p0['D'], p0['A']])
        bh, bd, ba = D['pH'][:, j], D['pD'][:, j], D['pA'][:, j]
        bag = np.array([bh.mean(), bd.mean(), ba.mean()])
        bag = bag / bag.sum()
        row = dict(date=r.date, season=r.season, home=r.home, away=r.away,
                   hg=r.hg, ag=r.ag, out=out,
                   n_draws=n_ok, n_fail=n_fail,
                   pH=pm[0], pD=pm[1], pA=pm[2],
                   lh=lh0, la=la0, tot=lh0 + la0, pOver=ov0,
                   rps=rps(pm, out), ll=-np.log(max(pm[out], 1e-12)),
                   bagH=bag[0], bagD=bag[1], bagA=bag[2],
                   rps_bag=rps(bag, out), ll_bag=-np.log(max(bag[out], 1e-12)),
                   sd_pH=bh.std(ddof=1), sd_pD=bd.std(ddof=1), sd_pA=ba.std(ddof=1),
                   q05_pH=np.percentile(bh, 5), q50_pH=np.percentile(bh, 50),
                   q95_pH=np.percentile(bh, 95),
                   q05_pD=np.percentile(bd, 5), q95_pD=np.percentile(bd, 95),
                   q05_pA=np.percentile(ba, 5), q95_pA=np.percentile(ba, 95),
                   sd_tot=D['tot'][:, j].std(ddof=1),
                   q05_tot=np.percentile(D['tot'][:, j], 5),
                   q95_tot=np.percentile(D['tot'][:, j], 95),
                   mean_tot=D['tot'][:, j].mean(),
                   sd_pOver=D['pOver'][:, j].std(ddof=1),
                   q05_pOver=np.percentile(D['pOver'][:, j], 5),
                   q95_pOver=np.percentile(D['pOver'][:, j], 95),
                   mean_pOver=D['pOver'][:, j].mean())
        if not any(pd.isna([r.odds_H, r.odds_D, r.odds_A])):
            o = [r.odds_H, r.odds_D, r.odds_A]
            q = DEVIG['shin'](o)
            row.update(oH=o[0], oD=o[1], oA=o[2], qH=q[0], qD=q[1], qA=q[2],
                       rps_book=rps(q, out), ll_book=-np.log(max(q[out], 1e-12)))
        rows.append(row)
    return rows


# ----------------------------------------------------------------- driver
def build(csv_path, B, params, start=START, workers=None):
    df = pd.read_csv(csv_path)
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    days = sorted(played.loc[played.date >= start, 'date'].unique())
    jobs = [(d, B, 20260000 + i) for i, d in enumerate(days)]

    rows = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(csv_path, params)) as ex:
            for k, res in enumerate(ex.map(day_job, jobs, chunksize=1)):
                rows.extend(res)
                if (k + 1) % 10 == 0:
                    print('  day %d/%d  rows=%d' % (k + 1, len(jobs), len(rows)), flush=True)
    else:
        _init(csv_path, params)
        for k, j in enumerate(jobs):
            rows.extend(day_job(j))
            if (k + 1) % 10 == 0:
                print('  day %d/%d  rows=%d' % (k + 1, len(jobs), len(rows)), flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- stats
def paired(a, b, label):
    """a, b: per-match losses (same matches). d = a - b. Negative = a better."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return dict(label=label, n=n, mean_a=np.mean(a), mean_b=np.mean(b),
                d=d.mean(), ci95=1.96 * se, t=d.mean() / se if se > 0 else np.nan)


def fmt_paired(p):
    return ('%-34s n=%3d  new=%.5f  ref=%.5f  d=%+.5f +/- %.5f  t=%+.2f'
            % (p['label'], p['n'], p['mean_a'], p['mean_b'], p['d'], p['ci95'], p['t']))


def temper(P, T):
    Q = np.clip(P, 1e-12, None) ** T
    return Q / Q.sum(axis=1, keepdims=True)


def rps_vec(P, out):
    cp = np.cumsum(P[:, :2], axis=1)
    co = np.cumsum(np.eye(3)[out][:, :2], axis=1)
    return ((cp - co) ** 2).sum(axis=1) / 2.0


def fit_T(P, out, metric='rps'):
    grid = np.linspace(0.40, 2.50, 211)
    vals = []
    for T in grid:
        Q = temper(P, T)
        if metric == 'rps':
            vals.append(rps_vec(Q, out).mean())
        else:
            vals.append(-np.log(np.clip(Q[np.arange(len(out)), out], 1e-12, None)).mean())
    i = int(np.argmin(vals))
    return float(grid[i]), float(vals[i])


BINS = [0.0, 0.10, 0.20, 0.275, 0.35, 0.45, 0.55, 0.70, 1.001]


def reliability(P, out, k, bins=BINS):
    p = P[:, k]
    y = (out == k).astype(float)
    lab = np.digitize(p, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        m = lab == i
        if m.sum() == 0:
            continue
        rows.append(dict(bin='%.2f-%.2f' % (bins[i], min(bins[i + 1], 1.0)),
                         n=int(m.sum()), pred=p[m].mean(), obs=y[m].mean(),
                         diff=y[m].mean() - p[m].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, 'exp_bootstrap_wf_B%d.csv' % a.B)
    csv_path = os.path.join(ROOT, 'data', 'matches.csv')

    if a.rebuild or not os.path.exists(cache):
        print('building bootstrap walk-forward: B=%d, workers=%d' % (a.B, a.workers))
        import time
        t0 = time.time()
        res = build(csv_path, a.B, BASE_PARAMS, workers=a.workers)
        res.to_csv(cache, index=False, encoding='utf-8-sig')
        print('built in %.0f s -> %s' % (time.time() - t0, cache))
    else:
        res = pd.read_csv(cache)
        print('loaded cache %s' % cache)

    res = res.sort_values('date').reset_index(drop=True)
    wA = (res.date >= START) & (res.date <= A_END)
    wB = res.date >= SPLIT
    wins = [('A  (%s..%s)' % (START, A_END), wA),
            ('B  (%s..)   ' % SPLIT, wB),
            ('both        ', wA | wB)]

    print('\n' + '=' * 78)
    print('0. SANITY: point model reproduces the stated baseline')
    print('=' * 78)
    for nm, m in wins:
        r = res[m]
        bk = r.dropna(subset=['rps_book'])
        print('%s n=%3d  RPS=%.5f  logloss=%.5f   [book n=%3d RPS=%.5f]'
              % (nm, len(r), r.rps.mean(), r.ll.mean(), len(bk),
                 bk.rps_book.mean() if len(bk) else np.nan))
    print('bootstrap draws used per fixture: min=%d median=%d (failed draws total=%d)'
          % (res.n_draws.min(), res.n_draws.median(), res.n_fail.sum()))

    # ---------------------------------------------------------------- (a)
    print('\n' + '=' * 78)
    print('(a) BOOTSTRAP POSTERIOR WIDTH  (B=%d refits per matchday)' % a.B)
    print('=' * 78)
    print('%-14s %-10s %8s %8s %8s %8s %8s' %
          ('window', 'quantity', 'p5', 'p25', 'median', 'p75', 'p95'))
    for nm, m in wins:
        r = res[m]
        for q, lab in (('sd_pH', 'SD P(H)'), ('sd_pD', 'SD P(D)'), ('sd_pA', 'SD P(A)'),
                       ('sd_tot', 'SD total'), ('sd_pOver', 'SD P(o2.5)')):
            v = r[q].values
            print('%-14s %-10s %8.4f %8.4f %8.4f %8.4f %8.4f' %
                  (nm, lab, *np.percentile(v, [5, 25, 50, 75, 95])))
        print()
    r = res[wA | wB]
    print('90%% interval WIDTH for P(H): median=%.4f  p5=%.4f  p95=%.4f'
          % tuple(np.percentile((r.q95_pH - r.q05_pH).values, [50, 5, 95])))
    print('90%% interval WIDTH for total goals: median=%.3f  p5=%.3f  p95=%.3f'
          % tuple(np.percentile((r.q95_tot - r.q05_tot).values, [50, 5, 95])))
    print('90%% interval WIDTH for P(Over 2.5): median=%.4f  p5=%.4f  p95=%.4f'
          % tuple(np.percentile((r.q95_pOver - r.q05_pOver).values, [50, 5, 95])))

    print('\n-- signal vs own noise (how much of the fixture-to-fixture spread is real) --')
    for nm, m in wins:
        rr = res[m]
        for c, s, lab in (('pH', 'sd_pH', 'P(H)'), ('pA', 'sd_pA', 'P(A)'),
                          ('pD', 'sd_pD', 'P(D)'), ('tot', 'sd_tot', 'total'),
                          ('pOver', 'sd_pOver', 'P(o2.5)')):
            sig = rr[c].std(ddof=1)
            noi = np.sqrt((rr[s] ** 2).mean())
            print('%s %-8s spread across fixtures SD=%.4f  own error RMS=%.4f  ratio=%.2f'
                  % (nm, lab, sig, noi, sig / noi))

    print('\n-- worst matchdays for bootstrap stability --')
    bad = res.groupby('date').n_draws.min().sort_values().head(5)
    print(bad.to_string())
    print('(draws discarded because a team was never drawn into the resample: %d of %d)'
          % (res.groupby('date').n_fail.first().sum(), a.B * res.date.nunique()))

    # ---------------------------------------------------------------- (b)
    print('\n' + '=' * 78)
    print('(b) BAGGING: ensemble mean vs point estimate  (paired, per match)')
    print('=' * 78)
    for nm, m in wins:
        r = res[m]
        print(fmt_paired(paired(r.rps_bag.values, r.rps.values, 'RPS bag-point  ' + nm)))
    for nm, m in wins:
        r = res[m]
        print(fmt_paired(paired(r.ll_bag.values, r.ll.values, 'LL  bag-point  ' + nm)))
    print('(d<0 means the bagged ensemble is BETTER; |d| must exceed ~0.0035 to matter)')

    # ---------------------------------------------------------------- (c)
    print('\n' + '=' * 78)
    print('(c) CALIBRATION / RELIABILITY')
    print('=' * 78)
    from scipy.optimize import minimize_scalar
    for nm, m in wins[:2]:
        r = res[m]
        P = r[['pH', 'pD', 'pA']].values
        out = r.out.values
        print('\n--- window %s   n=%d ---' % (nm.strip(), len(r)))
        for k, lab in enumerate(('HOME', 'DRAW', 'AWAY')):
            t = reliability(P, out, k)
            print(' %s  mean pred=%.4f  observed=%.4f  (diff %+0.4f)'
                  % (lab, P[:, k].mean(), (out == k).mean(),
                     (out == k).mean() - P[:, k].mean()))
            print(t.to_string(index=False, float_format=lambda x: '%.4f' % x))
            p = np.clip(P[:, k], 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            y = (out == k).astype(float)

            def nl(s):
                q = 1 / (1 + np.exp(-s * z))
                q = np.clip(q, 1e-9, 1 - 1e-9)
                return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
            s = minimize_scalar(nl, bounds=(0.05, 3.0), method='bounded').x
            print('  logistic slope on log-odds = %.3f  (<1 = overconfident)\n' % s)

    # ---------------------------------------------------------------- (d)
    print('\n' + '=' * 78)
    print('(d) TEMPERATURE  p ~ p^T renormalised   (fitted on A, reported on B)')
    print('=' * 78)
    rA, rB = res[wA], res[wB]
    for src, tgt, sname, tname in ((rA, rB, 'A', 'B'), (rB, rA, 'B', 'A')):
        PA, oA = src[['pH', 'pD', 'pA']].values, src.out.values
        PB, oB = tgt[['pH', 'pD', 'pA']].values, tgt.out.values
        T, v = fit_T(PA, oA, 'rps')
        Tll, _ = fit_T(PA, oA, 'll')
        base = rps_vec(PB, oB)
        new = rps_vec(temper(PB, T), oB)
        print('fit on %s: T(rps)=%.3f  T(logloss)=%.3f   in-sample RPS on %s %.5f -> %.5f'
              % (sname, T, Tll, sname, rps_vec(PA, oA).mean(), v))
        print('   ' + fmt_paired(paired(new, base, 'RPS temp-point on %s' % tname)))
    # the reliability table shows a consistent draw deficit -> test a draw shift
    print('\n-- DRAW SHIFT  p_D += delta, H and A scaled down pro rata (fit A -> report B) --')

    def draw_shift(P, delta):
        Q = P.copy()
        Q[:, 1] = np.clip(Q[:, 1] + delta, 1e-6, 0.99)
        rest = 1.0 - Q[:, 1]
        s = P[:, 0] + P[:, 2]
        Q[:, 0] = P[:, 0] / s * rest
        Q[:, 2] = P[:, 2] / s * rest
        return Q

    for src, tgt, sname, tname in ((rA, rB, 'A', 'B'), (rB, rA, 'B', 'A')):
        PA, oA = src[['pH', 'pD', 'pA']].values, src.out.values
        PB, oB = tgt[['pH', 'pD', 'pA']].values, tgt.out.values
        grid = np.linspace(-0.05, 0.10, 151)
        v = [rps_vec(draw_shift(PA, dd), oA).mean() for dd in grid]
        dd = float(grid[int(np.argmin(v))])
        base = rps_vec(PB, oB)
        new = rps_vec(draw_shift(PB, dd), oB)
        print('fit on %s: delta=%+.4f (in-sample %.5f -> %.5f)'
              % (sname, dd, rps_vec(PA, oA).mean(), min(v)))
        print('   ' + fmt_paired(paired(new, base, 'RPS drawshift-point on %s' % tname)))

    print('\n-- same, applied to the BAGGED probabilities --')
    for src, tgt, sname, tname in ((rA, rB, 'A', 'B'), (rB, rA, 'B', 'A')):
        PA, oA = src[['bagH', 'bagD', 'bagA']].values, src.out.values
        PB, oB = tgt[['bagH', 'bagD', 'bagA']].values, tgt.out.values
        T, v = fit_T(PA, oA, 'rps')
        base = rps_vec(PB, oB)
        new = rps_vec(temper(PB, T), oB)
        print('fit on %s: T(rps)=%.3f' % (sname, T))
        print('   ' + fmt_paired(paired(new, base, 'RPS tempbag-bag on %s' % tname)))

    # ---------------------------------------------------------------- (e)
    print('\n' + '=' * 78)
    print('(e) HOW OFTEN THE MODEL CANNOT TELL')
    print('=' * 78)
    for nm, m in wins:
        r = res[m]
        w = (r.q95_pH - r.q05_pH).values
        print('%s n=%3d  P(H) 90%% width >10pp: %5.1f%%  >5pp: %5.1f%%  >15pp: %5.1f%%  median %.4f'
              % (nm, len(r), 100 * (w > 0.10).mean(), 100 * (w > 0.05).mean(),
                 100 * (w > 0.15).mean(), np.median(w)))
    r = res[wA | wB]
    for col, lab in (('q95_pD', 'P(D)'), ('q95_pA', 'P(A)'), ('q95_pOver', 'P(o2.5)')):
        w = (r[col] - r[col.replace('q95', 'q05')]).values
        print('both         %s 90%% width >10pp: %5.1f%%  median %.4f'
              % (lab, 100 * (w > 0.10).mean(), np.median(w)))

    # edge-vs-uncertainty: the practical question
    print('\n-- betting edge vs model error (Bet365 subset) --')
    bk = res[(wA | wB) & res.oH.notna()].copy()
    if len(bk):
        edges, sds = [], []
        for c, q, s in (('pH', 'qH', 'sd_pH'), ('pD', 'qD', 'sd_pD'), ('pA', 'qA', 'sd_pA')):
            edges.append((bk[c] - bk[q]).abs().values)
            sds.append(bk[s].values)
        e = np.concatenate(edges)
        sd = np.concatenate(sds)
        print('n outcomes=%d  median |p_model - p_book| = %.4f   median bootstrap SD = %.4f'
              % (len(e), np.median(e), np.median(sd)))
        for th in (0.03, 0.05, 0.08, 0.12):
            m = e >= th
            if m.sum():
                print('  |edge|>=%.2f : n=%4d  share where bootstrap SD > edge: %5.1f%%'
                      % (th, m.sum(), 100 * (sd[m] > e[m]).mean()))
        print('  overall share of outcomes where bootstrap SD > |edge|: %.1f%%'
              % (100 * (sd > e).mean()))

    # ---------------------------------------------------------------- (f)
    print('\n' + '=' * 78)
    print('(f) IS THE UNCERTAINTY USABLE?  model vs Bet365 by bootstrap-SD tercile')
    print('=' * 78)
    for nm, m in wins:
        rr = res[m].dropna(subset=['rps_book']).copy()
        if len(rr) < 30:
            continue
        q = rr.sd_pH.quantile([1 / 3, 2 / 3]).values
        rr['g'] = np.digitize(rr.sd_pH.values, q)
        print('%s' % nm)
        for g, lab in enumerate(('low SD ', 'mid SD ', 'high SD')):
            s = rr[rr.g == g]
            if not len(s):
                continue
            p = paired(s.rps.values, s.rps_book.values, '')
            print('   %s n=%3d  sd=%.4f  model RPS=%.5f  book RPS=%.5f  d=%+.5f +/- %.5f'
                  % (lab, len(s), s.sd_pH.mean(), s.rps.mean(), s.rps_book.mean(),
                     p['d'], p['ci95']))
    print('\ncache: %s' % cache)


if __name__ == '__main__':
    main()
