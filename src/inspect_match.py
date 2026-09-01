# -*- coding: utf-8 -*-
"""Подробный разбор одного матча: почему исходы прошли или не прошли отбор."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_report import build, EDGE_LO, EDGE_HI, ODDS_LO, ODDS_HI, TOTALS_FAMILY
from model import score_matrix
from markets import wdl, DEVIG, margin
from calibrate import apply_shrink
from parse_betcity import load_all
from pricing import price_bet
from implied import fit_implied
from round_calib import fit_round_calibration, apply_round_calibration
from teams import to_en
import pinnacle
from sharp import pinnacle_constraints, fit_from_constraints, NAME_MAP
pd.set_option('display.width', 220)

want = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''
df, m, cal, up, wf = build()
pin = pinnacle.parse()
sharp = {}
for g in pin.values():
    c = pinnacle_constraints(g)
    if len(c) >= 5:
        sharp[(NAME_MAP.get(g['home'], g['home']), NAME_MAP.get(g['away'], g['away']))] = fit_from_constraints(c)

pages = load_all()
ml_pairs, mk_pairs, keys = [], [], []
for head, brows in pages:
    h, a = to_en(head.get('home')), to_en(head.get('away'))
    if h not in m.idx or a not in m.idx:
        continue
    imp = fit_implied(brows, head.get('main', {}))
    if imp is None:
        continue
    lh0, la0 = m.lambdas(h, a)
    x, y = apply_shrink(np.array([lh0]), np.array([la0]), cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
    ml_pairs.append((float(x[0]), float(y[0]))); mk_pairs.append((imp['lh'], imp['la'])); keys.append((h, a))
wf_sh = wf.copy()
_lh, _la = apply_shrink(wf.lh.values, wf.la.values, cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
wf_sh['lh'], wf_sh['la'] = _lh, _la
dmu, dgam, _ = fit_round_calibration(ml_pairs, mk_pairs, wf=wf_sh)

for head, brows in pages:
    h_ru, a_ru = head.get('home'), head.get('away')
    if want and want.lower() not in (h_ru + ' ' + a_ru).lower():
        continue
    h, a = to_en(h_ru), to_en(a_ru)
    if h not in m.idx:
        continue
    lh0, la0 = m.lambdas(h, a)
    x, y = apply_shrink(np.array([lh0]), np.array([la0]), cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
    lh, la = apply_round_calibration(float(x[0]), float(y[0]), dmu, dgam)
    M = score_matrix(lh, la, m.rho)
    p = wdl(M)
    Ms = sharp.get((h, a), {}).get('M')
    ml = head.get('main', {})
    print('=' * 108)
    print(f'{h_ru} — {a_ru}   модель ждёт {lh:.2f}:{la:.2f}')
    if all(k in ml for k in ('1', 'X', '2')):
        o = [ml['1'], ml['X'], ml['2']]
        q = DEVIG['power'](o)
        print(f'  линия {o}, маржа {100*margin(o):.1f}%')
        print(f'  рынок без маржи : 1={q[0]:.3f}  X={q[1]:.3f}  2={q[2]:.3f}')
        print(f'  модель          : 1={p["H"]:.3f}  X={p["D"]:.3f}  2={p["A"]:.3f}')
    mr = list(brows)
    if all(k in ml for k in ('1', 'X', '2')):
        for nm in ('1', 'X', '2'):
            mr.append(dict(market='1X2', line='', outcome=nm, price=ml[nm]))
    rows = []
    for r in mr:
        pm = price_bet(M, r['market'], r['line'], r['outcome'], h_ru, a_ru)
        if pm is None or pm[0] <= 1e-6 or not (ODDS_LO <= r['price'] <= ODDS_HI):
            continue
        w, pu, l = pm
        edge = w - (1 - pu) / r['price']
        ev = w * r['price'] - (1 - pu)
        pin_ev = np.nan
        if Ms is not None:
            s = price_bet(Ms, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            if s and s[0] > 1e-6:
                pin_ev = s[0] * r['price'] - (1 - s[1])
        rows.append(dict(рынок=r['market'][:22], исход=r['outcome'][:16], кэф=r['price'],
                         модель=round(100*w, 1), перевес_пп=round(100*edge, 1),
                         EV_пр=round(100*ev, 1), рынок_пр=round(100*pin_ev, 1)))
    R = pd.DataFrame(rows).sort_values('EV_пр', ascending=False)
    print(f'\n  топ-10 по матожиданию (EV), а не по перевесу в п.п.:')
    print(R.head(10).to_string(index=False))
    inc = R[(R.перевес_пп >= 100*EDGE_LO) & (R.перевес_пп <= 100*EDGE_HI)]
    print(f'\n  прошли коридор {EDGE_LO:.0%}-{EDGE_HI:.0%} по п.п.: {len(inc)} из {len(R)}')
    print(f'  но с EV выше +5%: {len(R[R.EV_пр > 5])};  выше +10%: {len(R[R.EV_пр > 10])}')
