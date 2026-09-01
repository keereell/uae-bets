# -*- coding: utf-8 -*-
"""
Финальный отчёт по туру: два независимых списка ставок.

СПИСОК А — «расхождение модели с линией» (методика из роликов).
СПИСОК Б — «БЕТСИТИ против Pinnacle» (методика Buchdahl 2017: справедливые
           вероятности Pinnacle как эталон; на 24 150 ставках даёт
           ожидаемые +1.60% и фактические +1.81% годовых к обороту).

Список А приводится для сравнения и НЕ является рекомендацией:
walk-forward показал, что перевес модели над линией не конвертируется в прибыль.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers, score_matrix
from markets import wdl, totals, btts, DEVIG, margin, kelly
from backtest import walk_forward, summarize
from calibrate import fit_shrink, apply_shrink
from parse_betcity import load_all
from pricing import price_bet
from predict import best_params, MAIN_MARKETS
from teams import to_en
import pinnacle
from sharp import pinnacle_constraints, fit_from_constraints, NAME_MAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 42)

EDGE_LO, EDGE_HI = 0.03, 0.12   # «коридор»: меньше — шум, больше — ошибка модели


def build():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    params = best_params()
    wf = walk_forward(df, params)
    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    cal['k_s'] = float(np.clip(cal['k_s'], 0.0, 1.5))
    cal['k_d'] = float(np.clip(cal['k_d'], 0.3, 1.5))
    nc = detect_newcomers(df)
    up = df[~df.played].sort_values('ts')
    mdl = DixonColes(**params).fit(df, ref_ts=float(up.ts.min()), newcomers=nc[max(nc)])
    return df, wf, cal, mdl, up, summarize(wf)


def main():
    df, wf, cal, mdl, up, s = build()
    print('#' * 118)
    print('ОТЧЁТ ПО ТУРУ — UAE PRO LEAGUE'.center(118))
    print('#' * 118)
    print(f"\nКачество модели вне выборки ({s['n']} матчей): RPS {s['rps']:.4f}"
          + (f" | линия Bet365: {s['rps_book']:.4f}" if 'rps_book' in s else '')
          + '   <- рынок точнее модели')

    # эталон Pinnacle
    try:
        pin = pinnacle.parse()
    except Exception as e:
        print('Pinnacle недоступен:', e)
        pin = {}
    sharp = {}
    for g in pin.values():
        cons = pinnacle_constraints(g)
        if len(cons) >= 5:
            sharp[(NAME_MAP.get(g['home'], g['home']),
                   NAME_MAP.get(g['away'], g['away']))] = fit_from_constraints(cons)

    A, B = [], []
    for head, brows in load_all():
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        if h not in mdl.idx or a not in mdl.idx:
            continue
        lh, la = mdl.lambdas(h, a)
        x, y = apply_shrink(np.array([lh]), np.array([la]),
                            cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        Mm = score_matrix(float(x[0]), float(y[0]), mdl.rho)
        Ms = sharp.get((h, a), {}).get('M')

        ml = head.get('main', {})
        mr = list(brows)
        if all(k in ml for k in ('1', 'X', '2')):
            for nm in ('1', 'X', '2'):
                mr.append(dict(market='1X2', line='', outcome=nm, price=ml[nm]))
        if ml.get('tot_line') is not None:
            mr.append(dict(market='Тотал', line=f"{ml['tot_line']:g}", outcome='Мен', price=ml['under']))
            mr.append(dict(market='Тотал', line=f"{ml['tot_line']:g}", outcome='Бол', price=ml['over']))
        for kl, ko, tag in (('h1_line', 'h1', 'Ф1'), ('h2_line', 'h2', 'Ф2')):
            if ml.get(kl) is not None:
                v = ml[kl]
                mr.append(dict(market='Фора', line='',
                               outcome=f"{tag} ({'+' if v >= 0 else ''}{v:g})", price=ml[ko]))

        for r in mr:
            pm = price_bet(Mm, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            if pm and pm[0] > 1e-6:
                w, pu, l = pm
                A.append(dict(матч=f'{h_ru} — {a_ru}', рынок=r['market'], линия=r['line'],
                              исход=r['outcome'], кэф=r['price'],
                              модель=w, справ=1 + l / w, перевес=w * r['price'] - (1 - pu),
                              тип='основной' if r['market'] in MAIN_MARKETS else 'производный'))
            if Ms is not None:
                ps = price_bet(Ms, r['market'], r['line'], r['outcome'], h_ru, a_ru)
                if ps and ps[0] > 1e-6:
                    w, pu, l = ps
                    B.append(dict(матч=f'{h_ru} — {a_ru}', рынок=r['market'], линия=r['line'],
                                  исход=r['outcome'], кэф=r['price'],
                                  pinnacle=w, справ=1 + l / w, перевес=w * r['price'] - (1 - pu),
                                  тип='основной' if r['market'] in MAIN_MARKETS else 'производный'))

    A = pd.DataFrame(A)
    B = pd.DataFrame(B)

    print('\n' + '=' * 118)
    print('СПИСОК А — «валуй» по расхождению модели с линией (методика из роликов)')
    print('=' * 118)
    a1 = A[(A['кэф'].between(1.35, 7.0)) & (A['тип'] == 'основной')
           & (A['перевес'].between(EDGE_LO, EDGE_HI))].sort_values('перевес', ascending=False)
    a2 = A[(A['кэф'].between(1.35, 7.0)) & (A['перевес'] > EDGE_HI)].sort_values('перевес', ascending=False)
    print(f'\nв «коридоре» {EDGE_LO:.0%}–{EDGE_HI:.0%} (то, что рекомендуют брать):')
    print(a1[['матч', 'рынок', 'линия', 'исход', 'кэф', 'справ', 'модель', 'перевес']]
          .head(15).round(3).to_string(index=False) if len(a1) else '  пусто')
    print(f'\nотброшено как «слишком жирно» (>{EDGE_HI:.0%}) — {len(a2)} исходов, топ-8:')
    print(a2[['матч', 'рынок', 'линия', 'исход', 'кэф', 'справ', 'модель', 'перевес']]
          .head(8).round(3).to_string(index=False) if len(a2) else '  пусто')

    print('\n' + '=' * 118)
    print('СПИСОК Б — БЕТСИТИ против справедливой линии Pinnacle (проверенная методика)')
    print('=' * 118)
    if B.empty:
        print('  нет линии Pinnacle')
    else:
        b1 = B[(B['кэф'].between(1.30, 10.0)) & (B['перевес'] > 0)].sort_values('перевес', ascending=False)
        if b1.empty:
            print('\n  ВАЛУЯ НЕТ. Ни один из', len(B), 'исходов БЕТСИТИ не дороже справедливой цены Pinnacle.')
            print('  Максимум по основным рынкам (наименьшая переплата):')
            near = B[(B['тип'] == 'основной') & (B['кэф'].between(1.3, 10))].sort_values('перевес', ascending=False)
            print(near[['матч', 'рынок', 'линия', 'исход', 'кэф', 'справ', 'перевес']]
                  .head(10).round(4).to_string(index=False))
        else:
            b1 = b1.copy()
            b1['Келли_%'] = [100 * kelly(r.pinnacle, r.кэф, frac=0.25, cap=0.02) for r in b1.itertuples()]
            print(b1[['матч', 'рынок', 'линия', 'исход', 'кэф', 'справ', 'перевес', 'тип', 'Келли_%']]
                  .head(25).round(4).to_string(index=False))

    # пересечение
    if not A.empty and not B.empty:
        key = ['матч', 'рынок', 'линия', 'исход', 'кэф']
        M = A.merge(B[key + ['pinnacle', 'перевес']], on=key, suffixes=('_модель', '_pinnacle'))
        both = M[(M['перевес_модель'] > EDGE_LO) & (M['перевес_pinnacle'] > 0)]
        print('\n' + '=' * 118)
        print('ПЕРЕСЕЧЕНИЕ ОБОИХ СПИСКОВ (единственное, что стоило бы играть)')
        print('=' * 118)
        print(both.round(4).to_string(index=False) if len(both) else '  пусто')

    A.to_csv(os.path.join(ROOT, 'data', 'list_A_model.csv'), index=False, encoding='utf-8-sig')
    B.to_csv(os.path.join(ROOT, 'data', 'list_B_pinnacle.csv'), index=False, encoding='utf-8-sig')
    print('\nСохранено: data/list_A_model.csv, data/list_B_pinnacle.csv')


if __name__ == '__main__':
    main()
