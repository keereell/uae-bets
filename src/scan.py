# -*- coding: utf-8 -*-
"""
ЕДИНАЯ ТОЧКА ВХОДА. Запускать перед каждым игровым туром:

    python src/scan.py            # полный цикл
    python src/scan.py --nofetch  # без обновления статистики (быстрее)

Что делает:
  1. Догружает свежие матчи и статистику xG с 365scores
  2. Пересобирает датасет и переобучает модель
  3. Тянет актуальную линию Pinnacle (эталон честных вероятностей)
  4. Разбирает сохранённые страницы БЕТСИТИ из папки Bets/
  5. Выдаёт: прогноз на все будущие матчи + список валуйных ставок
"""
import sys, os, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 44)


def refresh():
    print('>>> обновляю данные с 365scores ...')
    subprocess.run([sys.executable, os.path.join(HERE, 'crawl_365.py')], check=False)
    subprocess.run([sys.executable, os.path.join(HERE, 'build_dataset.py')], check=False)


def main():
    if '--nofetch' not in sys.argv:
        refresh()

    from model import DixonColes, detect_newcomers, score_matrix
    from markets import wdl, totals, btts, asian_handicap, kelly, margin, DEVIG
    from backtest import walk_forward, summarize
    from calibrate import fit_shrink, apply_shrink
    from predict import best_params, MAIN_MARKETS
    from parse_betcity import load_all
    from pricing import price_bet
    from teams import to_en
    import pinnacle
    from sharp import pinnacle_constraints, fit_from_constraints, NAME_MAP

    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    params = best_params()
    wf = walk_forward(df, params)
    s = summarize(wf)
    print(f"\n>>> качество модели на проверке вперёд по времени ({s['n']} матчей): "
          f"RPS {s['rps']:.4f}"
          + (f" | Bet365 {s['rps_book']:.4f}" if 'rps_book' in s else ''))

    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    cal['k_s'] = float(np.clip(cal['k_s'], 0.0, 1.5))
    cal['k_d'] = float(np.clip(cal['k_d'], 0.3, 1.5))
    nc = detect_newcomers(df)
    up = df[~df.played].sort_values('ts')
    mdl = DixonColes(**params).fit(df, ref_ts=float(up.ts.min()), newcomers=nc[max(nc)])

    # ---------- Pinnacle
    try:
        pin = pinnacle.parse()
    except Exception as e:
        print('!! Pinnacle недоступен:', e)
        pin = {}
    sharp = {}
    for g in pin.values():
        cons = pinnacle_constraints(g)
        if len(cons) >= 5:
            f = fit_from_constraints(cons)
            sharp[(NAME_MAP.get(g['home'], g['home']), NAME_MAP.get(g['away'], g['away']))] = f

    # ---------- прогноз
    print('\n' + '=' * 118)
    print('ПРОГНОЗ НА БЛИЖАЙШИЕ МАТЧИ  (кэф = справедливый коэффициент, без маржи)')
    print('=' * 118)
    out = []
    for _, r in up.iterrows():
        if r.home not in mdl.idx or r.away not in mdl.idx:
            continue
        lh, la = mdl.lambdas(r.home, r.away)
        a, b = apply_shrink(np.array([lh]), np.array([la]), cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        M = score_matrix(float(a[0]), float(b[0]), mdl.rho)
        p = wdl(M)
        row = dict(дата=r.date, матч=f'{r.home} — {r.away}',
                   λ1=round(float(a[0]), 2), λ2=round(float(b[0]), 2),
                   мод_1=round(1 / p['H'], 2), мод_X=round(1 / p['D'], 2), мод_2=round(1 / p['A'], 2),
                   ТБ25=round(totals(M, 2.5)[0], 3), ОЗ=round(btts(M)[0], 3))
        key = (r.home, r.away)
        if key in sharp:
            ps = wdl(sharp[key]['M'])
            row.update(пин_1=round(1 / ps['H'], 2), пин_X=round(1 / ps['D'], 2), пин_2=round(1 / ps['A'], 2))
        out.append(row)
    F = pd.DataFrame(out)
    print(F.to_string(index=False))
    F.to_csv(os.path.join(ROOT, 'data', 'forecast.csv'), index=False, encoding='utf-8-sig')

    # ---------- поиск валуя
    print('\n' + '=' * 118)
    print('ПОИСК ВАЛУЯ В СОХРАНЁННЫХ СТРАНИЦАХ БЕТСИТИ')
    print('=' * 118)
    rows = []
    for head, brows in load_all():
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        if (h, a) not in sharp:
            print(f'  {h_ru} — {a_ru}: нет линии Pinnacle -> пропускаю '
                  f'(без острого эталона оценка ненадёжна)')
            continue
        S = sharp[(h, a)]['M']
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
            ps = price_bet(S, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            if ps is None or ps[0] <= 1e-6:
                continue
            w, pu, l = ps
            rows.append(dict(матч=f'{h_ru} — {a_ru}', рынок=r['market'], линия=r['line'],
                             исход=r['outcome'], кэф=r['price'], честный=round(1 + l / w, 3),
                             EV=w * r['price'] - (1 - pu), p=w,
                             тип='основной' if r['market'] in MAIN_MARKETS else 'производный'))
    V = pd.DataFrame(rows)
    if V.empty:
        print('  Нечего оценивать.')
        return
    V.to_csv(os.path.join(ROOT, 'data', 'vs_pinnacle.csv'), index=False, encoding='utf-8-sig')

    val = V[(V.EV > 0.005) & (V['кэф'].between(1.30, 10.0))].sort_values('EV', ascending=False)
    print(f'\nоценено исходов: {len(V)}')
    if val.empty:
        print('\n  >>> ВАЛУЙНЫХ СТАВОК НЕТ. Линия БЕТСИТИ не отстаёт от Pinnacle.')
    else:
        val = val.copy()
        val['Келли_%'] = [100 * kelly(r.p, r.кэф, frac=0.25, cap=0.02) for r in val.itertuples()]
        print('\n  >>> НАЙДЕНО:')
        print(val[['матч', 'рынок', 'линия', 'исход', 'кэф', 'честный', 'EV', 'тип', 'Келли_%']]
              .head(30).round(4).to_string(index=False))

    print('\nБЛИЖЕ ВСЕГО К ЧЕСТНОЙ ЦЕНЕ (наименьшая переплата, основные рынки):')
    near = V[(V['тип'] == 'основной') & (V['кэф'].between(1.3, 10))].sort_values('EV', ascending=False)
    print(near[['матч', 'рынок', 'линия', 'исход', 'кэф', 'честный', 'EV']].head(12).round(4).to_string(index=False))
    val.to_csv(os.path.join(ROOT, 'data', 'value_vs_pinnacle.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
