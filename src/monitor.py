# -*- coding: utf-8 -*-
"""
МОНИТОР: БЕТСИТИ против справедливой линии Pinnacle, полностью автоматически.

    python src/monitor.py           # один прогон
    python src/monitor.py watch 600 # проверять каждые 10 минут

Логика. Валуй в мягкой конторе появляется не когда открывают линию, а когда
острый рынок УЖЕ сдвинулся, а мягкая контора ещё нет. Поэтому смотрим не на
абсолютные цены, а на разрыв между ними, и отслеживаем его во времени.

Обе стороны снимаются без участия человека:
  БЕТСИТИ  — ad.betcity.ru/d/off/events (тот же запрос, что делает их сайт)
  Pinnacle — guest.api.arcadia.pinnacle.com (публичный API их сайта)
"""
import os, sys, time, json, csv
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG, asian_handicap, asian_total, wdl
import betcity_api
import pinnacle
from sharp import pinnacle_constraints, fit_from_constraints, NAME_MAP
from teams import to_en

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'data', 'monitor_log.csv')
MIN_EV = 0.01          # порог, начиная с которого объявляем валуй


def sharp_lines():
    out = {}
    for g in pinnacle.parse().values():
        cons = pinnacle_constraints(g)
        if len(cons) >= 5:
            key = (NAME_MAP.get(g['home'], g['home']), NAME_MAP.get(g['away'], g['away']))
            out[key] = dict(fit=fit_from_constraints(cons), start=g.get('start'),
                            ml=g.get('moneyline'))
    return out


def price_from_key(M, key):
    """Справедливые (win, push, lose) по ключу рынка БЕТСИТИ."""
    p = wdl(M)
    if key.startswith('Фактический исход|'):
        o = key.split('|')[1]
        return {'P1': (p['H'], 0.0, 1 - p['H']),
                'X': (p['D'], 0.0, 1 - p['D']),
                'P2': (p['A'], 0.0, 1 - p['A'])}.get(o)
    if key.startswith('Двойной исход|'):
        o = key.split('|')[1]
        v = {'1X': p['H'] + p['D'], '12': p['H'] + p['A'], 'X2': p['D'] + p['A']}.get(o)
        return None if v is None else (v, 0.0, 1 - v)
    if key.startswith('Тотал|'):
        body = key.split('|')[1]
        try:
            L = float(body[body.index('(') + 1:body.index(')')])
        except ValueError:
            return None
        return asian_total(M, L, over=body.startswith('Tb'))
    if key.startswith('Фора|'):
        body = key.split('|')[1]
        try:
            L = float(body[body.index('(') + 1:body.index(')')])
        except ValueError:
            return None
        side = 'home' if 'F1' in body else 'away'
        return asian_handicap(M, L, side)
    return None


def scan(verbose=True):
    games = betcity_api.snapshot()
    sharp = sharp_lines()
    now = time.time()
    rows, hits = [], []
    for g in games:
        h, a = to_en(g['home']), to_en(g['away'])
        ent = sharp.get((h, a))
        if not ent:
            if verbose:
                print(f"  {g['home']} — {g['away']}: Pinnacle не торгует, пропуск")
            continue
        M = ent['fit']['M']
        rmse = ent['fit']['rmse']
        for key, mk in g['markets'].items():
            pr = price_from_key(M, key)
            if pr is None or pr[0] <= 1e-6:
                continue
            w, pu, l = pr
            ev = w * mk['kf'] - (1 - pu)
            rows.append(dict(ts=dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
                             матч=f"{g['home']} — {g['away']}", старт=g['start'],
                             рынок=key, кэф=mk['kf'], справедливый=round(1 + l / w, 3),
                             EV=round(ev, 4), rmse=round(rmse, 4),
                             цена_не_менялась_ч=round((now - mk['md']) / 3600, 1) if mk['md'] else None))
            if ev > MIN_EV and ev > 2 * rmse:
                hits.append(rows[-1])
    if rows:
        new = not os.path.exists(LOG)
        with open(LOG, 'a', encoding='utf-8-sig', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wr.writeheader()
            wr.writerows(rows)
    return rows, hits


def report(rows, hits):
    print(f'\n{dt.datetime.now():%Y-%m-%d %H:%M}  проверено исходов: {len(rows)}')
    if not rows:
        print('  нет данных')
        return
    best = sorted(rows, key=lambda r: -r['EV'])[:5]
    print('  ближе всего к справедливой цене:')
    for r in best:
        st = 'ВАЛУЙ' if r in hits else '     '
        print(f"    {st} {r['матч'][:34]:34s} {r['рынок'][:26]:26s} "
              f"@{r['кэф']:5.2f}  справ. {r['справедливый']:5.2f}  "
              f"EV {100*r['EV']:+6.2f}%  цена стоит {r['цена_не_менялась_ч']} ч")
    if hits:
        print(f'\n  >>> НАЙДЕНО {len(hits)} ВАЛУЙНЫХ ИСХОДОВ <<<')
        for r in hits:
            print(f"      {r['матч']} | {r['рынок']} @{r['кэф']} | "
                  f"справедливый {r['справедливый']} | EV {100*r['EV']:+.2f}%")
    else:
        print('\n  валуя нет: ни один исход не превышает справедливую цену '
              f'больше чем на удвоенную погрешность подгонки')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        period = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        print(f'слежу за линией, проверка каждые {period} с. Ctrl+C чтобы выйти.')
        while True:
            try:
                report(*scan())
            except Exception as e:
                print('ошибка:', e)
            time.sleep(period)
    else:
        report(*scan())
