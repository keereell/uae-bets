# -*- coding: utf-8 -*-
"""
КОНСЕНСУС МНОГИХ КОНТОР И ПОИСК ПЕРЕВЕСА ПО ЦЕНЕ.

Зачем не модель. На 717 исходах walk-forward проверено: там, где xG-модель
расходится с рынком, ближе к факту оказывается рынок В КАЖДОМ бакете
расхождения (на краях модель говорит 50.9% при рынке 36.6% и факте 36.4%).
Ставки по перевесу модели дали ROI -23.1%. Поэтому здесь модели нет вообще:
перевес ищется сравнением ЦЕН, а не прогнозом.

Так устроены обе стратегии с опубликованной доходностью:
  * Kaunitz и др. (2017): консенсус 32 контор, ставка там, где МАКСИМАЛЬНАЯ
    цена выше консенсусной справедливой. Прогнозной модели нет вовсе.
    R^2 консенсуса с исходами 0.999 -- он калиброван лучше любой одной конторы.
  * Buchdahl: цена мягкой конторы выше справедливой цены Pinnacle,
    24 150 ставок, +1.81%.

Мы используем ОБА эталона сразу и берём более осторожный. Причина: у каждого
своя слепая зона. Pinnacle точнее всех, но не котирует производные рынки;
консенсус покрывает всё, но его тянут вниз клоны и мягкие конторы.

    python src/consensus.py            # снять всё и показать перевесы
    python src/consensus.py --save     # то же + записать снимок в журнал
"""
import os
import sys
import time
import json
import gzip
import datetime as dt
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from books import fetch_all, BETTABLE, Quote          # noqa: E402
from markets import DEVIG                             # noqa: E402

SNAPSHOTS = os.path.join(ROOT, 'data', 'odds_snapshots.jsonl.gz')

DEVIG3 = 'shin'      # трёхсторонние рынки: Штрумбель (2014) на 37 турнирах
DEVIG2 = 'power'     # двусторонние: степенной и мультипликативный почти совпадают

# Минимальное число контор, чтобы консенсусу вообще можно было верить.
# При двух-трёх конторах медиана -- это не консенсус, а шум.
MIN_BOOKS = 5

# Порог перевеса. Стартовое значение, а не истина: 1% -- это измеренный
# разброс между пятью методами снятия маржи на прямых ценах Pinnacle,
# то есть погрешность самого измерения. Ниже неё «перевес» неотличим от
# способа счёта. Уточняется по накопленным снимкам, а не подбирается.
THETA = 0.01


# ---------------------------------------------------------------------------
#                     РАЗБОР ИСХОДОВ НА ЗАМКНУТЫЕ РЫНКИ
# ---------------------------------------------------------------------------
def _parse_sel(sel):
    """'O2.5' -> ('total', 2.5, 'over'); 'AH1-0.25' -> ('ah', -0.25, 1); ..."""
    if sel in ('1', 'X', '2'):
        return ('1x2', None, sel)
    if sel in ('1X', '12', 'X2'):
        return ('dc', None, sel)
    if sel[:1] in 'OU':
        try:
            return ('total', float(sel[1:]), 'over' if sel[0] == 'O' else 'under')
        except ValueError:
            return (None, None, None)
    if sel[:2] in ('AH', 'EH'):
        try:
            team = int(sel[2])
            return ('ah' if sel[:2] == 'AH' else 'eh', float(sel[3:]), team)
        except (ValueError, IndexError):
            return (None, None, None)
    return (None, None, None)


def devig_one_book(quotes):
    """
    Котировки ОДНОЙ конторы на ОДИН матч -> {sel: справедливая вероятность}.

    Снимать маржу можно только с ЗАМКНУТОГО рынка, где исходы образуют полную
    группу: 1X2 целиком, пара тотала на одной линии, пара форы на зеркальных
    линиях. Одиночная цена без пары непригодна -- из неё нельзя вычесть маржу,
    и попытка это сделать даёт систематическую ошибку в свою пользу.
    """
    by = {q.sel: q.price for q in quotes}
    out = {}

    if all(k in by for k in ('1', 'X', '2')):
        q = DEVIG[DEVIG3]([by['1'], by['X'], by['2']])
        if not any(np.isnan(q)):
            out['1'], out['X'], out['2'] = float(q[0]), float(q[1]), float(q[2])

    tot = defaultdict(dict)
    ah = defaultdict(dict)
    for sel, price in by.items():
        kind, line, side = _parse_sel(sel)
        if kind == 'total':
            tot[line][side] = price
        elif kind in ('ah', 'eh'):
            ah[(kind, abs(line) if side == 1 else abs(line))][side] = (price, line)

    for line, sides in tot.items():
        if 'over' in sides and 'under' in sides:
            q = DEVIG[DEVIG2]([sides['over'], sides['under']])
            if not any(np.isnan(q)):
                out[f'O{line:g}'] = float(q[0])
                out[f'U{line:g}'] = float(q[1])

    # Фора: 'AH1-0.5' замыкается с 'AH2+0.5'. Ключ группировки -- модуль линии.
    pairs = defaultdict(dict)
    for sel, price in by.items():
        kind, line, team = _parse_sel(sel)
        if kind in ('ah', 'eh'):
            pairs[(kind, abs(line))][team] = (price, line)
    for (kind, _mag), sides in pairs.items():
        if 1 in sides and 2 in sides:
            (p1, l1), (p2, l2) = sides[1], sides[2]
            if abs(l1 + l2) > 1e-9:          # линии обязаны быть зеркальными
                continue
            q = DEVIG[DEVIG2]([p1, p2])
            if not any(np.isnan(q)):
                pre = 'AH' if kind == 'ah' else 'EH'
                out[f'{pre}1{"+" if l1 >= 0 else "-"}{abs(l1):g}'] = float(q[0])
                out[f'{pre}2{"+" if l2 >= 0 else "-"}{abs(l2):g}'] = float(q[1])
    return out


def build_consensus(quotes):
    """
    Все котировки -> {(home, away): {sel: {'p': медиана, 'n': контор,
                                           'spread': разброс}}}

    Медиана, а не среднее: клоны одного оператора (1xbet=22bet=megapari и
    подобные) и просто кривые цены не должны тянуть оценку. Медиана к ним
    устойчива, среднее -- нет.
    """
    by_match_book = defaultdict(lambda: defaultdict(list))
    for q in quotes:
        by_match_book[(q.home, q.away)][q.book].append(q)

    out = {}
    for match, books in by_match_book.items():
        probs = defaultdict(list)
        for book, qs in books.items():
            for sel, p in devig_one_book(qs).items():
                if 0.001 < p < 0.999:
                    probs[sel].append(p)
        agg = {}
        for sel, vals in probs.items():
            if len(vals) >= MIN_BOOKS:
                a = np.array(vals)
                agg[sel] = dict(p=float(np.median(a)), n=len(vals),
                                spread=float(a.max() - a.min()))
        if agg:
            out[match] = agg
    return out


# ---------------------------------------------------------------------------
#                              ПОИСК ПЕРЕВЕСА
# ---------------------------------------------------------------------------
def find_value(quotes, consensus, pin_fair=None, theta=THETA, bettable=BETTABLE):
    """
    -> список находок, отсортированный по перевесу.

    Правило: берём МАКСИМАЛЬНУЮ цену среди контор, где можно поставить,
    и сравниваем со справедливой ценой. Справедливая -- более осторожная
    из двух: консенсуса многих контор и прямой цены Pinnacle.

    Осторожная = дающая МЕНЬШУЮ вероятность, то есть более высокую
    справедливую цену. Так порог труднее пройти случайно: чтобы ставка
    прошла, она должна быть выгодна против ОБОИХ эталонов сразу.
    """
    best = defaultdict(lambda: (0.0, None))
    for q in quotes:
        if q.book not in bettable:
            continue
        k = (q.home, q.away, q.sel)
        if q.price > best[k][0]:
            best[k] = (q.price, q.book)

    out = []
    for (h, a, sel), (price, book) in best.items():
        c = (consensus.get((h, a)) or {}).get(sel)
        pf = (pin_fair.get((h, a)) or {}).get(sel) if pin_fair else None
        if c is None and pf is None:
            continue
        cands = [x for x in (c['p'] if c else None, pf) if x is not None]
        p_fair = min(cands)                    # осторожная оценка
        src = ('консенсус' if (c and p_fair == c['p']) else 'Pinnacle')
        ev = price * p_fair - 1.0
        out.append(dict(
            home=h, away=a, sel=sel, price=price, book=book,
            p_fair=p_fair, fair_price=1.0 / p_fair, ev=ev, ref=src,
            n_books=(c['n'] if c else 0),
            spread=(c['spread'] if c else None),
            p_cons=(c['p'] if c else None), p_pin=pf,
            hit=ev > theta))
    return sorted(out, key=lambda r: -r['ev'])


# ---------------------------------------------------------------------------
#                                 ЖУРНАЛ
# ---------------------------------------------------------------------------
def save_snapshot(quotes, errs):
    """
    Пишем СЫРЫЕ котировки, а не выводы. Пороги и методы снятия маржи ещё
    будут меняться; переоценить их задним числом можно только по сырым
    ценам. Один снимок -- одна строка gzip-jsonl.
    """
    rec = dict(
        at=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        errors=errs,
        quotes=[dict(b=q.book, h=q.home, a=q.away, s=q.sel, p=q.price,
                     k=q.kickoff) for q in quotes],
    )
    os.makedirs(os.path.dirname(SNAPSHOTS), exist_ok=True)
    with gzip.open(SNAPSHOTS, 'at', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(rec['quotes'])


def pinnacle_fair():
    """Прямые справедливые вероятности Pinnacle -> {(home, away): {sel: p}}."""
    try:
        import pinnacle
        from sharp import NAME_MAP
    except Exception:
        return {}
    out = {}
    try:
        games = pinnacle.parse()
    except Exception:
        return {}
    for g in games.values():
        h = NAME_MAP.get(g['home'], g['home'])
        a = NAME_MAP.get(g['away'], g['away'])
        d = {}
        ml = g.get('moneyline') or {}
        if all(k in ml for k in ('home', 'draw', 'away')):
            q = DEVIG[DEVIG3]([ml['home'], ml['draw'], ml['away']])
            if not any(np.isnan(q)):
                d['1'], d['X'], d['2'] = float(q[0]), float(q[1]), float(q[2])
        for L, v in (g.get('totals') or {}).items():
            if 'over' in v and 'under' in v:
                q = DEVIG[DEVIG2]([v['over'], v['under']])
                if not any(np.isnan(q)):
                    d[f'O{float(L):g}'] = float(q[0])
                    d[f'U{float(L):g}'] = float(q[1])
        for L, v in (g.get('spreads') or {}).items():
            if 'home' in v and 'away' in v:
                q = DEVIG[DEVIG2]([v['home'], v['away']])
                if not any(np.isnan(q)):
                    L = float(L)
                    d[f'AH1{"+" if L >= 0 else "-"}{abs(L):g}'] = float(q[0])
                    d[f'AH2{"+" if -L >= 0 else "-"}{abs(L):g}'] = float(q[1])
        if d:
            out[(h, a)] = d
    return out


def main():
    t0 = time.time()
    print('Снимаю линии...')
    quotes, errs = fetch_all()
    print(f'\nвсего котировок: {len(quotes)}, за {time.time()-t0:.0f} с')
    if errs:
        print('ошибки:', ', '.join(f'{k}' for k in errs))

    books_ok = sorted({q.book for q in quotes})
    bet_ok = sorted({q.book for q in quotes if q.book in BETTABLE})
    print(f'источников с данными: {len(books_ok)}, из них доступных для ставки: {len(bet_ok)}')
    print(f'  ставить можно в: {", ".join(bet_ok) or "нет"}')

    cons = build_consensus(quotes)
    pin = pinnacle_fair()
    print(f'матчей в консенсусе: {len(cons)}, у Pinnacle: {len(pin)}')

    val = find_value(quotes, cons, pin)
    hits = [v for v in val if v['hit']]
    print(f'\nисходов с ценой: {len(val)}, прошли порог +{100*THETA:.1f}%: {len(hits)}')
    print()
    if not val:
        print('нечего показывать')
        return
    print('%-34s %-10s %7s %-10s %8s %8s %5s' %
          ('матч', 'исход', 'кэф', 'контора', 'справ.', 'EV', 'контор'))
    for v in val[:25]:
        mark = '  <<<' if v['hit'] else ''
        print('%-34s %-10s %7.2f %-10s %8.3f %+7.2f%% %5d%s' % (
            f"{v['home'][:15]} — {v['away'][:15]}", v['sel'], v['price'],
            v['book'][:10], v['fair_price'], 100 * v['ev'], v['n_books'], mark))

    if '--save' in sys.argv:
        n = save_snapshot(quotes, errs)
        print(f'\nснимок записан: {n} котировок -> {SNAPSHOTS}')


if __name__ == '__main__':
    main()
