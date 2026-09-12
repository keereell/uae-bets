# -*- coding: utf-8 -*-
"""
АУДИТ ОКОН ПО СНИМКАМ: сколько сигналов дало бы текущее правило на всех
сохранённых снимках и что стало с каждым к закрытию.

Зачем. После каждого тура нужен ответ не «что бот отправил», а «что он
ОТПРАВИЛ БЫ по нынешнему правилу» -- правило меняется, снимки нет. Для
каждого окна считается EV против закрытия консенсуса (последний снимок
до начала матча): если оно отрицательное, «перевес» был запаздыванием
эталона, а не ценой конторы.

Тур 5 (10-11 сентября 2026, 407 снимков): 9 окон, все -- один матч,
одни 16 минут, все от 0 до 4 минут жизни, у всех CLV от -1.4% до -11.1%.
Прямые конторы двигались синхронно, эталон отставал.

    python src/replay_windows.py                          # сводка
    python src/replay_windows.py "Khor Fakkan" Baniyas 1  # + ряд по исходу
"""
import sys, os, gzip, glob, json, datetime as dt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import consensus as C
from books import Quote, BETTABLE

def ts(s):
    return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).timestamp()

recs = []
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'snapshots', '*.jsonl.gz'))):
    for l in gzip.open(f, 'rt', encoding='utf-8'):
        recs.append(json.loads(l))
recs.sort(key=lambda r: r['at'])
print('снимков:', len(recs), recs[0]['at'], '..', recs[-1]['at'])

TARGET = tuple(sys.argv[1:4]) if len(sys.argv) >= 4 else None
series, hits_all, presence = [], [], defaultdict(int)
last_cons, ko_of = {}, {}
for i, r in enumerate(recs):
    now = ts(r['at'])
    qs = [Quote(book=x['b'], home=x['h'], away=x['a'], sel=x['s'], price=x['p'], kickoff=x.get('k'))
          for x in r['quotes']]
    ko = {}
    for q in qs:
        if q.kickoff:
            ko[(q.home, q.away)] = min(ko.get((q.home, q.away), q.kickoff), q.kickoff)
    ko_of.update(ko)
    live = {m for m, k in ko.items() if k <= now + 60} | ({(q.home, q.away) for q in qs} - set(ko))
    qs = [q for q in qs if (q.home, q.away) not in live]
    for b in {q.book for q in qs if q.book in BETTABLE}:
        presence[b] += 1
    if not qs:
        continue
    cons = C.build_consensus(qs)
    for m, c in cons.items():
        last_cons[m] = (now, c)
    val = C.find_value(qs, cons, pin_fair=None)
    for v in val:
        if v['hit']:
            hits_all.append(dict(at=r['at'], now=now, **{k: v[k] for k in ('home', 'away', 'sel', 'price', 'book', 'ev', 'ev_worst', 'n_books', 'fair_price')}))
    if not TARGET:
        continue
    m = TARGET[:2]
    tq = [q for q in qs if (q.home, q.away) == m and q.sel == TARGET[2] and q.book in BETTABLE]
    c = (cons.get(m) or {}).get(TARGET[2])
    if tq or c:
        series.append((r['at'][11:16], {q.book: q.price for q in tq}, (1 / c['p']) if c else None))
    if (i + 1) % 50 == 0:
        print('  ...', i + 1, file=sys.stderr)

print()
print('=== АДАПТЕРЫ: в скольких снимках присутствовали (из %d) ===' % len(recs))
for b in sorted(BETTABLE):
    print('  %-9s %4d' % (b, presence.get(b, 0)))

print()
print('=== ОКНА (сигналы по текущему правилу, все снимки) ===')
by_key = defaultdict(list)
for h in hits_all:
    by_key[(h['home'], h['away'], h['sel'], h['book'])].append(h)
print('  снимков с сигналом: %d из %d; уникальных окон (матч, исход, контора): %d' %
      (len({h['at'] for h in hits_all}), len(recs), len(by_key)))
for (h, a, sel, book), lst in sorted(by_key.items(), key=lambda kv: kv[1][0]['at']):
    ko = ko_of.get((h, a))
    close = last_cons.get((h, a), (None, {}))[1].get(sel)
    p_hit = lst[0]['price']
    clv = (p_hit * close['p'] - 1) if close else None
    mins = (lst[-1]['now'] - lst[0]['now']) / 60
    hrs = (ko - lst[0]['now']) / 3600 if ko else float('nan')
    print('  %s — %s  %-8s @%.2f %-8s %s..%s (%d сн., %.0f мин, за %.1f ч)  EV %+.2f%% худш %+.2f%% опер.%d | к закрытию: %s' % (
        h[:12], a[:12], sel, p_hit, book, lst[0]['at'][11:16], lst[-1]['at'][11:16], len(lst), mins, hrs,
        100 * lst[0]['ev'], 100 * lst[0]['ev_worst'], lst[0]['n_books'],
        f'{100*clv:+.2f}%' if clv is not None else '—'))

print()
if not TARGET:
    sys.exit(0)
print('=== %s — %s, %s: olimp против остальных и консенсуса ===' % TARGET)
print('%-6s %-6s %-44s %-8s %s' % ('время', 'olimp', 'остальные', 'справ.', 'EV olimp'))
prev = None
for at, prices, fair in series:
    ol = prices.get('olimp')
    others = ' '.join(f'{b[:4]}={p:.2f}' for b, p in sorted(prices.items()) if b != 'olimp')
    ev = (ol / fair - 1) if (ol and fair) else None
    row = (ol, fair, others)
    if row == prev and at[-1] not in '05':
        continue
    prev = row
    print('%-6s %-6s %-44s %-8s %s' % (at, f'{ol:.2f}' if ol else '—', others[:44], f'{fair:.3f}' if fair else '—',
                                       f'{100*ev:+.2f}%' if ev is not None else '—'))
