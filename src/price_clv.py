# -*- coding: utf-8 -*-
"""
CLV СИГНАЛОВ ЦЕНОВОГО ПУТИ ПО СНИМКАМ.

Что меряем. Для каждого отправленного сигнала (data/price_signals.csv) --
справедливую цену исхода в ПОСЛЕДНЕМ снимке до начала матча и матожидание
взятой цены против неё: CLV = цена_взятия * p_закрытия - 1. Это единственная
метрика, по которой на сотнях ставок можно отличить перевес от везения:
прибыль за сезон утонет в дисперсии, а CLV -- нет.

Закрытие берётся из снимков, а не живым запросом: снимки уже лежат в
data/snapshots/, а Pinnacle с раннеров GitHub отвечал 403 часами.
Эталон закрытия -- консенсус независимых операторов (тот же расчёт, что
и при поиске сигнала), плюс для сравнения цена той же конторы на закрытии.

    python src/price_clv.py
"""
import os, sys, csv, gzip, glob, json, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import pandas as pd
import consensus as C
from books import Quote

SIGNALS = os.path.join(ROOT, 'data', 'price_signals.csv')


def _ts(s):
    for fmt in ('%Y-%m-%dT%H:%MZ', '%Y-%m-%dT%H:%M:%SZ'):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def load_snapshots():
    recs = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'snapshots', '*.jsonl.gz'))):
        for l in gzip.open(f, 'rt', encoding='utf-8'):
            r = json.loads(l)
            recs.append((_ts(r['at']), r))
    recs.sort(key=lambda x: x[0])
    return recs


def closing(recs, home, away, sel, book, kickoff_ts):
    """-> (p_закрытия консенсуса, цена той же конторы, время снимка) либо (None, None, None)."""
    for t, r in reversed(recs):
        if t >= kickoff_ts:
            continue
        qs = [Quote(book=x['b'], home=x['h'], away=x['a'], sel=x['s'], price=x['p'], kickoff=x.get('k'))
              for x in r['quotes'] if (x['h'], x['a']) == (home, away)]
        if not qs:
            continue
        cons = C.build_consensus(qs)
        c = (cons.get((home, away)) or {}).get(sel)
        same = next((q.price for q in qs if q.book == book and q.sel == sel), None)
        return (c['p'] if c else None), same, r['at']
    return None, None, None


def result_1x2(home, away, sel, matches):
    m = matches[(matches.home == home) & (matches.away == away) & matches.hg.notna()]
    if m.empty or sel not in ('1', 'X', '2'):
        return None, None
    r = m.sort_values('ts').iloc[-1]
    out = '1' if r.hg > r.ag else ('X' if r.hg == r.ag else '2')
    return out == sel, f'{int(r.hg)}:{int(r.ag)}'


def main():
    if not os.path.exists(SIGNALS):
        print('журнал сигналов пуст'); return
    rows = list(csv.DictReader(open(SIGNALS, encoding='utf-8-sig')))
    recs = load_snapshots()
    matches = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    print('сигналов: %d, снимков: %d' % (len(rows), len(recs)))
    print()
    print('%-16s %-24s %-7s %-7s %6s %8s | %8s %8s %8s | %-6s %6s' % (
        'отправлен', 'матч', 'исход', 'контора', 'цена', 'справ.', 'закр.спр', 'закр.та же', 'CLV', 'счёт', 'P&L'))
    clvs, pnls = [], []
    for s in rows:
        ko = _ts(s.get('kickoff') or '')
        price = float(s['price'])
        p_close, same, at = closing(recs, s['home'], s['away'], s['sel'], s['book'], ko) if ko else (None, None, None)
        clv = (price * p_close - 1) if p_close else None
        won, score = result_1x2(s['home'], s['away'], s['sel'], matches)
        pnl = None if won is None else ((price - 1) if won else -1.0)
        if clv is not None: clvs.append(clv)
        if pnl is not None: pnls.append(pnl)
        print('%-16s %-24s %-7s %-7s %6.2f %8s | %8s %8s %8s | %-6s %6s' % (
            s['sent_at'], f"{s['home'][:10]} — {s['away'][:10]}", s['sel'], s['book'][:7], price, s['fair_price'],
            f'{1/p_close:.3f}' if p_close else '—', f'{same:.2f}' if same else '—',
            f'{100*clv:+.1f}%' if clv is not None else '—', score or '—',
            f'{pnl:+.2f}' if pnl is not None else '—'))
    if clvs:
        print()
        print('средний CLV против закрытия консенсуса: %+.2f%% (n=%d, положительных %d)'
              % (100 * sum(clvs) / len(clvs), len(clvs), sum(1 for c in clvs if c > 0)))
    if pnls:
        print('P&L по 1X2 (ставка 1 ед.): %+.2f на %d ставках' % (sum(pnls), len(pnls)))


if __name__ == '__main__':
    main()
