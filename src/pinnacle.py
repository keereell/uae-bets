# -*- coding: utf-8 -*-
"""
Загрузка линии Pinnacle по UAE Pro League.

Pinnacle — эталон остроты линии: минимальная маржа (2-3% на основных рынках),
не ограничивает выигрывающих игроков, и её закрывающая линия в академической
литературе используется как лучшая доступная оценка истинной вероятности.
Расхождение мягкой конторы с Pinnacle — это и есть настоящий валуй.

API публичное (используется самим сайтом Pinnacle), ключ статический.
Цены приходят в АМЕРИКАНСКОМ формате, period=0 — полный матч.
"""
import json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
API = 'https://guest.api.arcadia.pinnacle.com/0.1'
KEY = 'CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R'
LEAGUE = 8126  # UAE - Pro League
HDRS = {'User-Agent': 'Mozilla/5.0', 'X-API-Key': KEY, 'Accept': 'application/json'}


def get(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def american_to_decimal(a):
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def load(cache=True):
    os.makedirs(RAW, exist_ok=True)
    mp = os.path.join(RAW, 'pinnacle_matchups.json')
    kp = os.path.join(RAW, 'pinnacle_markets.json')
    if cache and os.path.exists(mp) and os.path.exists(kp):
        return json.load(open(mp, encoding='utf-8')), json.load(open(kp, encoding='utf-8'))
    m = get(f'{API}/leagues/{LEAGUE}/matchups')
    k = get(f'{API}/leagues/{LEAGUE}/markets/straight')
    json.dump(m, open(mp, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(k, open(kp, 'w', encoding='utf-8'), ensure_ascii=False)
    return m, k


def parse(period=0):
    """-> {matchup_id: {'home','away','start','moneyline','totals','spreads'}}"""
    matchups, markets = load(cache=False)
    games = {}
    for x in matchups:
        if x.get('parentId') or x.get('type') != 'matchup':
            continue
        parts = {p.get('alignment'): p.get('name') for p in x.get('participants', [])}
        games[x['id']] = dict(id=x['id'], start=x.get('startTime'),
                              home=parts.get('home'), away=parts.get('away'),
                              moneyline=None, totals={}, spreads={})
    for k in markets:
        gid = k.get('matchupId')
        if gid not in games or k.get('period') != period or k.get('status') != 'open':
            continue
        pr = {p['designation']: american_to_decimal(p['price']) for p in k.get('prices', [])
              if p.get('price') is not None}
        if k['type'] == 'moneyline' and not k.get('isAlternate', False):
            games[gid]['moneyline'] = pr
        elif k['type'] == 'total':
            line = None
            for p in k.get('prices', []):
                line = p.get('points', line)
            if line is not None:
                games[gid]['totals'][float(line)] = pr
        elif k['type'] == 'spread':
            pts = {}
            for p in k.get('prices', []):
                pts[p['designation']] = p.get('points')
            line = pts.get('home')
            if line is not None:
                games[gid]['spreads'][float(line)] = pr
    return games


if __name__ == '__main__':
    g = parse()
    for x in g.values():
        print('=' * 80)
        print(f"{x['home']} — {x['away']}  ({x['start']})")
        if x['moneyline']:
            ml = x['moneyline']
            s = sum(1 / v for v in ml.values())
            print(f"  1X2: 1={ml.get('home'):.3f}  X={ml.get('draw'):.3f}  2={ml.get('away'):.3f}"
                  f"   маржа {100*(s-1):.2f}%")
        for L in sorted(x['totals']):
            v = x['totals'][L]
            if 'over' in v and 'under' in v:
                s = 1 / v['over'] + 1 / v['under']
                print(f"  тотал {L:>5}: Бол {v['over']:.3f}  Мен {v['under']:.3f}  маржа {100*(s-1):.2f}%")
        for L in sorted(x['spreads']):
            v = x['spreads'][L]
            if 'home' in v and 'away' in v:
                s = 1 / v['home'] + 1 / v['away']
                print(f"  фора  {L:>5}: Ф1 {v['home']:.3f}  Ф2 {v['away']:.3f}  маржа {100*(s-1):.2f}%")
