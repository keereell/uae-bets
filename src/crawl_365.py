# -*- coding: utf-8 -*-
"""
Краулер UAE Pro League (365scores, competition=549).

1) Сметает ленту матчей окнами по датам (endpoint /web/games/?startDate&endDate)
2) Для каждого сыгранного матча тянет командную статистику /web/game/stats/ (там xG)
3) Кэширует сырые JSON в data/raw, чтобы не долбить API повторно
"""
import json, os, sys, time, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, 'data', 'raw')
os.makedirs(os.path.join(RAW, 'stats'), exist_ok=True)
os.makedirs(os.path.join(RAW, 'windows'), exist_ok=True)

UA   = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
BASE = 'https://webws.365scores.com'
COMP = 549

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            if i == tries - 1:
                print('FAIL', url, e, file=sys.stderr); return None
            time.sleep(1.5 * (i + 1))

def window(d0, d1):
    """Матчи в окне дат [d0, d1] включительно."""
    tag = f"{d0:%Y%m%d}_{d1:%Y%m%d}"
    cache = os.path.join(RAW, 'windows', tag + '.json')
    if os.path.exists(cache):
        try: return json.load(open(cache, encoding='utf-8'))
        except Exception: pass
    url = (f'{BASE}/web/games/?appTypeId=5&langId=1&timezoneName=Asia/Dubai&userCountryId=1'
           f'&competitions={COMP}&startDate={d0:%d/%m/%Y}&endDate={d1:%d/%m/%Y}&showOdds=true')
    d = get(url) or {}
    json.dump(d, open(cache, 'w', encoding='utf-8'), ensure_ascii=False)
    return d

def crawl_games(start='2021-07-01', end=None, step_days=25):
    d0 = dt.date.fromisoformat(start)
    dend = dt.date.fromisoformat(end) if end else dt.date.today() + dt.timedelta(days=45)
    games = {}
    while d0 <= dend:
        d1 = min(d0 + dt.timedelta(days=step_days - 1), dend)
        d = window(d0, d1)
        got = d.get('games', [])
        for g in got: games[g['id']] = g
        if got: print(f'{d0}..{d1}: +{len(got)} (total {len(games)})')
        d0 = d1 + dt.timedelta(days=1)
    return games

def fetch_stats(gid):
    cache = os.path.join(RAW, 'stats', f'{gid}.json')
    if os.path.exists(cache):
        try: return json.load(open(cache, encoding='utf-8'))
        except Exception: pass
    d = get(f'{BASE}/web/game/stats/?appTypeId=5&langId=1&games={gid}')
    if d is not None:
        json.dump(d, open(cache, 'w', encoding='utf-8'), ensure_ascii=False)
    return d

if __name__ == '__main__':
    start = sys.argv[1] if len(sys.argv) > 1 else '2021-07-01'
    games = crawl_games(start)
    json.dump(list(games.values()), open(os.path.join(RAW, 'games.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    ended = [g['id'] for g in games.values() if g.get('statusGroup') == 4]
    print('total games', len(games), '| ended', len(ended))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fetch_stats, ended))
    print('stats cached:', len(os.listdir(os.path.join(RAW, 'stats'))))
