# -*- coding: utf-8 -*-
"""
Универсальный краулер 365scores: любая лига, любой период.

    python src/crawl_any.py <competition_id> <папка> [дата_начала]

Пример: python src/crawl_any.py 89 data/rpl 2022-06-01
"""
import json, os, sys, time, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
BASE = 'https://webws.365scores.com'


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            if i == tries - 1:
                print('FAIL', url, e, file=sys.stderr)
                return None
            time.sleep(1.5 * (i + 1))


def crawl(comp, out, start, tz='Europe/Moscow', step=25):
    raw = os.path.join(out, 'raw')
    os.makedirs(os.path.join(raw, 'stats'), exist_ok=True)
    os.makedirs(os.path.join(raw, 'windows'), exist_ok=True)
    d0 = dt.date.fromisoformat(start)
    dend = dt.date.today() + dt.timedelta(days=45)
    games = {}
    while d0 <= dend:
        d1 = min(d0 + dt.timedelta(days=step - 1), dend)
        cache = os.path.join(raw, 'windows', f'{d0:%Y%m%d}_{d1:%Y%m%d}.json')
        if os.path.exists(cache):
            d = json.load(open(cache, encoding='utf-8'))
        else:
            url = (f'{BASE}/web/games/?appTypeId=5&langId=1&timezoneName={tz}&userCountryId=1'
                   f'&competitions={comp}&startDate={d0:%d/%m/%Y}&endDate={d1:%d/%m/%Y}&showOdds=true')
            d = get(url) or {}
            json.dump(d, open(cache, 'w', encoding='utf-8'), ensure_ascii=False)
        got = d.get('games', [])
        for g in got:
            games[g['id']] = g
        if got:
            print(f'{d0}..{d1}: +{len(got)} (всего {len(games)})')
        d0 = d1 + dt.timedelta(days=1)

    json.dump(list(games.values()), open(os.path.join(raw, 'games.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    ended = [g['id'] for g in games.values() if g.get('statusGroup') == 4]
    print('всего', len(games), '| сыграно', len(ended))

    def stats(gid):
        p = os.path.join(raw, 'stats', f'{gid}.json')
        if os.path.exists(p):
            return
        d = get(f'{BASE}/web/game/stats/?appTypeId=5&langId=1&games={gid}')
        if d is not None:
            json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(stats, ended))
    print('статистики закэшировано:', len(os.listdir(os.path.join(raw, 'stats'))))


if __name__ == '__main__':
    comp = int(sys.argv[1])
    out = sys.argv[2]
    start = sys.argv[3] if len(sys.argv) > 3 else '2023-07-01'
    os.makedirs(out, exist_ok=True)
    crawl(comp, out, start)
