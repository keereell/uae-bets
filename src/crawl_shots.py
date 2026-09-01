# -*- coding: utf-8 -*-
"""
Загрузка ПОУДАРНЫХ данных с 365scores (endpoint /web/game/ -> chartEvents).

По каждому удару: xG, xGOT, минута, часть тела, координаты, тип ситуации
(Regular Play / From Corner / Free Kick / Fast Break / Assisted), исход.

Это позволяет посчитать то, чего нет в суммарной статистике матча:
  * NPxG — xG без пенальти (пенальти не характеризуют качество игры)
  * xG по игровому состоянию (при равном счёте / ведя / отыгрываясь)
  * xG с игры против xG со стандартов

    python src/crawl_shots.py [папка данных]
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
BASE = 'https://webws.365scores.com'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(1.0 + i)


def fetch(args):
    gid, outdir = args
    p = os.path.join(outdir, f'{gid}.json')
    if os.path.exists(p):
        return 'cached'
    d = get(f'{BASE}/web/game/?appTypeId=5&langId=1&timezoneName=Asia/Dubai'
            f'&userCountryId=1&gameId={gid}')
    if d is None:
        return 'fail'
    g = d.get('game', {})
    slim = dict(
        id=gid,
        chartEvents=g.get('chartEvents'),
        events=g.get('events'),
        venue=g.get('venue'),
        actualPlayTime=g.get('actualPlayTime'),
    )
    json.dump(slim, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 'ok'


def main(data_dir):
    raw = os.path.join(data_dir, 'raw')
    outdir = os.path.join(raw, 'shots')
    os.makedirs(outdir, exist_ok=True)
    games = json.load(open(os.path.join(raw, 'games.json'), encoding='utf-8'))
    ids = [g['id'] for g in games if g.get('statusGroup') == 4]
    print(f'матчей к загрузке: {len(ids)}')
    res = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(fetch, [(g, outdir) for g in ids]), 1):
            res.append(r)
            if i % 50 == 0:
                print(f'  {i}/{len(ids)}')
    import collections
    print(collections.Counter(res))
    print('файлов:', len(os.listdir(outdir)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data'))
