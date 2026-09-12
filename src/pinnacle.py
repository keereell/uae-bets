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
import json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
API = 'https://guest.api.arcadia.pinnacle.com/0.1'
KEY = 'CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R'
LEAGUE = 8126  # UAE - Pro League
HDRS = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'),
        'X-API-Key': KEY, 'Accept': 'application/json',
        'Origin': 'https://www.pinnacle.com', 'Referer': 'https://www.pinnacle.com/'}

# КЭШ И БЭКОФФ. Опрос дёргал parse() каждые 3 минуты -- два запроса за проход,
# ~40 в час с одного IP раннера GitHub. 11 сентября 2026 гостевой API отвечал
# 403 с 13:14 до 17:59, эталон был пуст, и единственный сигнал за весь тур
# ушёл против одного консенсуса -- ложный. Как эталон Pinnacle не нуждается
# в трёхминутной свежести: держим разбор 10 минут, при 403/429 ждём с
# удвоением паузы и отдаём последний удачный разбор, пока ему меньше 30 минут.
CACHE_TTL = 600          # секунд, сколько живёт удачный разбор
STALE_MAX = 1800         # секунд, до какого возраста разбор ещё годится при блокировке
_cache = dict(at=0.0, games=None, blocked_until=0.0, backoff=60.0)


def status():
    """Возраст последнего удачного разбора и состояние блокировки -- для логов."""
    now = time.time()
    return dict(age_s=(now - _cache['at']) if _cache['games'] is not None else None,
                stale=(_cache['games'] is not None and now - _cache['at'] > CACHE_TTL),
                blocked_s=max(0.0, _cache['blocked_until'] - now))


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


def parse(period=0, max_age=CACHE_TTL):
    """
    -> {matchup_id: {'home','away','start','moneyline','totals','spreads'}}

    Свежий разбор моложе max_age отдаётся из памяти без запросов. При
    HTTP 403/429 поднимается пауза (60 с, далее удвоение до 30 мин) и
    возвращается последний удачный разбор, если ему меньше STALE_MAX;
    иначе исключение летит наверх -- пустой эталон честнее устаревшего.
    """
    now = time.time()
    if _cache['games'] is not None and now - _cache['at'] <= max_age:
        return _cache['games']
    if now < _cache['blocked_until']:
        if _cache['games'] is not None and now - _cache['at'] <= STALE_MAX:
            return _cache['games']
        raise RuntimeError(f"Pinnacle заблокирован ещё {_cache['blocked_until']-now:.0f} с")
    try:
        games = _parse_live(period)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            _cache['blocked_until'] = now + _cache['backoff']
            _cache['backoff'] = min(_cache['backoff'] * 2, 1800.0)
            if _cache['games'] is not None and now - _cache['at'] <= STALE_MAX:
                return _cache['games']
        raise
    _cache.update(at=now, games=games, backoff=60.0, blocked_until=0.0)
    return games


def _parse_live(period=0):
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
