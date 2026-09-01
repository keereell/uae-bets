# -*- coding: utf-8 -*-
"""
СТОРОЖ. Дешёвая проверка: изменилось ли что-нибудь, ради чего стоит
запускать полный расчёт.

Зачем отдельно. Полный прогон переобучает модель заново перед каждым игровым
днём (около сотни подгонок) и занимает ~2 минуты. Гонять его каждые 20 минут
означает выйти за бесплатный лимит GitHub Actions. Сторож делает три запроса,
работает на голой стандартной библиотеке (без pip install) и укладывается
в ~15 секунд, а полный расчёт запускается только когда есть повод.

Поводы:
  * НОВЫЙ МАТЧ    — в расписании 365scores появилась игра, которой не было
  * ОТКРЫЛАСЬ ЛИНИЯ — БЕТСИТИ начала торговать матч, которого у неё не было
  * СДВИГ ОСТРОЙ ЛИНИИ — Pinnacle подвинул цену больше чем на порог. Это
    главный повод: валуй у мягкой конторы живёт ровно в том окне, когда
    острый рынок уже уехал, а мягкий ещё не догнал.
  * СДВИГ МЯГКОЙ ЛИНИИ — БЕТСИТИ подвинула цену: прежний расчёт устарел

    python src/watch.py            # проверить и напечатать вердикт
    python src/watch.py --json     # то же в json (для CI)
"""
import json, os, sys, urllib.request, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, 'state.json')

MOVE_SHARP = 0.02       # 2% по коэффициенту Pinnacle
MOVE_SOFT = 0.02        # 2% по коэффициенту БЕТСИТИ

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
S365 = ('https://webws.365scores.com/web/games/?appTypeId=5&langId=1'
        '&timezoneName=Asia/Dubai&userCountryId=1&competitions=549'
        '&startDate={d0}&endDate={d1}')
BETCITY = ('https://ad.betcity.ru/d/off/events?rev=6&add=dep_events'
           '&template=1&ver=88&csn=ooca9s')
PIN = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/8126/markets/straight'
PIN_MATCHUPS = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/8126/matchups'
PIN_KEY = 'CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R'


def get(url, headers=None, timeout=40):
    h = {'User-Agent': UA, 'Accept': 'application/json'}
    h.update(headers or {})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        print(f'  запрос не удался: {url[:60]}... {e}', file=sys.stderr)
        return None


def fixtures_365():
    d0 = dt.date.today()
    d1 = d0 + dt.timedelta(days=21)
    d = get(S365.format(d0=f'{d0:%d/%m/%Y}', d1=f'{d1:%d/%m/%Y}')) or {}
    return {str(g['id']): f"{g['homeCompetitor']['name']} — {g['awayCompetitor']['name']}"
            for g in d.get('games', [])}


def line_betcity():
    d = get(BETCITY, timeout=90)
    if not d:
        return {}
    ch = (d.get('reply', {}).get('sports', {}).get('1', {})
          .get('chmps', {}).get('11183'))
    if not ch:
        return {}
    out = {}
    for e in (ch.get('evts') or {}).values():
        key = f"{e.get('name_ht')} — {e.get('name_at')}"
        prices = {}
        for g in (e.get('main') or {}).values():
            data = (g.get('data') or {}).get(str(e.get('id_ev'))) or {}
            for blk in (data.get('blocks') or {}).values():
                if not isinstance(blk, dict):
                    continue
                for oname, node in blk.items():
                    if isinstance(node, dict) and 'kf' in node:
                        prices[f"{g.get('name', '')}|{oname}"] = float(node['kf'])
        if prices:
            out[key] = prices
    return out


def line_pinnacle():
    hdr = {'X-API-Key': PIN_KEY}
    mt = get(PIN_MATCHUPS, hdr) or []
    names = {}
    for x in mt:
        if x.get('parentId') or x.get('type') != 'matchup':
            continue
        p = {q.get('alignment'): q.get('name') for q in x.get('participants', [])}
        names[x['id']] = f"{p.get('home')} — {p.get('away')}"
    mk = get(PIN, hdr) or []
    out = {}
    for k in mk:
        if k.get('period') != 0 or k.get('type') != 'moneyline' or k.get('isAlternate'):
            continue
        nm = names.get(k.get('matchupId'))
        if not nm:
            continue
        out[nm] = {p['designation']: float(p['price']) for p in k.get('prices', [])
                   if p.get('price') is not None}
    return out


def _moved(old, new, thr):
    """Максимальное относительное изменение цены по общим исходам."""
    worst, where = 0.0, None
    for k, v in (new or {}).items():
        o = (old or {}).get(k)
        if o in (None, 0) or v in (None, 0):
            continue
        d = abs(v - o) / abs(o)
        if d > worst:
            worst, where = d, k
    return (worst >= thr), worst, where


def check():
    st = {}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE, encoding='utf-8'))
        except Exception:
            st = {}
    prev = st.get('watch', {})

    fx = fixtures_365()
    bc = line_betcity()
    pn = line_pinnacle()

    reasons = []
    new_fx = sorted(set(fx) - set(prev.get('fixtures', {})))
    if new_fx and prev.get('fixtures'):
        reasons.append(f"новых матчей в расписании: {len(new_fx)} "
                       f"({', '.join(fx[i] for i in new_fx[:3])})")

    new_line = sorted(set(bc) - set(prev.get('betcity', {})))
    if new_line and prev.get('betcity'):
        reasons.append(f"БЕТСИТИ открыла линию: {', '.join(new_line[:3])}")

    for nm, prices in pn.items():
        moved, d, where = _moved((prev.get('pinnacle') or {}).get(nm), prices, MOVE_SHARP)
        if moved:
            reasons.append(f"острая линия сдвинулась: {nm} {where} на {100*d:.1f}%")

    for nm, prices in bc.items():
        moved, d, where = _moved((prev.get('betcity') or {}).get(nm), prices, MOVE_SOFT)
        if moved:
            reasons.append(f"БЕТСИТИ подвинула цену: {nm} {where} на {100*d:.1f}%")

    first_run = not prev
    if first_run:
        reasons.append('первый запуск сторожа, снимаю базовый снимок')

    st['watch'] = dict(fixtures=fx, betcity=bc, pinnacle=pn,
                       checked_at=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'))
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)

    return dict(changed=bool(reasons), reasons=reasons,
                n_fixtures=len(fx), n_betcity=len(bc), n_pinnacle=len(pn))


if __name__ == '__main__':
    r = check()
    if '--json' in sys.argv:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print(f"матчей в расписании {r['n_fixtures']}, в линии БЕТСИТИ {r['n_betcity']}, "
              f"у Pinnacle {r['n_pinnacle']}")
        if r['changed']:
            print('ЕСТЬ ИЗМЕНЕНИЯ:')
            for x in r['reasons']:
                print('  -', x)
        else:
            print('изменений нет, полный расчёт не нужен')
    # код выхода 10 = есть изменения; так CI решает, запускать ли полный прогон
    sys.exit(10 if r['changed'] else 0)
