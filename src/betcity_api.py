# -*- coding: utf-8 -*-
"""
Снятие линии БЕТСИТИ напрямую, без сохранения страниц вручную.

Эндпоинт, который использует сам сайт:
    https://ad.betcity.ru/d/off/events?rev=6&add=dep_events&template=1&ver=88&csn=ooca9s

Отдаёт всю линию одним JSON. Нас интересует
    reply/sports/1/chmps/<id_чемпионата>/evts/<id_матча>/main
где группы рынков:
    69 — фактический исход (1X2)
    71 — фора
    72 — тотал

У каждого коэффициента есть поле md — момент последнего изменения цены.
Именно оно позволяет ловить главное: острая линия сдвинулась, а эта нет.

    python src/betcity_api.py            # линия ОАЭ сейчас
    python src/betcity_api.py 11183      # другой чемпионат по id
"""
import json, os, sys, time, urllib.request, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ('https://ad.betcity.ru/d/off/events?rev=6&add=dep_events'
       '&template=1&ver=88&csn=ooca9s')
UAE_CHAMP = '11183'
HDRS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'),
    'Accept': 'application/json',
    'Referer': 'https://betcity.ru/ru/line/soccer/' + UAE_CHAMP,
}


def fetch(tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL, headers=HDRS)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode('utf-8'))['reply']
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _kf(node):
    return float(node['kf']) if isinstance(node, dict) and 'kf' in node else None


def _md(node):
    return int(node.get('md', 0)) if isinstance(node, dict) else 0


def parse_champ(reply, champ=UAE_CHAMP):
    """-> список матчей с основной линией и временем последнего изменения цен."""
    ch = reply.get('sports', {}).get('1', {}).get('chmps', {}).get(champ)
    if not ch:
        return []
    out = []
    for ev_id, e in (ch.get('evts') or {}).items():
        rec = dict(id=e.get('id_ev'), home=e.get('name_ht'), away=e.get('name_at'),
                   start=dt.datetime.fromtimestamp(e['date_ev']).strftime('%Y-%m-%d %H:%M')
                   if e.get('date_ev') else None,
                   maximum=e.get('maximum'), markets={}, md=0)
        main = e.get('main') or {}
        for gid, g in main.items():
            data = (g.get('data') or {}).get(str(rec['id'])) or {}
            for bname, blk in (data.get('blocks') or {}).items():
                if not isinstance(blk, dict):
                    continue
                for oname, node in blk.items():
                    k = _kf(node)
                    if k is None:
                        continue
                    line = node.get('lv')
                    key = f'{g.get("name", gid)}|{oname}'
                    if line is not None:
                        key += f'({line:+g})' if oname.startswith(('Kf_F', 'F')) else f'({line:g})'
                    rec['markets'][key] = dict(kf=k, line=line, md=_md(node))
                    rec['md'] = max(rec['md'], _md(node))
        out.append(rec)
    return sorted(out, key=lambda r: r['start'] or '')


def snapshot(champ=UAE_CHAMP, save=True):
    reply = fetch()
    games = parse_champ(reply, champ)
    if save:
        d = os.path.join(ROOT, 'data', 'betcity')
        os.makedirs(d, exist_ok=True)
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        json.dump(dict(taken_at=reply.get('curr_time'), games=games),
                  open(os.path.join(d, f'{stamp}.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    return games


if __name__ == '__main__':
    champ = sys.argv[1] if len(sys.argv) > 1 else UAE_CHAMP
    games = snapshot(champ)
    now = time.time()
    print(f'матчей в линии: {len(games)}')
    for g in games:
        age = (now - g['md']) / 3600 if g['md'] else None
        print(f"\n{g['home']} — {g['away']}  ({g['start']}, максимум {g.get('maximum')})")
        print(f"  цены не менялись: {age:.1f} ч" if age else '  время изменения неизвестно')
        for k in sorted(g['markets']):
            if any(t in k for t in ('исход', 'Фора', 'Тотал')):
                print(f"    {k:34s} {g['markets'][k]['kf']:.2f}")
