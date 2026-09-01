# -*- coding: utf-8 -*-
"""Собирает из сырых JSON один плоский датасет data/matches.csv."""
import json, os, datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('DATA_DIR') or os.path.join(ROOT, 'data')
RAW  = os.path.join(DATA, 'raw')

STAT_MAP = {
    'Expected Goals': 'xg', 'Expected Goals On Target': 'xgot',
    'Expected Goals Conceded': 'xgc', 'Expected Assists': 'xa',
    'Total Shots': 'shots', 'Shots On Target': 'sot', 'Big Chances Created': 'bigch',
    'Possession': 'poss', 'Corners': 'corners', 'Fouls': 'fouls',
    'Yellow Cards': 'yellow', 'Red Cards': 'red', 'Goalkeeper Saves': 'saves',
    'Shots Inside The Box': 'shots_ib', 'Attacks': 'attacks',
}

def load_stats(gid):
    p = os.path.join(RAW, 'stats', f'{gid}.json')
    if not os.path.exists(p): return {}
    try: j = json.load(open(p, encoding='utf-8'))
    except Exception: return {}
    out = {}
    for s in j.get('statistics', []):
        key = STAT_MAP.get(s['name'])
        if not key: continue
        v = str(s.get('value', '')).replace('%', '').strip()
        try: v = float(v)
        except ValueError: continue
        out.setdefault(s['competitorId'], {})[key] = v
    return out

def main():
    games = json.load(open(os.path.join(RAW, 'games.json'), encoding='utf-8'))
    rows = []
    for g in games:
        h, a = g['homeCompetitor'], g['awayCompetitor']
        t = dt.datetime.fromisoformat(g['startTime'])          # уже в Asia/Dubai
        st = load_stats(g['id'])
        hs, as_ = st.get(h['id'], {}), st.get(a['id'], {})
        played = g.get('statusGroup') == 4
        r = dict(
            game_id=g['id'], season=g['seasonNum'], round=g.get('roundNum', 0),
            date=t.date().isoformat(), ts=t.timestamp(),
            kickoff_hour=t.hour + t.minute / 60.0, weekday=t.weekday(),
            played=played, status=g.get('statusText'),
            home_id=h['id'], home=h['name'], away_id=a['id'], away=a['name'],
            hg=h.get('score') if played else None,
            ag=a.get('score') if played else None,
        )
        for k in set(STAT_MAP.values()):
            r['h_' + k] = hs.get(k); r['a_' + k] = as_.get(k)
        o = g.get('odds') or {}
        book = (o.get('bookmaker') or {}).get('name')
        r['book'] = book
        for op in o.get('options', []):
            nm = {1: 'H', 2: 'D', 3: 'A'}.get(op['num'])
            if not nm: continue
            r['odds_' + nm] = (op.get('rate') or {}).get('decimal')
            r['open_' + nm] = (op.get('originalRate') or {}).get('decimal')
        rows.append(r)

    df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    # дни отдыха с предыдущего матча в лиге (по каждой команде)
    last = {}
    hr, ar = [], []
    for i, row in df.iterrows():
        for side, col in (('home_id', hr), ('away_id', ar)):
            tid = row[side]
            prev = last.get(tid)
            col.append(None if prev is None else (row['ts'] - prev) / 86400.0)
        if row['played']:
            last[row['home_id']] = row['ts']; last[row['away_id']] = row['ts']
    df['h_rest'] = hr; df['a_rest'] = ar
    sp = os.path.join(DATA, 'shots.csv')
    if os.path.exists(sp):
        S = pd.read_csv(sp)
        S = S.drop(columns=[c for c in ('rec_h', 'rec_a') if c in S.columns])
        df = df.merge(S, on='game_id', how='left')
        print('подмешаны поударные метрики:', int(df.h_npxg.notna().sum()), 'матчей')
    out = os.path.join(DATA, 'matches.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('rows', len(df), '| played', int(df.played.sum()), '| with xG',
          int(df.h_xg.notna().sum()), '->', out)
    print(df.groupby('season').agg(n=('game_id', 'size'), xg=('h_xg', 'count'),
                                   odds=('odds_H', 'count')))
    return df

if __name__ == '__main__':
    main()
