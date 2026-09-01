# -*- coding: utf-8 -*-
"""
ИНКРЕМЕНТАЛЬНОЕ обновление данных. Работает без локального кэша.

В репозитории лежат только производные таблицы (data/matches.csv и
data/shots.csv, вместе ~230 КБ). Сырые JSON (16 МБ, 829 файлов) в git не
попадают: вместо них скрипт догружает только то, чего ещё нет.

  1. Тянет матчи 365scores за окно [сегодня - days_back, сегодня + 45]
  2. Обновляет строки в matches.csv (новые матчи, изменившиеся счета)
  3. Догружает поударные данные ТОЛЬКО для сыгранных матчей, которых нет
     в shots.csv
  4. Пересобирает объединённую таблицу

Так один прогон в CI делает 3-5 запросов вместо 800.
"""
import json, os, sys, time, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

from build_shots import metrics_from_game
import archive
from build_dataset import STAT_MAP

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
BASE = 'https://webws.365scores.com'
COMP = 549


def get(url, tries=3, timeout=45):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA,
                                                       'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            if i == tries - 1:
                print(f'  запрос не удался: {e}', file=sys.stderr)
                return None
            time.sleep(1.5 * (i + 1))


def fetch_games(days_back=60, days_fwd=45):
    """Матчи лиги в окне дат, кусками по 25 дней."""
    d0 = dt.date.today() - dt.timedelta(days=days_back)
    dend = dt.date.today() + dt.timedelta(days=days_fwd)
    games = {}
    while d0 <= dend:
        d1 = min(d0 + dt.timedelta(days=24), dend)
        url = (f'{BASE}/web/games/?appTypeId=5&langId=1&timezoneName=Asia/Dubai'
               f'&userCountryId=1&competitions={COMP}'
               f'&startDate={d0:%d/%m/%Y}&endDate={d1:%d/%m/%Y}&showOdds=true')
        d = get(url) or {}
        for g in d.get('games', []):
            games[g['id']] = g
        d0 = d1 + dt.timedelta(days=1)
    return games


def fetch_shots(gid):
    d = get(f'{BASE}/web/game/?appTypeId=5&langId=1&timezoneName=Asia/Dubai'
            f'&userCountryId=1&gameId={gid}')
    if d is None:
        return None
    g = d.get('game', {})
    return dict(id=gid, chartEvents=g.get('chartEvents'), events=g.get('events'))


def game_row(g):
    """Строка matches.csv из объекта матча 365scores."""
    h, a = g['homeCompetitor'], g['awayCompetitor']
    t = dt.datetime.fromisoformat(g['startTime'])
    played = g.get('statusGroup') == 4
    r = dict(game_id=g['id'], season=g['seasonNum'], round=g.get('roundNum', 0),
             date=t.date().isoformat(), ts=t.timestamp(),
             kickoff_hour=t.hour + t.minute / 60.0, weekday=t.weekday(),
             played=played, status=g.get('statusText'),
             home_id=h['id'], home=h['name'], away_id=a['id'], away=a['name'],
             hg=h.get('score') if played else None,
             ag=a.get('score') if played else None)
    o = g.get('odds') or {}
    r['book'] = (o.get('bookmaker') or {}).get('name')
    for op in o.get('options', []):
        nm = {1: 'H', 2: 'D', 3: 'A'}.get(op['num'])
        if nm:
            r['odds_' + nm] = (op.get('rate') or {}).get('decimal')
            r['open_' + nm] = (op.get('originalRate') or {}).get('decimal')
    return r


def main(days_back=60):
    mpath = os.path.join(DATA, 'matches.csv')
    spath = os.path.join(DATA, 'shots.csv')
    M = pd.read_csv(mpath) if os.path.exists(mpath) else pd.DataFrame()
    S = pd.read_csv(spath) if os.path.exists(spath) else pd.DataFrame(columns=['game_id'])
    print(f'было: матчей {len(M)}, с ударами {len(S)}')

    games = fetch_games(days_back)
    print(f'получено из API за окно {days_back} дней назад: {len(games)} матчей')
    if not games:
        print('API ничего не вернул — оставляю данные как есть')
        return M

    fresh = pd.DataFrame([game_row(g) for g in games.values()])
    # столбцы статистики матча в новых строках пустые: их даёт build_dataset
    # из кэша, которого в CI нет. Для модели важны поударные метрики.
    if len(M):
        keep = M[~M.game_id.isin(fresh.game_id)]
        cols = [c for c in M.columns if c in fresh.columns or c not in fresh.columns]
        M2 = pd.concat([keep, fresh], ignore_index=True)
        for c in M.columns:
            if c not in M2.columns:
                M2[c] = pd.NA
        M2 = M2[list(dict.fromkeys(list(M.columns) + list(fresh.columns)))]
    else:
        M2 = fresh
    M2 = M2.sort_values('ts').reset_index(drop=True)

    # --- поударные данные: только для сыгранных матчей, которых ещё нет
    have = set(S.game_id.astype(int)) if len(S) else set()
    need = [int(g) for g in M2[M2.played.fillna(False)].game_id if int(g) not in have]
    print(f'нужно догрузить ударов: {len(need)}')
    new_rows, new_raw = [], []
    if need:
        with ThreadPoolExecutor(max_workers=5) as ex:
            for slim in ex.map(fetch_shots, need):
                if not slim:
                    continue
                new_raw.append(slim)
                r = metrics_from_game(slim)
                if r:
                    new_rows.append(r)
    if new_raw:
        # сырые удары тоже кладём в репозиторий: 0.42 МБ в сжатом виде,
        # зато любую новую метрику можно посчитать без повторной выкачки
        total = archive.add(new_raw)
        print(f'в архив сырых ударов добавлено {len(new_raw)}, всего {total}')
    if new_rows:
        S = pd.concat([S, pd.DataFrame(new_rows)], ignore_index=True)
        S = S.drop_duplicates(subset=['game_id'], keep='last')
        S.to_csv(spath, index=False, encoding='utf-8-sig')
        print(f'добавлено матчей с ударами: {len(new_rows)}')

    # --- подмешать поударные метрики в matches.csv
    drop = [c for c in S.columns if c != 'game_id' and c in M2.columns]
    M2 = M2.drop(columns=drop).merge(
        S.drop(columns=[c for c in ('rec_h', 'rec_a') if c in S.columns]),
        on='game_id', how='left')
    M2.to_csv(mpath, index=False, encoding='utf-8-sig')
    played = int(M2.played.fillna(False).sum())
    print(f'стало: матчей {len(M2)}, сыграно {played}, '
          f'с xG с игры {int(M2.h_xg_open.notna().sum())}')
    return M2


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
