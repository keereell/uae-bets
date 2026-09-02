# -*- coding: utf-8 -*-
"""
ИНКРЕМЕНТАЛЬНОЕ обновление данных. Работает без локального кэша.

В репозитории лежат только производные таблицы (data/matches.csv и
data/shots.csv, вместе ~230 КБ). Сырые JSON (16 МБ, 829 файлов) в git не
попадают: вместо них скрипт догружает только то, чего ещё нет.

  1. Тянет матчи 365scores за окно [сегодня - days_back, сегодня + 45]
  2. ОБНОВЛЯЕТ в matches.csv только то, что даёт расписание: счёт, статус,
     дату, коэффициенты. Остальные колонки (статистика матча, xG, удары)
     НЕ ТРОГАЕТ — замена строки целиком стирала их у всех матчей окна.
  3. Догружает статистику матча (/web/game/stats/: xG, удары, владение...)
     для сыгранных матчей, у которых её ещё нет
  4. Догружает поударные данные для сыгранных матчей, которых нет в shots.csv
  5. Пересчитывает дни отдыха и пересобирает объединённую таблицу

Так один прогон в CI делает 3-10 запросов вместо 800.
"""
import json, os, sys, time, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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


def fetch_stats(gid):
    """Статистика матча (та же, что собирал crawl_365/build_dataset)."""
    d = get(f'{BASE}/web/game/stats/?appTypeId=5&langId=1&games={gid}')
    if not d:
        return None
    out = {}
    for st in d.get('statistics', []):
        key = STAT_MAP.get(st.get('name'))
        if not key:
            continue
        v = str(st.get('value', '')).replace('%', '').strip()
        try:
            v = float(v)
        except ValueError:
            continue
        out.setdefault(st.get('competitorId'), {})[key] = v
    return gid, out


def canonical_names(df):
    """
    Одно имя на команду. 365scores меняет написание между сезонами
    ('Al-Wahda' -> 'Wahda Abu Dhabi' при том же id 8350), и модель считает
    одну команду за две: 53 матча против 2, рейтинг новичка вместо настоящего.
    Каноническим берём самое частое имя для каждого id.
    """
    pairs = pd.concat([
        df[['home_id', 'home']].rename(columns={'home_id': 'id', 'home': 'name'}),
        df[['away_id', 'away']].rename(columns={'away_id': 'id', 'away': 'name'})])
    pairs = pairs.dropna()
    best = (pairs.groupby(['id', 'name']).size().reset_index(name='n')
            .sort_values(['id', 'n'], ascending=[True, False])
            .drop_duplicates('id').set_index('id')['name'].to_dict())
    before = set(df.home) | set(df.away)
    df['home'] = df['home_id'].map(best).fillna(df['home'])
    df['away'] = df['away_id'].map(best).fillna(df['away'])
    dropped = before - (set(df.home) | set(df.away))
    if dropped:
        print(f'приведены к каноническим именам: {sorted(dropped)}')
    return df


def rest_days(df):
    """Дни с предыдущего матча каждой команды в лиге (как в build_dataset)."""
    df = df.sort_values('ts').copy()
    last, hr, ar = {}, [], []
    for _, row in df.iterrows():
        for side, col in (('home_id', hr), ('away_id', ar)):
            prev = last.get(row[side])
            col.append(np.nan if prev is None else (row['ts'] - prev) / 86400.0)
        if bool(row['played']):
            last[row['home_id']] = row['ts']
            last[row['away_id']] = row['ts']
    df['h_rest'], df['a_rest'] = hr, ar
    return df


def game_row(g):
    """Строка matches.csv из объекта матча 365scores."""
    h, a = g['homeCompetitor'], g['awayCompetitor']
    t = dt.datetime.fromisoformat(g['startTime'])
    # statusGroup==4 значит «матч обработан», а не «сыгран»: перенесённые и
    # отменённые тоже попадают сюда, но со счётом -1. Такие строки уходили
    # в обучение и в оценку как ничьи 0:0 — три штуки уже лежали в данных.
    hs, as_ = h.get('score'), a.get('score')
    played = (g.get('statusGroup') == 4
              and str(g.get('statusText', '')).strip().lower() in (
                  'ended', 'final', 'ft', 'after et', 'after penalties')
              and hs is not None and as_ is not None
              and float(hs) >= 0 and float(as_) >= 0)
    r = dict(game_id=g['id'], season=g['seasonNum'], round=g.get('roundNum', 0),
             date=t.date().isoformat(), ts=t.timestamp(),
             kickoff_hour=t.hour + t.minute / 60.0, weekday=t.weekday(),
             played=played, status=g.get('statusText'),
             home_id=h['id'], home=h['name'], away_id=a['id'], away=a['name'],
             hg=hs if played else None,
             ag=as_ if played else None)
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
    API_COLS = [c for c in fresh.columns if c != 'game_id']
    if len(M):
        M2 = M.set_index('game_id')
        F = fresh.set_index('game_id')
        known = F.index.intersection(M2.index)
        new = F.index.difference(M2.index)
        # у известных матчей обновляем ТОЛЬКО поля расписания
        for c in API_COLS:
            if c not in M2.columns:
                M2[c] = np.nan
            M2.loc[known, c] = F.loc[known, c]
        # новые матчи добавляем целиком
        if len(new):
            M2 = pd.concat([M2, F.loc[new]])
        M2 = M2.reset_index()
        print(f'обновлено известных матчей: {len(known)}, добавлено новых: {len(new)}')
    else:
        M2 = fresh
    M2['played'] = M2['played'].fillna(False).astype(bool)
    M2 = canonical_names(M2)
    # Санитария существующих строк: API больше не отдаёт матчи, отменённые
    # давно, поэтому обновление их не чинит. Перенесённые попадали в CSV со
    # счётом -1:-1 и played=True -- в обучении это ничья 0:0, в оценке тоже.
    bad = (M2.hg.fillna(0) < 0) | (M2.ag.fillna(0) < 0) | (
        M2.played & ~M2.status.fillna('').str.strip().str.lower().isin(
            ['ended', 'final', 'ft', 'after et', 'after penalties']))
    if bad.any():
        print(f'снято с учёта как несыгранные: {int(bad.sum())} '
              f'({sorted(set(M2.loc[bad, "status"].fillna("?")))})')
        M2.loc[bad, ['played', 'hg', 'ag']] = [False, np.nan, np.nan]
    M2 = M2.sort_values('ts').reset_index(drop=True)

    # --- статистика матча: для сыгранных, у которых её нет
    # маркер «статистика уже есть» -- владение: оно есть у всех матчей со
    # статистикой, тогда как xG у ~30 старых матчей отсутствует, и по нему
    # эти матчи перекачивались бы каждый прогон
    if 'h_poss' not in M2.columns:
        M2['h_poss'] = np.nan
    need_stats = [int(g) for g, pl, x in zip(M2.game_id, M2.played, M2.h_poss)
                  if pl and pd.isna(x)]
    print(f'нужно догрузить статистику матча: {len(need_stats)}')
    if need_stats:
        hid = M2.set_index('game_id')['home_id'].to_dict()
        aid = M2.set_index('game_id')['away_id'].to_dict()
        got = 0
        with ThreadPoolExecutor(max_workers=5) as ex:
            for r in ex.map(fetch_stats, need_stats):
                if not r:
                    continue
                gid, st = r
                hs, as_ = st.get(hid.get(gid), {}), st.get(aid.get(gid), {})
                if not hs and not as_:
                    continue
                idx = M2.index[M2.game_id == gid]
                for k in set(STAT_MAP.values()):
                    if 'h_' + k not in M2.columns:
                        M2['h_' + k] = np.nan
                        M2['a_' + k] = np.nan
                    M2.loc[idx, 'h_' + k] = hs.get(k, np.nan)
                    M2.loc[idx, 'a_' + k] = as_.get(k, np.nan)
                got += 1
        print(f'статистика догружена: {got}')

    # --- поударные данные: только для сыгранных матчей, которых ещё нет
    have = set(S.game_id.astype(int)) if len(S) else set()
    have |= set(archive.load().keys())     # матч без ударов тоже в архиве -- не перекачивать
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
    M2 = rest_days(M2)
    M2.to_csv(mpath, index=False, encoding='utf-8-sig')
    played = int(M2.played.fillna(False).sum())
    print(f'стало: матчей {len(M2)}, сыграно {played}, '
          f'с xG с игры {int(M2.h_xg_open.notna().sum())}')
    return M2


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
