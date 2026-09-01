# -*- coding: utf-8 -*-
"""
Архив сырых поударных данных: data/shots_raw.jsonl.gz

Зачем он нужен, если есть shots.csv. В CSV лежат УЖЕ ПОСЧИТАННЫЕ метрики
(NPxG, xG с игры, xG при равном счёте и т.д.). Если завтра понадобится новая
метрика — скажем, xG по частям тела, по зонам поля или по 15-минуткам, — из
CSV её не достать, придётся заново выкачивать 374 матча. Архив хранит удары
целиком: 374 матча, 4.6 МБ текста, 0.42 МБ в сжатом виде. Это дешевле, чем
одна картинка, и делает проект полностью воспроизводимым из репозитория.

Формат: по одному JSON на строку, {'id':..., 'chartEvents':..., 'events':...}

    python src/archive.py pack     # собрать архив из data/raw/shots/*.json
    python src/archive.py info     # что внутри
"""
import gzip, json, os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, 'data', 'shots_raw.jsonl.gz')


def load():
    """-> {game_id: объект}. Пустой словарь, если архива нет."""
    if not os.path.exists(ARCHIVE):
        return {}
    out = {}
    with gzip.open(ARCHIVE, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('id') is not None:
                out[int(o['id'])] = o
    return out


def save(games):
    """games: {game_id: объект}. Пишет атомарно, отсортировав по id."""
    os.makedirs(os.path.dirname(ARCHIVE), exist_ok=True)
    tmp = ARCHIVE + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=9) as f:
        for gid in sorted(games):
            f.write(json.dumps(games[gid], ensure_ascii=False,
                               separators=(',', ':')) + '\n')
    os.replace(tmp, ARCHIVE)
    return len(games)


def add(objs):
    """Дописывает новые матчи в архив. -> сколько стало всего."""
    games = load()
    for o in objs:
        if o and o.get('id') is not None:
            games[int(o['id'])] = o
    return save(games)


def pack_from_dir(d=None):
    """Собирает архив из россыпи json-файлов (первичная упаковка)."""
    d = d or os.path.join(ROOT, 'data', 'raw', 'shots')
    objs = []
    for p in sorted(glob.glob(os.path.join(d, '*.json'))):
        try:
            objs.append(json.load(open(p, encoding='utf-8')))
        except Exception:
            pass
    n = add(objs)
    size = os.path.getsize(ARCHIVE)
    print(f'упаковано матчей: {len(objs)}, всего в архиве: {n}, '
          f'размер {size/1e6:.2f} МБ')
    return n


def info():
    g = load()
    if not g:
        print('архива нет')
        return
    shots = sum(len((o.get('chartEvents') or {}).get('events') or []) for o in g.values())
    print(f'матчей в архиве: {len(g)}, ударов: {shots}, '
          f'размер {os.path.getsize(ARCHIVE)/1e6:.2f} МБ')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'info'
    {'pack': pack_from_dir, 'info': info}.get(cmd, info)()
