# -*- coding: utf-8 -*-
"""
xG С ПОПРАВКОЙ НА ИГРОВОЙ КОНТЕКСТ.

Сырой xG за матч смешивает две разные вещи: насколько команда СИЛЬНА и в
каком положении она провела матч. Команда, поведшая 2:0 на двадцатой минуте,
садится назад и до свистка почти не атакует. Её xG за матч выйдет низким --
не потому что она играла плохо, а потому что ей не нужно было играть иначе.
Рейтинг, подогнанный на сыром xG, спишет это в слабость атаки.

Skripnikov, Cemek & Gillman (2026) оценили мультипликаторы приведения к
базовому состоянию на поминутных данных пяти лиг за 15 сезонов. У них
скорректированные удары коррелируют с очками за сезон на 0.787 против
0.718 у сырых, скорректированные угловые -- 0.737 против 0.603.

Здесь то же самое применяется к каждому удару по состоянию НА МОМЕНТ УДАРА:
счёт с точки зрения бьющей команды и разница в удалениях.

    python src/xg_adjust.py            # построить data/xg_adj.csv
    python src/xg_adjust.py --check    # заодно сверить с сырым xG
"""
import os, sys, gzip, json, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from build_shots import _t, PENALTY, OWN_GOAL, PEN_COORD, SP_SUBTYPES, OPEN_STRICT

RAW = os.path.join(ROOT, 'data', 'shots_raw.jsonl.gz')
OUT = os.path.join(ROOT, 'data', 'xg_adj.csv')

# Мультипликаторы приведения к базовому состоянию (Skripnikov и др., 2026).
# Больше единицы -- состояние ПОДАВЛЯЛО генерацию моментов, и наблюдённый xG
# надо поднять; меньше единицы -- состояние её раздувало.
SCORE_MULT = {3: 1.06, 2: 1.07, 1: 1.08, 0: 1.00,
              -1: 0.89, -2: 0.85, -3: 0.87}
RED_MULT = {0: 1.00, -1: 1.86, -2: 3.20}


def _mult(state, red):
    s = SCORE_MULT.get(int(np.clip(state, -3, 3)), 1.00)
    r = RED_MULT.get(int(np.clip(red, -2, 0)), 1.00)
    # Игра в большинстве -- зеркало игры в меньшинстве. Отдельной оценки
    # у авторов нет, берём обратную величину, чтобы поправка не создавала
    # xG из воздуха на уровне матча.
    if red > 0:
        r = 1.0 / RED_MULT.get(int(np.clip(-red, -2, 0)), 1.00)
    return s * r


def red_timeline(d):
    """-> список (минута, competitorNum) удалений."""
    out = []
    for e in (d.get('events') or []):
        t = (e.get('eventType') or {})
        if t.get('id') != 3:
            continue
        gt = e.get('gameTime')
        cid = e.get('competitorId')
        if gt is None or cid is None:
            continue
        out.append((float(gt), int(cid)))
    return out


def one(d, home_id=None, away_id=None):
    ce = d.get('chartEvents') or {}
    shots = ce.get('events') or []
    if not shots:
        return None
    norm = []
    for sh in shots:
        try:
            xg = float(sh.get('xg') or 0)
        except (TypeError, ValueError):
            xg = 0.0
        st = sh.get('subType')
        coord = (round(sh.get('line', -1), 1), round(sh.get('side', -1), 1))
        pen = (st == PENALTY) or (coord == PEN_COORD and 0.68 <= xg <= 0.88)
        norm.append(dict(cn=sh.get('competitorNum') or 1, xg=xg, st=st, pen=pen,
                         own=(st == OWN_GOAL), t=_t(sh.get('time')) or 0.0,
                         goal=(sh.get('outcome') or {}).get('name') == 'Goal'))
    norm.sort(key=lambda s: s['t'])

    reds = red_timeline(d)
    # competitorId -> competitorNum. Порядок в events совпадает с нумерацией
    # в chartEvents: первый по появлению competitorId -- хозяева.
    ids = []
    for e in (d.get('events') or []):
        c = e.get('competitorId')
        if c is not None and c not in ids:
            ids.append(c)
    cn_of = {}
    if home_id is not None and away_id is not None:
        cn_of = {int(home_id): 1, int(away_id): 2}
    elif len(ids) >= 2:
        cn_of = {ids[0]: 1, ids[1]: 2}

    agg = {1: collections.defaultdict(float), 2: collections.defaultdict(float)}
    sc = {1: 0, 2: 0}
    for sh in norm:
        cn = sh['cn']
        state = sc[cn] - sc[3 - cn]
        nr = {1: 0, 2: 0}
        for t, cid in reds:
            k = cn_of.get(cid)
            if k and t <= sh['t']:
                nr[k] += 1
        red = -(nr[cn]) + nr[3 - cn]          # + значит соперник в меньшинстве
        m = _mult(state, red)
        if not sh['pen'] and not sh['own'] and sh['st'] not in SP_SUBTYPES:
            agg[cn]['xg_open'] += sh['xg']
            agg[cn]['xg_open_adj'] += sh['xg'] * m
            if sh['st'] in OPEN_STRICT:
                agg[cn]['xg_openstr'] += sh['xg']
                agg[cn]['xg_openstr_adj'] += sh['xg'] * m
        if sh['goal']:
            sc[3 - cn if sh['own'] else cn] += 1

    keys = ('xg_open', 'xg_open_adj', 'xg_openstr', 'xg_openstr_adj')
    return dict(game_id=d.get('id'),
                **{f'h_{k}': agg[1][k] for k in keys},
                **{f'a_{k}': agg[2][k] for k in keys})


def build():
    m = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    ids = m.set_index('game_id')[['home_id', 'away_id']].to_dict('index')
    rows = []
    for l in gzip.open(RAW, 'rt', encoding='utf-8'):
        d = json.loads(l)
        g = ids.get(d.get('id')) or {}
        r = one(d, g.get('home_id'), g.get('away_id'))
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'сохранено: {OUT}  матчей {len(out)}')
    return out


if __name__ == '__main__':
    o = build()
    if '--check' in sys.argv:
        for side in ('h', 'a'):
            raw, adj = o[f'{side}_xg_open'], o[f'{side}_xg_open_adj']
            print('%s: сырой %.3f -> скорректированный %.3f (%+.1f%%), корреляция %.4f'
                  % (side, raw.mean(), adj.mean(), 100 * (adj.mean() / raw.mean() - 1),
                     raw.corr(adj)))
        d = (o.h_xg_open_adj - o.h_xg_open)
        print('сдвиг по матчам: sd %.3f, от %+.2f до %+.2f' % (d.std(), d.min(), d.max()))
