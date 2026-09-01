# -*- coding: utf-8 -*-
"""
Из поударных данных считает продвинутые метрики матча.

Типы ситуаций (recovered из chartEvents.eventSubTypes):
    0 Fast Break, 1 Set Piece, 2 From Corner, 3 Free Kick,
    4 Regular Play, 7 Assisted, 8 Throw In, 9 Penalty, 10 Own Goal
    None -- подтип не проставлен (около 12.6% ударов)

По каждой команде:
  xg_all      суммарный xG по ударам
  npxg        xG без пенальти
  xg_open     xG с игры: всё, кроме пенальти, стандартов и автоголов
              (подтипы 0/4/7 и непроставленные)
  xg_openstr  строгий вариант: только подтипы 0/4/7
  xg_sp       xG со стандартов (1/2/3/8)
  xg_level    xG при равном счёте
  xgot_sum    xG в створ
  pens        назначено пенальти
  shots_n     число ударов

Пенальти определяются по подтипу 9 (все они имеют ровно xg = 0.79);
координатная эвристика оставлена как страховка для старых записей.
Автоголы (подтип 10) засчитываются В ПОЛЬЗУ СОПЕРНИКА — иначе
реконструкция счёта ломается, а вместе с ней и xg_level.
"""
import json, os, sys, collections
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PENALTY = 9
OWN_GOAL = 10
SP_SUBTYPES = {1, 2, 3, 8}
OPEN_STRICT = {0, 4, 7}
PEN_COORD = (50.0, 88.5)


def _t(s):
    if s is None:
        return None
    s = str(s).replace("'", '').strip()
    if '+' in s:
        s = s.split('+')[0]
    try:
        return float(s)
    except ValueError:
        return None


def metrics_from_game(d, diag=None):
    """
    Метрики одного матча из сохранённого chartEvents.
    d -- словарь вида {'id':..., 'chartEvents': {...}, 'events': [...]}.
    -> dict со строкой для shots.csv, либо None если ударов нет.
    """
    diag = diag if diag is not None else collections.Counter()
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
        try:
            xgot = float(sh.get('xgot') or 0)
        except (TypeError, ValueError):
            xgot = 0.0
        st = sh.get('subType')
        coord = (round(sh.get('line', -1), 1), round(sh.get('side', -1), 1))
        pen = (st == PENALTY) or (coord == PEN_COORD and 0.68 <= xg <= 0.88)
        norm.append(dict(cn=sh.get('competitorNum') or 1, xg=xg, xgot=xgot, st=st,
                         pen=pen, own=(st == OWN_GOAL),
                         t=_t(sh.get('time')) or 0.0,
                         goal=(sh.get('outcome') or {}).get('name') == 'Goal'))
        diag['pen_by_subtype'] += int(st == PENALTY)
        diag['own_goals'] += int(st == OWN_GOAL)
        diag['subtype_none'] += int(st is None)
    norm.sort(key=lambda s: s['t'])

    agg = {1: collections.defaultdict(float), 2: collections.defaultdict(float)}
    sc = {1: 0, 2: 0}
    for sh in norm:
        cn = sh['cn']
        state = sc[cn] - sc[3 - cn]
        a = agg[cn]
        a['xg_all'] += sh['xg']
        a['xgot_sum'] += sh['xgot']
        a['shots_n'] += 1
        if sh['pen']:
            a['pens'] += 1
        elif not sh['own']:
            a['npxg'] += sh['xg']
            if sh['st'] in SP_SUBTYPES:
                a['xg_sp'] += sh['xg']
            else:
                a['xg_open'] += sh['xg']
                if sh['st'] in OPEN_STRICT:
                    a['xg_openstr'] += sh['xg']
            if state == 0:
                a['xg_level'] += sh['xg']
        if sh['goal']:
            sc[3 - cn if sh['own'] else cn] += 1

    keys = ('xg_all', 'npxg', 'xg_open', 'xg_openstr', 'xg_sp',
            'xg_level', 'xgot_sum', 'pens', 'shots_n')
    return dict(game_id=d.get('id'), rec_h=sc[1], rec_a=sc[2],
                **{f'h_{k}': agg[1][k] for k in keys},
                **{f'a_{k}': agg[2][k] for k in keys})


def main(data_dir):
    """Источник ударов: россыпь json в data/raw/shots или сжатый архив."""
    import archive
    shots_dir = os.path.join(data_dir, 'raw', 'shots')
    rows, diag = [], collections.Counter()

    if os.path.isdir(shots_dir) and os.listdir(shots_dir):
        src = []
        for f in sorted(os.listdir(shots_dir)):
            try:
                src.append(json.load(open(os.path.join(shots_dir, f), encoding='utf-8')))
            except Exception:
                pass
        print(f'источник: россыпь файлов ({len(src)})')
    else:
        src = list(archive.load().values())
        print(f'источник: сжатый архив ({len(src)})')

    for d in src:
        r = metrics_from_game(d, diag)
        if r is not None:
            rows.append(r)

    S = pd.DataFrame(rows)
    out = os.path.join(data_dir, 'shots.csv')
    S.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'матчей с ударами: {len(S)} -> {out}')
    print('диагностика ударов:', dict(diag))

    m = pd.read_csv(os.path.join(data_dir, 'matches.csv'))
    dup = [c for c in S.columns if c != 'game_id' and c in m.columns]
    j = m.drop(columns=dup).merge(S, on='game_id', how='inner')
    ok = ((j.rec_h == j.hg) & (j.rec_a == j.ag)).mean()
    print(f'\nреконструкция счёта совпала с фактом: {ok:.1%} матчей '
          f'({int((1-ok)*len(j))} расхождений из {len(j)})')
    k = j.dropna(subset=['h_xg'])
    if len(k):
        print(f'\nсверка (N={len(k)}), за матч:')
        print(f'  голы                : {(k.hg + k.ag).mean():.3f}')
        print(f'  xG из статистики    : {(k.h_xg + k.a_xg).mean():.3f}')
        print(f'  xG из суммы ударов  : {(k.h_xg_all + k.a_xg_all).mean():.3f}')
        print(f'  NPxG                : {(k.h_npxg + k.a_npxg).mean():.3f}')
        print(f'  xG с игры (широкий) : {(k.h_xg_open + k.a_xg_open).mean():.3f}')
        print(f'  xG с игры (строгий) : {(k.h_xg_openstr + k.a_xg_openstr).mean():.3f}')
        print(f'  xG со стандартов    : {(k.h_xg_sp + k.a_xg_sp).mean():.3f}')
        print(f'  пенальти            : {(k.h_pens + k.a_pens).mean():.3f} '
              f'({100*(1-(k.h_npxg+k.a_npxg).sum()/(k.h_xg_all+k.a_xg_all).sum()):.1f}% всего xG)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data'))
