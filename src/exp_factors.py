# -*- coding: utf-8 -*-
"""
ЧЕСТНЫЙ ИЗМЕРИТЕЛЬ ФАКТОРОВ.

Гипотезы из литературы проверяются не корреляцией на всей выборке (там любой
фактор что-нибудь да покажет), а скользящим прогнозом вперёд: модель учится
только на прошлом, предсказывает следующий игровой день, и мы сравниваем
её ошибку с ошибкой той же модели без фактора.

Сравнение ПОПАРНОЕ, матч к матчу. Это принципиально: RPS двух моделей на одной
выборке коррелирует под 0.99, поэтому разность оценивается на порядок точнее,
чем каждая величина по отдельности. Непарное сравнение здесь не увидело бы
вообще ничего.

    python src/exp_factors.py                 # весь набор гипотез
    python src/exp_factors.py heat ramadan    # только названные
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from backtest import walk_forward
from predict import best_params

FACTORS = os.path.join(ROOT, 'data', 'factors.csv')
MATCHES = os.path.join(ROOT, 'data', 'matches.csv')
OUT = os.path.join(ROOT, 'data', 'exp_factors.csv')

# Гипотезы. sym -- признак матча (двигает тотал), team -- пара (хозяева, гости)
# с общим коэффициентом (двигает баланс).
HYPOTHESES = {
    'heat':        dict(sym=['heat'],                 why='жара давит темп / повышает ошибки'),
    'temp':        dict(sym=['temp'],                 why='температура воздуха без влажности'),
    'evening':     dict(sym=['evening'],              why='вечерние матчи прохладнее'),
    'ramadan':     dict(sym=['ramadan'],              why='пост снижает работоспособность'),
    'rest':        dict(team=[('rest_h', 'rest_a')],  why='отдых перед матчем'),
    'congestion':  dict(team=[('cong_h', 'cong_a')],  why='матчей за 21 день'),
    'travel':      dict(team=[('travel_h', 'travel_a')], why='переезд гостей'),
    'heat_rest':   dict(sym=['heat'], team=[('rest_h', 'rest_a')], why='жара + отдых вместе'),
}


def load():
    m = pd.read_csv(MATCHES)
    f = pd.read_csv(FACTORS)
    keep = [c for c in f.columns if c not in m.columns or c == 'game_id']
    d = m.merge(f[keep], on='game_id', how='left')
    d['travel_h'] = 0.0                      # хозяева никуда не едут
    d['travel_a'] = d['travel_a'].fillna(0.0)
    for c in ('heat', 'temp', 'rh'):
        if c in d:
            d[c] = d[c] / 10.0               # шкала: коэффициент на +10 °C
    for c in ('travel_h', 'travel_a'):
        d[c] = d[c] / 100.0                  # на +100 км
    for c in ('rest_h', 'rest_a', 'cong_h', 'cong_a', 'heat', 'temp',
              'evening', 'ramadan'):
        if c in d:
            d[c] = d[c].fillna(d[c].median())
    return d


def paired(a, b):
    """
    Разность ошибок на ОБЩИХ матчах. Возвращает (среднее, парный SE, t, n).
    Ключ -- дата+команды: строки в двух прогонах идут в одном порядке,
    но полагаться на это нельзя.
    """
    k = ['date', 'home', 'away']
    j = a[k + ['rps', 'll']].merge(b[k + ['rps', 'll']], on=k, suffixes=('_a', '_b'))
    out = {}
    for met in ('rps', 'll'):
        d = j[f'{met}_b'] - j[f'{met}_a']          # <0 значит b лучше
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        out[met] = (float(d.mean()), float(se),
                    float(d.mean() / se) if se and se > 0 else np.nan)
    return out, len(j)


def run(names=None):
    d = load()
    base_p = best_params()
    print('база: скользящий прогноз без факторов')
    t0 = time.time()
    base = walk_forward(d, base_p)
    print('  матчей в оценке: %d, RPS %.5f, log-loss %.5f  (%.0f c)'
          % (len(base), base.rps.mean(), base.ll.mean(), time.time() - t0))
    if 'rps_book' in base:
        bb = base.dropna(subset=['rps_book'])
        print('  букмекер на тех же: RPS %.5f, log-loss %.5f' % (bb.rps_book.mean(), bb.ll_book.mean()))
    print()

    rows = []
    todo = names or list(HYPOTHESES)
    for name in todo:
        h = HYPOTHESES.get(name)
        if not h:
            print(f'нет такой гипотезы: {name}'); continue
        p = dict(base_p)
        p['covariates'] = tuple(h.get('sym', ()))
        p['covariates_team'] = tuple(h.get('team', ()))
        t0 = time.time()
        try:
            res = walk_forward(d, p)
        except Exception as e:
            print(f'{name}: упало: {e}'); continue
        st, n = paired(base, res)
        dr, sr, tr = st['rps']
        dl, sl, tl = st['ll']
        rows.append(dict(name=name, why=h['why'], n=n,
                         rps=res.rps.mean(), d_rps=dr, se_rps=sr, t_rps=tr,
                         ll=res.ll.mean(), d_ll=dl, se_ll=sl, t_ll=tl,
                         secs=time.time() - t0))
        mark = 'ЛУЧШЕ' if tr < -2 else ('хуже' if tr > 2 else 'без разницы')
        print('%-12s RPS %.5f  Δ %+.5f ± %.5f  t=%+5.2f   %s'
              % (name, res.rps.mean(), dr, 1.96 * sr, tr, mark))

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(OUT, index=False)
        print()
        print('Δ<0 -- фактор УМЕНЬШАЕТ ошибку. Порог значимости |t|>2.')
        print('сохранено:', OUT)
    return out


if __name__ == '__main__':
    run(sys.argv[1:] or None)
