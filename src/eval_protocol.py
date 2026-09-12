# -*- coding: utf-8 -*-
"""
ЕДИНЫЙ ПРОТОКОЛ ОЦЕНКИ ПРОГНОЗНЫХ МОДЕЛЕЙ.

Одна линейка для всех. Любая модель -- это функция

    predict(train: DataFrame, test: DataFrame) -> ndarray (n_test, 3)

где строки -- вероятности исходов [П1, X, П2], сумма = 1. Протокол сам
режет данные по дням (walk-forward), сам прячет от модели всё, чего до
матча знать нельзя, и сам считает метрики против одного и того же рынка.

ЧТО ВИДИТ МОДЕЛЬ.
  train -- сыгранные матчи строго ДО дня прогноза, со всеми колонками
           (результаты, xG, удары, открытие и закрытие Bet365).
  test  -- матчи одного игрового дня БЕЗ результатов, БЕЗ статистики матча,
           БЕЗ закрывающих цен. Есть: команды, дата, время, тур, отдых,
           ОТКРЫВАЮЩИЕ цены Bet365 (open_H/open_D/open_A) -- они известны
           за сутки-двое до матча и модели разрешены как признак.

ПРОТИВ ЧЕГО МЕРЯЕМ. Два рыночных ориентира, оба -- Bet365 без маржи
(степенной де-виг; по калибровке на этой лиге он лучший, log-loss 0.91675):
  * open  -- открывающая цена: то, что модель ВПРАВЕ знать. Честная цель.
  * close -- закрывающая: известна только к свистку. Обыграть её, зная
             только открытие и xG, -- это уже настоящий результат.

МЕТРИКИ. Log-loss основная (Wheatcroft 2021: логарифмическая оценка
различает прогнозы лучше RPS и Brier), RPS для сравнимости с литературой,
парные разницы с ошибкой -- потому что на 300 матчах разница в третьем
знаке без SE ничего не значит. Плюс калибровка по бакетам фаворита:
на этой лиге все методы недооценивают сильных фаворитов, это надо видеть.

    from eval_protocol import evaluate
    rep = evaluate(my_predict)          # dict с метриками
    print(format_report(rep))

    python src/eval_protocol.py         # прогнать базовую модель проекта
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from markets import DEVIG                                  # noqa: E402

MATCHES = os.path.join(ROOT, 'data', 'matches.csv')
START = '2025-02-01'          # начало окна оценки
MIN_TRAIN = 120               # минимум сыгранных матчей в обучении
DEVIG_METHOD = 'power'

# Колонки, которых в test быть НЕ должно: результат, статистика матча,
# закрывающие цены. Всё остальное модель может использовать.
_HIDE_PREFIX = ('h_', 'a_')
_HIDE_EXACT = {'hg', 'ag', 'odds_H', 'odds_D', 'odds_A', 'status', 'book'}
_KEEP_EVEN_IF_PREFIXED = {'h_rest', 'a_rest'}      # отдых известен заранее


def load():
    m = pd.read_csv(MATCHES)
    m = m[m.played.fillna(False) & m.hg.notna()].sort_values('ts').reset_index(drop=True)
    return m


def _hide(test):
    cols = [c for c in test.columns
            if not ((c in _HIDE_EXACT) or
                    (c.startswith(_HIDE_PREFIX) and c not in _KEEP_EVEN_IF_PREFIXED))]
    return test[cols].copy()


def outcome(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def rps(p, o):
    """Ranked probability score для одного матча; p -- (3,), o -- индекс исхода."""
    c = np.cumsum(p)
    e = np.cumsum(np.eye(3)[o])
    return float(np.sum((c - e) ** 2) / 2)


def _market(row, kind):
    ks = ('open_H', 'open_D', 'open_A') if kind == 'open' else ('odds_H', 'odds_D', 'odds_A')
    o = [row[k] for k in ks]
    if any(pd.isna(x) for x in o):
        return None
    q = DEVIG[DEVIG_METHOD](o)
    if any(np.isnan(q)):
        return None
    return np.array(q, float)


def evaluate(predict, start=START, min_train=MIN_TRAIN, verbose=False):
    """
    -> dict: n, logloss, rps, ll_open, rps_open, ll_close, rps_close,
             d_ll_open (модель минус открытие, парная), se_..., t_...,
             calib (таблица по бакетам фаворита), rows (DataFrame по матчам).
    """
    df = load()
    days = sorted(df.loc[df.date >= start, 'date'].unique())
    rows = []
    for day in days:
        train = df[df.date < day]
        if len(train) < min_train:
            continue
        test = df[df.date == day]
        try:
            P = np.asarray(predict(train.copy(), _hide(test)), float)
        except Exception as e:
            if verbose:
                print(f'  {day}: модель упала: {type(e).__name__}: {e}', file=sys.stderr)
            continue
        if P.shape != (len(test), 3):
            raise ValueError(f'{day}: ожидалась форма {(len(test), 3)}, получена {P.shape}')
        P = np.clip(P, 1e-6, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        outs = outcome(test.hg.values, test.ag.values)
        for (_, r), p, o in zip(test.iterrows(), P, outs):
            mo, mc = _market(r, 'open'), _market(r, 'close')
            rows.append(dict(
                date=r.date, home=r.home, away=r.away, out=int(o),
                pH=p[0], pD=p[1], pA=p[2],
                ll=-np.log(p[o]), rps=rps(p, o),
                ll_open=(-np.log(mo[o]) if mo is not None else np.nan),
                rps_open=(rps(mo, o) if mo is not None else np.nan),
                ll_close=(-np.log(mc[o]) if mc is not None else np.nan),
                rps_close=(rps(mc, o) if mc is not None else np.nan),
                fav_p=float(p.max()), fav_win=int(int(np.argmax(p)) == o),
                open_fav=(float(mo.max()) if mo is not None else np.nan),
            ))
    R = pd.DataFrame(rows)
    if R.empty:
        raise RuntimeError('ни одного прогноза')

    def paired(a, b):
        d = (R[a] - R[b]).dropna()
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        return float(d.mean()), float(se), (float(d.mean() / se) if se and se > 0 else np.nan), int(len(d))

    rep = dict(n=int(len(R)), logloss=float(R.ll.mean()), rps=float(R.rps.mean()))
    for k in ('open', 'close'):
        sub = R.dropna(subset=[f'll_{k}'])
        rep[f'll_{k}'] = float(sub[f'll_{k}'].mean())
        rep[f'rps_{k}'] = float(sub[f'rps_{k}'].mean())
        rep[f'n_{k}'] = int(len(sub))
        m, se, t, n = paired('ll', f'll_{k}')
        rep[f'd_ll_{k}'], rep[f'se_ll_{k}'], rep[f't_ll_{k}'] = m, se, t
        m, se, t, n = paired('rps', f'rps_{k}')
        rep[f'd_rps_{k}'], rep[f'se_rps_{k}'], rep[f't_rps_{k}'] = m, se, t

    # калибровка по вероятности фаворита у МОДЕЛИ
    bins = [0.0, 0.45, 0.55, 0.65, 0.75, 1.01]
    R['_b'] = pd.cut(R.fav_p, bins)
    cal = R.groupby('_b', observed=True).agg(n=('fav_win', 'size'),
                                             pred=('fav_p', 'mean'),
                                             fact=('fav_win', 'mean'))
    rep['calib'] = cal
    rep['rows'] = R.drop(columns=['_b'])
    return rep


def format_report(rep, name='модель'):
    L = [f'{name}: n={rep["n"]}  log-loss {rep["logloss"]:.5f}  RPS {rep["rps"]:.5f}']
    for k, lab in (('open', 'открытие Bet365'), ('close', 'закрытие Bet365')):
        L.append(f'  против {lab:<16} (n={rep[f"n_{k}"]}): log-loss {rep[f"ll_{k}"]:.5f}  '
                 f'RPS {rep[f"rps_{k}"]:.5f}  |  Δlog-loss {rep[f"d_ll_{k}"]:+.5f} '
                 f'± {rep[f"se_ll_{k}"]:.5f} (t={rep[f"t_ll_{k}"]:+.2f})  '
                 f'ΔRPS {rep[f"d_rps_{k}"]:+.5f} (t={rep[f"t_rps_{k}"]:+.2f})')
    L.append('  калибровка фаворита (модель):')
    for b, r in rep['calib'].iterrows():
        L.append(f'    {str(b):<14} n={int(r.n):3d}  предсказано {100*r.pred:5.1f}%  факт {100*r.fact:5.1f}%')
    return '\n'.join(L)


# ---------------------------------------------------------------------------
#                        БАЗОВЫЕ МОДЕЛИ ДЛЯ СРАВНЕНИЯ
# ---------------------------------------------------------------------------
def predict_market_open(train, test):
    """Нулевая модель: открывающая цена Bet365 без маржи. Её надо обыграть."""
    out = []
    for _, r in test.iterrows():
        q = _market(r, 'open')
        out.append(q if q is not None else np.array([1 / 3, 1 / 3, 1 / 3]))
    return np.vstack(out)


def predict_dixon_coles(train, test):
    """Действующая модель проекта: Диксон-Коулз на xG с игры."""
    from model import DixonColes, detect_newcomers
    from markets import wdl
    from predict import best_params
    nc = detect_newcomers(pd.concat([train, test]))
    new_here = set()
    for s in test.season.unique():
        new_here |= nc.get(s, set())
    m = DixonColes(**best_params()).fit(train, ref_ts=float(test.ts.min()), newcomers=new_here)
    out = []
    for _, r in test.iterrows():
        if r.home in m.idx and r.away in m.idx:
            M, _, _ = m.matrix(r.home, r.away)
            p = wdl(M)
            out.append([p['H'], p['D'], p['A']])
        else:
            out.append([1 / 3, 1 / 3, 1 / 3])
    return np.array(out, float)


if __name__ == '__main__':
    for name, fn in (('рынок (открытие)', predict_market_open),
                     ('Диксон-Коулз на xG', predict_dixon_coles)):
        rep = evaluate(fn, verbose=True)
        print(format_report(rep, name))
        print()
