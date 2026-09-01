# -*- coding: utf-8 -*-
"""
Проверка контекстных факторов: даёт ли фактор информацию СВЕРХ силы команд
и преимущества поля?

Метод: берём остатки walk-forward прогноза (реальные голы минус ожидаемые
по модели) и регрессируем их на фактор. Если коэффициент значим — фактор
несёт информацию, которой в модели нет. Так проверяются:
  * время начала матча (жара: 17:45 против 20:15)
  * месяц (август-сентябрь -- пик жары в ОАЭ)
  * дни отдыха и разница в отдыхе между командами
  * номер тура (начало сезона -- «сыгранность»)
  * день недели
Дополнительно — тест на преимущество поля в голах против xG.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward
from predict import best_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)


def ols(y, X, names):
    """МНК с robust-стандартными ошибками (HC0)."""
    X = np.column_stack([np.ones(len(y)), X])
    names = ['const'] + list(names)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    S = (X * (r ** 2)[:, None]).T @ X
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = b / np.where(se > 0, se, np.nan)
    return pd.DataFrame(dict(фактор=names, коэф=b, ст_ошибка=se, t=t)).round(4)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    wf = walk_forward(df, best_params())
    d = wf.merge(df[['date', 'home', 'away', 'kickoff_hour', 'weekday', 'round',
                     'season', 'h_rest', 'a_rest', 'h_xg', 'a_xg']],
                 on=['date', 'home', 'away'], how='left', suffixes=('', '_y'))
    d = d.dropna(subset=['kickoff_hour'])
    print(f'матчей в анализе: {len(d)}')

    # остатки: сколько голов забито сверх ожидания модели
    d['res_total'] = (d.hg + d.ag) - (d.lh + d.la)
    d['res_diff'] = (d.hg - d.ag) - (d.lh - d.la)
    d['вечер'] = (d.kickoff_hour >= 19.5).astype(float)
    d['жаркий_месяц'] = pd.to_datetime(d.date).dt.month.isin([8, 9, 10, 5]).astype(float)
    d['rest_diff'] = (d.h_rest - d.a_rest).fillna(0).clip(-10, 10)
    d['h_rest_c'] = d.h_rest.fillna(d.h_rest.median()).clip(2, 20)
    d['начало_сезона'] = (d['round'] <= 4).astype(float)

    print('\n=== ВЛИЯНИЕ НА ОБЩУЮ РЕЗУЛЬТАТИВНОСТЬ (остаток «голы минус ожидание») ===')
    cols = ['вечер', 'жаркий_месяц', 'начало_сезона', 'h_rest_c']
    print(ols(d.res_total.values, d[cols].values, cols).to_string(index=False))

    print('\n=== ВЛИЯНИЕ НА ПЕРЕВЕС ХОЗЯЕВ (остаток разницы мячей) ===')
    cols2 = ['вечер', 'rest_diff', 'начало_сезона']
    print(ols(d.res_diff.values, d[cols2].values, cols2).to_string(index=False))

    print('\n=== ПРЕИМУЩЕСТВО ПОЛЯ: ГОЛЫ ПРОТИВ xG ===')
    p = df[df.played].dropna(subset=['h_xg', 'a_xg'])
    gd = (p.hg - p.ag).values
    xd = (p.h_xg - p.a_xg).values * 1.2139
    for nm, v in (('по голам', gd), ('по xG (в шкале голов)', xd)):
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f'  {nm:24s}: {v.mean():+.3f} ± {1.96*se:.3f} (95% ДИ)  N={len(v)}')
    diff = gd - xd
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f'  разница (голы - xG)     : {diff.mean():+.3f} ± {1.96*se:.3f}  '
          f'-> t = {diff.mean()/se:.2f}')

    print('\n=== ВРЕМЯ НАЧАЛА: СЫРЫЕ СРЕДНИЕ (без поправки на силу команд) ===')
    p2 = df[df.played].copy()
    p2['слот'] = np.where(p2.kickoff_hour < 19.0, 'ранний (17:00-18:30)', 'поздний (19:30+)')
    g = p2.groupby('слот').apply(lambda x: pd.Series({
        'матчей': len(x),
        'голов': (x.hg + x.ag).mean(),
        'xG': (x.h_xg + x.a_xg).mean() * 1.2139,
        'победы хозяев': (x.hg > x.ag).mean(),
        'ничьи': (x.hg == x.ag).mean()}), include_groups=False)
    print(g.round(3).to_string())
    a = p2[p2.kickoff_hour < 19.0]
    b = p2[p2.kickoff_hour >= 19.0]
    ta = (a.hg + a.ag).values
    tb = (b.hg + b.ag).values
    se = np.sqrt(ta.var(ddof=1) / len(ta) + tb.var(ddof=1) / len(tb))
    print(f'\n  разница в тотале ранний-поздний: {ta.mean()-tb.mean():+.3f} ± {1.96*se:.3f} '
          f'-> t = {(ta.mean()-tb.mean())/se:.2f}  (|t|>2 = значимо)')


if __name__ == '__main__':
    main()
