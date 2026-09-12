# -*- coding: utf-8 -*-
"""
Градиентный бустинг на признаках (мультиномиальный, своя реализация на numpy:
sklearn в окружении не установлен).

Идея. Вместо структурной модели счёта -- прямая классификация исхода [П1, X, П2]
по таблице признаков, известных ДО матча:
  * скользящие средние xG с игры (h_xg_open/a_xg_open) за и против для хозяев
    и гостей по окнам 5 и 10 матчей, их разности;
  * сила соперников: средний xG «за» соперников в последних 10 матчах команды;
  * открывающие цены Bet365 без маржи (степенной де-виг) и их разность;
  * отдых обеих команд и его разница.
Все признаки строки строятся ТОЛЬКО по матчам строго до неё: для строки train --
по предыдущим строкам train, для test -- по всему train. Обучение, подбор числа
итераций, импутация пропусков -- всё внутри predict и только на train.

Стартовая точка бустинга (offset) -- логарифмы рыночных вероятностей открытия.
Причина: на 150-380 матчах деревья не способны с нуля восстановить то, что
уже заложено в котировке (Pitcan 2026: вес структурной модели в пуле с ценой
около нуля; Wunderlich & Memmert 2018: рейтинг на котировках бьёт рейтинг на
голах). Поэтому бустинг учит ПОПРАВКИ к рынку, а при нуле итераций модель
совпадает с рынком. Там, где открытия нет (12 матчей), offset -- частоты
исходов в train.

Бустинг -- ньютоновский, в духе XGBoost: g = p_k - y_k, h = p_k(1 - p_k),
значение листа -sum(g)/(sum(h) + L2), прирост от расщепления через суммы
G, H. Регуляризация фиксирована из теории, не подбирается: глубина 3,
learning_rate 0.05, L2 = 5 на значения листьев, минимум 15 матчей в листе,
веса матчей с полураспадом 180 дней (лучшее известное для этой лиги).
Число итераций (0..200) выбирается по forward-chaining валидации ВНУТРИ train:
первые 70% по времени -- подгонка, последние 30% -- проверка, берётся argmin
сглаженной кривой log-loss; затем модель переобучается на всём train с этим
числом итераций.

Слабое место: выборка. 264 тестовых матча и 120-380 обучающих -- деревьям
попросту негде выучить что-то, чего нет в цене; ожидаемый итог -- в лучшем
случае вровень с открытием, и любые «улучшения» на уровне 0.005 log-loss
тонут в SE ~0.013.
"""
import numpy as np
import pandas as pd

from markets import devig_power

# --- фиксированные гиперпараметры (см. докстринг: подбора по выборке нет)
HALF_LIFE = 180.0      # полураспад веса обучающего матча, дней
LR = 0.05              # learning rate
MAX_ITER = 200         # верхняя граница числа деревьев на класс
DEPTH = 3              # глубина дерева
L2 = 5.0               # L2 на значения листьев
MIN_LEAF = 15          # минимум матчей в листе
VAL_FRAC = 0.30        # доля train (по времени) под валидацию числа итераций
SMOOTH = 9             # окно сглаживания кривой валидации, итераций
PATIENCE = 40          # валидационный прогон останавливается, если минимум не обновлялся столько итераций
DIAG = []              # (n_train, n_iter) по дням -- только диагностика, на прогноз не влияет
XG_H, XG_A = 'h_xg_open', 'a_xg_open'
REST_CAP = 30.0        # отдых больше месяца -- это пауза, а не отдых


# ---------------------------------------------------------------------------
#                                 ПРИЗНАКИ
# ---------------------------------------------------------------------------
class _History:
    """Матчи train в разрезе команд: для каждой команды отсортированные по ts
    массивы (ts, xG за, xG против, соперник). Запрос «до момента ts» -- через
    searchsorted, поэтому утечки будущего нет по построению."""

    def __init__(self, train):
        d = train.dropna(subset=[XG_H, XG_A])
        rows = []
        for r in d.itertuples(index=False):
            rows.append((r.home, r.ts, r._asdict()[XG_H], r._asdict()[XG_A], r.away))
            rows.append((r.away, r.ts, r._asdict()[XG_A], r._asdict()[XG_H], r.home))
        self.by_team = {}
        for team, ts, xf, xa, opp in rows:
            self.by_team.setdefault(team, []).append((ts, xf, xa, opp))
        for team, lst in self.by_team.items():
            lst.sort(key=lambda z: z[0])
            self.by_team[team] = (np.array([z[0] for z in lst], float),
                                  np.array([z[1] for z in lst], float),
                                  np.array([z[2] for z in lst], float),
                                  [z[3] for z in lst])

    def before(self, team, ts):
        """Всё, что команда сыграла строго до ts."""
        if team not in self.by_team:
            return None
        t, xf, xa, opp = self.by_team[team]
        k = int(np.searchsorted(t, ts, side='left'))
        if k == 0:
            return None
        return xf[:k], xa[:k], opp[:k]

    def avg_for(self, team, ts):
        """Средний xG «за» команды по всем матчам до ts (сила соперника)."""
        b = self.before(team, ts)
        return np.nan if b is None else float(b[0].mean())

    def team_feats(self, team, ts):
        b = self.before(team, ts)
        if b is None:
            return [np.nan] * 5
        xf, xa, opp = b
        f5, a5 = xf[-5:].mean(), xa[-5:].mean()
        f10, a10 = xf[-10:].mean(), xa[-10:].mean()
        o = [self.avg_for(p, ts) for p in opp[-10:]]
        o = [v for v in o if not np.isnan(v)]
        opp10 = float(np.mean(o)) if o else np.nan
        return [f5, a5, f10, a10, opp10]


FEATURES = ['h_for5', 'h_ag5', 'h_for10', 'h_ag10', 'h_opp10',
            'a_for5', 'a_ag5', 'a_for10', 'a_ag10', 'a_opp10',
            'net5_diff', 'net10_diff', 'for10_diff', 'ag10_diff',
            'mkt_H', 'mkt_D', 'mkt_A', 'mkt_HA',
            'h_rest', 'a_rest', 'rest_diff']


def _market(row):
    o = [row.get('open_H'), row.get('open_D'), row.get('open_A')]
    if any(pd.isna(x) for x in o) or any(x <= 1.0 for x in o):
        return None
    q = np.asarray(devig_power(o), float)
    if np.any(np.isnan(q)) or q.sum() <= 0:
        return None
    return q / q.sum()


def _features(hist, rows):
    """Матрица признаков и рыночные вероятности (NaN, если открытия нет)."""
    X, M = [], []
    for _, r in rows.iterrows():
        h = hist.team_feats(r.home, r.ts)
        a = hist.team_feats(r.away, r.ts)
        q = _market(r)
        if q is None:
            q = np.array([np.nan] * 3)
        hr = min(float(r.h_rest), REST_CAP) if pd.notna(r.h_rest) else np.nan
        ar = min(float(r.a_rest), REST_CAP) if pd.notna(r.a_rest) else np.nan
        X.append(h + a + [
            (h[0] - h[1]) - (a[0] - a[1]),          # net5_diff
            (h[2] - h[3]) - (a[2] - a[3]),          # net10_diff
            h[2] - a[2],                            # for10_diff
            h[3] - a[3],                            # ag10_diff
            q[0], q[1], q[2], q[0] - q[2],
            hr, ar, hr - ar,
        ])
        M.append(q)
    return np.array(X, float), np.array(M, float)


# ---------------------------------------------------------------------------
#                        ДЕРЕВО РЕГРЕССИИ (ньютоновское)
# ---------------------------------------------------------------------------
def _leaf(G, H):
    return -G / (H + L2)


def _gain(G, H):
    return G * G / (H + L2)


def _build(X, g, h, order, idx, depth):
    """Рекурсивно строит дерево. order -- предсортированные индексы по каждому
    признаку (по всему X), idx -- булева маска строк узла. Возвращает либо
    ('leaf', value), либо ('split', j, thr, left, right)."""
    G, H = g[idx].sum(), h[idx].sum()
    n = int(idx.sum())
    if depth == 0 or n < 2 * MIN_LEAF:
        return ('leaf', _leaf(G, H))
    base = _gain(G, H)
    best = (0.0, None, None)
    for j in range(X.shape[1]):
        o = order[:, j]
        o = o[idx[o]]                                # строки узла в порядке признака j
        xs = X[o, j]
        cg, ch = np.cumsum(g[o]), np.cumsum(h[o])
        pos = np.arange(1, n)                        # левая часть = первые pos строк
        ok = (pos >= MIN_LEAF) & (n - pos >= MIN_LEAF) & (xs[:-1] != xs[1:])
        if not ok.any():
            continue
        GL, HL = cg[:-1][ok], ch[:-1][ok]
        gain = _gain(GL, HL) + _gain(G - GL, H - HL) - base
        k = int(np.argmax(gain))
        if gain[k] > best[0]:
            p = pos[ok][k]
            best = (float(gain[k]), j, 0.5 * (xs[p - 1] + xs[p]))
    if best[1] is None or best[0] <= 1e-12:
        return ('leaf', _leaf(G, H))
    j, thr = best[1], best[2]
    left = idx & (X[:, j] <= thr)
    right = idx & ~(X[:, j] <= thr)
    return ('split', j, thr,
            _build(X, g, h, order, left, depth - 1),
            _build(X, g, h, order, right, depth - 1))


def _apply(tree, X):
    out = np.zeros(len(X))
    stack = [(tree, np.ones(len(X), bool))]
    while stack:
        node, m = stack.pop()
        if node[0] == 'leaf':
            out[m] = node[1]
        else:
            _, j, thr, l, r = node
            lm = m & (X[:, j] <= thr)
            stack.append((l, lm))
            stack.append((r, m & ~lm))
    return out


def _softmax(F):
    Z = F - F.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())


def _boost(X, y, w, F0, n_iter, X_val=None, y_val=None, F0_val=None):
    """Ньютоновский мультиномиальный бустинг. Возвращает список деревьев
    (по n_iter троек) и, если задана валидация, кривую log-loss по итерациям
    (curve[0] -- при нуле деревьев)."""
    n, K = len(y), 3
    Y = np.eye(K)[y]
    order = np.argsort(X, axis=0, kind='stable')
    F = F0.copy()
    trees, curve = [], []
    Fv = None if X_val is None else F0_val.copy()
    if Fv is not None:
        curve.append(_logloss(_softmax(Fv), y_val))
    for _ in range(n_iter):
        P = _softmax(F)
        step = []
        for k in range(K):
            g = w * (P[:, k] - Y[:, k])
            h = w * np.maximum(P[:, k] * (1 - P[:, k]), 1e-6)
            t = _build(X, g, h, order, np.ones(n, bool), DEPTH)
            step.append(t)
            F[:, k] += LR * _apply(t, X)
            if Fv is not None:
                Fv[:, k] += LR * _apply(t, X_val)
        trees.append(step)
        if Fv is not None:
            curve.append(_logloss(_softmax(Fv), y_val))
            if len(curve) - 1 - int(np.argmin(curve)) >= PATIENCE:
                break
    return trees, curve


def _predict_trees(trees, X, F0):
    F = F0.copy()
    for step in trees:
        for k, t in enumerate(step):
            F[:, k] += LR * _apply(t, X)
    return _softmax(F)


# ---------------------------------------------------------------------------
#                                 МОДЕЛЬ
# ---------------------------------------------------------------------------
def _offset(M, prior):
    """Стартовые логиты: рынок, где он есть, иначе частоты исходов train."""
    F = np.log(np.clip(np.where(np.isnan(M), prior, M), 1e-6, 1))
    return F


def _impute(X, means):
    X = X.copy()
    bad = np.isnan(X)
    X[bad] = np.take(means, np.where(bad)[1])
    return X


def predict(train, test):
    train = train[train.played & train.hg.notna()].sort_values('ts').reset_index(drop=True)
    hist = _History(train)

    # признаки train строятся по матчам строго до каждой строки (searchsorted по ts)
    Xtr, Mtr = _features(hist, train)
    y = np.where(train.hg > train.ag, 0, np.where(train.hg == train.ag, 1, 2)).astype(int)
    prior = np.bincount(y, minlength=3) / len(y)

    ref = float(test.ts.min())
    w = np.exp(-np.log(2) / HALF_LIFE * np.maximum((ref - train.ts.values) / 86400.0, 0.0))
    w = w / w.mean()

    # импутация -- по средним train (первые матчи сезона без истории и матчи без открытия)
    means = np.nanmean(Xtr, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    Xtr = _impute(Xtr, means)
    F0 = _offset(Mtr, prior)

    # --- число итераций: forward-chaining внутри train
    n = len(train)
    cut = int(round(n * (1 - VAL_FRAC)))
    n_iter = 0
    if n - cut >= 30 and cut >= 60:
        _, curve = _boost(Xtr[:cut], y[:cut], w[:cut], F0[:cut], MAX_ITER,
                          Xtr[cut:], y[cut:], F0[cut:])
        c = np.asarray(curve)
        # сглаживание: скользящее среднее, чтобы argmin не ловил случайный зубец
        pad = SMOOTH // 2
        cs = np.convolve(np.pad(c, pad, mode='edge'), np.ones(SMOOTH) / SMOOTH, mode='valid')
        n_iter = int(np.argmin(cs))
        if cs[n_iter] >= cs[0]:
            n_iter = 0                       # поправки к рынку не помогают -- остаёмся рынком
    DIAG.append((n, n_iter))

    trees, _ = _boost(Xtr, y, w, F0, n_iter)

    # --- test: признаки по всему train
    Xte, Mte = _features(hist, test)
    Xte = _impute(Xte, means)
    P = _predict_trees(trees, Xte, _offset(Mte, prior))
    return np.clip(P, 1e-6, 1.0)
