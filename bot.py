"""
Бот: обновляет данные, считает ставки, шлёт в Telegram.

    python bot.py                  # посчитать ставки и прислать
    python bot.py --mode clv       # зафиксировать закрывающую линию Pinnacle
    python bot.py --mode settle    # рассчитать сыгравшее и прислать сводку
    python bot.py --force          # прислать даже то, что уже отправляли
    python bot.py --dry            # ничего не слать, только напечатать

Что делает:
  1. src/update_data.py  — догружает новые матчи и поударные данные
  2. src/value_report.py — переобучает модель и считает ставки в data/picks.csv
  3. сверяется со state.json, чтобы не слать одно и то же дважды
  4. форматирует и отправляет

Ставка пересылается повторно, только если её матожидание выросло минимум
на 3 процентных пункта — то есть линия реально сдвинулась в нашу сторону.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import datetime as dt

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
STATE = os.path.join(ROOT, 'state.json')
PICKS = os.path.join(ROOT, 'data', 'picks.csv')
FORECAST = os.path.join(ROOT, 'data', 'forecast.csv')

from telegram_sender import send_message   # noqa: E402

RESEND_IF_EV_GAIN = 0.03      # переслать, если EV вырос на 3 п.п.
TIER_NAME = {
    '1': 'ПОДТВЕРЖДЕНО ОСТРОЙ ЛИНИЕЙ',
    '2': 'РЫНОК НЕ ВОЗРАЖАЕТ',
    '3': 'РЫНОК ВОЗРАЖАЕТ',
    '?': 'ЭТАЛОНА НЕТ',
}


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(SRC, script), *args],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=ROOT)
    print(f'--- {script} ---')
    print((r.stdout or '')[-2500:])
    if r.returncode != 0:
        print((r.stderr or '')[-2500:], file=sys.stderr)
    return r.returncode == 0


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding='utf-8'))
        except Exception:
            pass
    return {'sent': {}, 'last_run': None, 'runs': 0}


def save_state(st):
    st['last_run'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    st['runs'] = st.get('runs', 0) + 1
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def key_of(r):
    line = '' if pd.isna(r.get('линия')) else str(r.get('линия'))
    return f"{r['матч']}|{r['рынок']}|{line}|{r['исход']}"


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def format_pick(r):
    line = '' if pd.isna(r.get('линия')) else f" {r['линия']}"
    pin = ('эталона нет' if pd.isna(r.get('ev_pin'))
           else f"рынок {100*r['ev_pin']:+.1f}%")
    tot = ' · тотал' if r.get('тотальный') else ''
    return (f"<b>{esc(r['матч'])}</b>\n"
            f"{esc(r['рынок'])}{esc(line)} <b>{esc(r['исход'])}</b> @ <b>{r['кэф']:.2f}</b>\n"
            f"модель {100*r['p']:.1f}% · справедливый {r['fair']:.2f} · "
            f"перевес {100*r['edge_pp']:+.1f} п.п.\n"
            f"EV {100*r['ev']:+.1f}% · {pin}{tot}")


def _pct(x):
    return '—' if x is None or (isinstance(x, float) and x != x) else f'{100*float(x):.0f}%'


def forecast_key(fc):
    """Ключ тура: набор матчей. Меняется -- значит, появились новые матчи."""
    return '|'.join(sorted(f"{r['дата']}:{r['home']}-{r['away']}" for _, r in fc.iterrows()))


def build_forecast(fc):
    """
    ДАЙДЖЕСТ ПРОГНОЗОВ -- то, ради чего бот вообще существует.

    Отправляется каждый тур независимо от того, нашлась ли ставка. Раньше бот
    молчал, пока модель не найдёт перевес над рынком, а перевеса она не
    находит (на 717 исходах она хуже рынка в каждом бакете) -- и пользователь
    не видел ни одного прогноза. Прогноз и ставка -- разные вещи: прогноз
    показывается всегда, ставка -- только когда есть подтверждение ценой.

    Честность встроена в текст: модель по log-loss уступает открывающей линии
    Bet365 (0.962 против 0.947 на 264 матчах), поэтому там, где модель
    расходится с рынком, чаще прав рынок. Расхождение -- повод присмотреться,
    а не ставить.
    """
    if fc is None or fc.empty:
        return None
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)
    parts = [f"📊 <b>Про-Лига ОАЭ — прогноз на тур</b>\n"
             f"<i>{now:%d.%m %H:%M} по Дубаю · xG-модель Диксона-Коулза</i>"]
    fc = fc.sort_values(['дата', 'время'])
    MK = {'1': 'рынок1', 'X': 'рынокX', '2': 'рынок2'}
    for _, r in fc.iterrows():
        fav_m = max(('1', r['p1']), ('X', r['pX']), ('2', r['p2']), key=lambda t: t[1])
        has_mkt = r.get('рынок1') is not None and r['рынок1'] == r['рынок1']
        flag = ''
        if has_mkt:
            fav_k = max(('1', r['рынок1']), ('X', r['рынокX']), ('2', r['рынок2']), key=lambda t: t[1])[0]
            if fav_k != fav_m[0]:
                flag = '  ⚠️ <i>рынок за другой исход</i>'
            elif abs(float(fav_m[1]) - float(r[MK[fav_m[0]]])) >= 0.10:
                flag = '  ⚠️ <i>расхождение с рынком ≥10 п.п.</i>'
        line = (f"\n⚽ <b>{esc(r['хозяева'])} — {esc(r['гости'])}</b>  "
                f"{str(r['дата'])[5:].replace('-', '.')} {r['время']}\n"
                f"   модель:  1 {_pct(r['p1'])} · X {_pct(r['pX'])} · 2 {_pct(r['p2'])}"
                f"  ·  счёт {r['ож_голы_х']:.1f}:{r['ож_голы_г']:.1f}\n"
                f"   рынок:   1 {_pct(r.get('рынок1'))} · X {_pct(r.get('рынокX'))} · 2 {_pct(r.get('рынок2'))}")
        if has_mkt:
            line += f"  ·  кэфы {r['кэф1']:.2f}/{r['кэфX']:.2f}/{r['кэф2']:.2f}"
        line += (f"\n   xG за матч: хоз. {r['атака_х']:.2f}/{r['оборона_х']:.2f}, "
                 f"гости {r['атака_г']:.2f}/{r['оборона_г']:.2f}" + flag)
        parts.append(line)
    parts.append("\n<i>Это прогноз, не ставка. По точности модель уступает открывающей линии "
                 "(log-loss 0.962 против 0.947): где они расходятся, чаще прав рынок. "
                 "Ставка приходит отдельным сообщением только при подтверждении ценой.</i>")
    return '\n'.join(parts)


def build_message(picks, state, force=False):
    if picks.empty:
        return None, []
    sent = state.get('sent', {})
    fresh = []
    for _, r in picks.iterrows():
        k = key_of(r)
        prev = sent.get(k)
        if not force and prev is not None and r['ev'] <= prev.get('ev', -9) + RESEND_IF_EV_GAIN:
            continue
        fresh.append(r)
    if not fresh:
        return None, []

    F = pd.DataFrame(fresh)
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)   # время Дубая
    parts = [f"⚽ <b>Про-Лига ОАЭ</b> — ставки на тур\n<i>{now:%d.%m %H:%M} по Дубаю</i>"]

    playable = F[F['ярус'].isin(['1', '2'])]
    for tier in ('1', '2', '3', '?'):
        sub = F[F['ярус'] == tier].sort_values('ev', ascending=False)
        if sub.empty:
            continue
        mark = {'1': '✅', '2': '🟡', '3': '🔴', '?': '⚪'}[tier]
        parts.append(f"\n{mark} <b>{TIER_NAME[tier]}</b>")
        if tier == '3':
            parts.append('<i>такой отбор на 239 матчах дал ROI −18.6%. Не рекомендуется.</i>')
        for _, r in sub.iterrows():
            parts.append('\n' + format_pick(r))

    if not playable.empty:
        tot = float(playable['Келли_%'].sum())
        scale = min(1.0, 6.0 / tot) if tot > 0 else 1.0
        parts.append(f"\n💰 <b>Размер ставок</b> (четверть-Келли, потолок 6% банка)")
        for _, r in playable.iterrows():
            parts.append(f"· {esc(r['исход'])} @ {r['кэф']:.2f} — "
                         f"<b>{r['Келли_%']*scale:.2f}%</b> банка")
    else:
        parts.append('\n<i>Ставить по Келли нечего: ярусы 1-2 пусты.</i>')

    return '\n'.join(parts), fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='слать даже уже отправленное')
    ap.add_argument('--dry', action='store_true', help='не отправлять, только напечатать')
    ap.add_argument('--days-back', default='60')
    ap.add_argument('--skip-update', action='store_true')
    ap.add_argument('--mode', default='picks', choices=('picks', 'clv', 'settle'),
                    help='picks — ставки, clv — цена закрытия, settle — расчёт и сводка')
    a = ap.parse_args()

    if a.mode == 'clv':
        # перед самым стартом матчей: фиксируем справедливую цену по закрытию.
        # Падение здесь означает потерю наблюдения навсегда: после свистка
        # Pinnacle закрывает рынок. Поэтому шумим, а не возвращаем 0.
        if not run('clv.py', 'update'):
            send_message('⚠️ Бот ставок ОАЭ: не удалось зафиксировать '
                         'закрывающую линию Pinnacle. Наблюдение CLV потеряно.')
            return 1
        return 0

    if a.mode == 'settle':
        # 30 дней, а не 14: перенесённый матч может сыграться сильно позже,
        # а без его счёта ставка так и останется нерассчитанной.
        if not run('update_data.py', '30'):
            send_message('⚠️ Бот ставок ОАЭ: не удалось обновить счета, '
                         'расчёт ставок пропущен.')
            return 1
        from settle import settle, digest
        led = settle()
        if led is None:
            return 0
        text = digest(led)
        if a.dry:
            print(text)
        else:
            send_message(text)
        return 0

    if not a.skip_update:
        run('update_data.py', a.days_back)
    if not run('value_report.py'):
        send_message('⚠️ Бот ставок ОАЭ: расчёт упал, ставки не посчитаны.')
        return 1

    state = load_state()

    # ---- ПРОГНОЗ: каждый тур, один раз на набор матчей (или по --force)
    fc = None
    if os.path.exists(FORECAST):
        fc = pd.read_csv(FORECAST, encoding='utf-8-sig')
        # файл старого формата (до сентября 2026) или пустой -- игнорируем,
        # иначе бот падал на KeyError ещё до отправки чего бы то ни было
        if fc.empty or not {'home', 'away', 'дата', 'p1', 'pX', 'p2'}.issubset(fc.columns):
            print('data/forecast.csv устаревшего формата или пуст -- прогноз не отправлен')
            fc = None
    # Пауза в календаре -- сказать один раз, а не молчать. Осенью 2026 между
    # турами вышло 34 дня, и «прогнозы не приходят» выглядело как поломка.
    nr_path = os.path.join(ROOT, 'data', 'next_round.json')
    if fc is None and os.path.exists(nr_path):
        try:
            nr = json.load(open(nr_path, encoding='utf-8'))
        except Exception:
            nr = {}
        nd = nr.get('next_date')
        if nd and state.get('pause_notice') != nd:
            d = dt.datetime.strptime(nd, '%Y-%m-%d')
            text = (f"⏸ <b>Про-Лига ОАЭ: пауза в календаре</b>\n"
                    f"Ближайший тур {'№' + str(nr['round']) + ' ' if nr.get('round') else ''}"
                    f"— <b>{d:%d.%m.%Y}</b>, матчей в первый день: {nr.get('n_matches', '?')}.\n"
                    f"<i>Прогноз по нему придёт, когда до матчей останется 8 дней "
                    f"(~{(d - dt.timedelta(days=8)):%d.%m}); ставки — только при подтверждении ценой.</i>")
            if a.dry:
                print(text)                      # сухой прогон ничего не помечает
            elif send_message(text):
                state['pause_notice'] = nd
                print(f'уведомление о паузе до {nd} отправлено')
    if fc is not None:
        fkey = forecast_key(fc)
        if a.force or state.get('forecast_key') != fkey:
            ftext = build_forecast(fc)
            if ftext:
                if a.dry:
                    print(ftext)
                elif send_message(ftext):
                    state['forecast_key'] = fkey
                    state['forecast_at'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
                    print(f'прогноз по {len(fc)} матчам отправлен')
        else:
            print('прогноз по этому набору матчей уже отправлен')

    # ---- СТАВКИ: только при подтверждении ценой
    if not os.path.exists(PICKS):
        print('нет data/picks.csv -- кандидатов на ставку нет')
        save_state(state)
        return 0
    picks = pd.read_csv(PICKS)
    picks['ярус'] = picks['ярус'].astype(str)
    msg, fresh = build_message(picks, state, force=a.force)

    if msg is None:
        print('новых ставок нет — сообщение о ставках не отправляю')
        save_state(state)
        return 0

    if a.dry:
        print(msg)
        return 0

    ok = send_message(msg)
    if ok:
        run('clv.py', 'log')       # журнал CLV ведётся автоматически
        for r in fresh:
            state.setdefault('sent', {})[key_of(r)] = {
                'ev': float(r['ev']), 'кэф': float(r['кэф']),
                'ярус': str(r['ярус']),
                'at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}
        # чистим записи старше 30 дней, чтобы state.json не рос
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).strftime('%Y-%m-%d')
        state['sent'] = {k: v for k, v in state['sent'].items()
                         if v.get('at', '9999')[:10] >= cutoff}
    save_state(state)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
