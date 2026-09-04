"""
Отправка сообщений в Telegram через Bot API.
Длинные сообщения режутся по границе строк (лимит Telegram — 4096).
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional

import requests

TELEGRAM_LIMIT = 4000  # запас под лимит 4096


def _split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list:
    """Режет по строкам, чтобы не рвать разметку посреди тега."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ''
    for line in text.split('\n'):
        if len(cur) + len(line) + 1 > limit:
            if cur:
                chunks.append(cur.rstrip())
            cur = ''
            while len(line) > limit:          # аварийный случай: строка длиннее лимита
                chunks.append(line[:limit])
                line = line[limit:]
        cur += line + '\n'
    if cur.strip():
        chunks.append(cur.rstrip())
    return chunks


def _redact(s: str, token: str) -> str:
    """Убрать токен из любого текста, который может уйти в лог."""
    s = str(s)
    if token:
        s = s.replace(token, '<токен скрыт>')
        head = token.split(':')[0]
        if head and len(head) >= 6:
            s = s.replace(head, '<токен скрыт>')
    return s


def send_message(text: str, bot_token: Optional[str] = None,
                 chat_id: Optional[str] = None,
                 parse_mode: str = 'HTML',
                 reply_markup: Optional[dict] = None) -> bool:
    """
    Отправляет сообщение. При отсутствии токена печатает в консоль —
    это штатный режим для локального прогона и для проверки в CI.
    """
    bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')
    if not bot_token or not chat_id:
        print('[telegram] нет токена или chat_id, печатаю сообщение:')
        print(text)
        return False

    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    chunks = _split_message(text)
    ok = True
    for i, chunk in enumerate(chunks):
        payload = {
            'chat_id': chat_id,
            'text': chunk,
            'parse_mode': parse_mode,
            'disable_web_page_preview': 'true',
        }
        if reply_markup and i + 1 == len(chunks):
            payload['reply_markup'] = json.dumps(reply_markup)
        try:
            r = requests.post(url, data=payload, timeout=30)
            if r.status_code != 200:
                print(f'[telegram] часть {i+1} не ушла: {r.status_code} '
                      f'{_redact(r.text, bot_token)}')
                ok = False
            if i + 1 < len(chunks):
                time.sleep(0.3)
        except Exception as e:
            # Текст исключения requests почти всегда содержит полный URL,
            # а в URL лежит токен. Логи прогонов Actions на публичном
            # репозитории видны всем. Маскировка секретов на стороне GitHub
            # это ловит, но закладываться на неё как на единственный барьер
            # нельзя: она срабатывает только на точное совпадение строки.
            print(f'[telegram] часть {i+1}, ошибка: {_redact(str(e), bot_token)}')
            ok = False
    return ok


def get_chat_id(bot_token: Optional[str] = None) -> None:
    """
    Подсказка для первой настройки: напиши боту любое сообщение,
    затем запусти `python telegram_sender.py` — он покажет chat_id.
    """
    bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
    if not bot_token:
        print('нет TELEGRAM_BOT_TOKEN в окружении')
        return
    r = requests.get(f'https://api.telegram.org/bot{bot_token}/getUpdates', timeout=20)
    data = r.json()
    if not data.get('result'):
        print('обновлений нет. Напиши боту любое сообщение в Telegram и запусти снова.')
        return
    seen = {}
    for u in data['result']:
        msg = u.get('message') or u.get('channel_post') or {}
        ch = msg.get('chat') or {}
        if ch.get('id'):
            seen[ch['id']] = ch.get('username') or ch.get('title') or ch.get('first_name')
    for cid, who in seen.items():
        print(f'chat_id = {cid}   ({who})')


if __name__ == '__main__':
    get_chat_id()
