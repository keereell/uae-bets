# -*- coding: utf-8 -*-
"""
ЕДИНЫЙ КОНТРАКТ ДЛЯ ВСЕХ БУКМЕКЕРОВ.

Зачем это существует. С одной мягкой конторой валуйных ставок не возникает
вообще: за 2.2 суток наблюдений (72 пары снимок × матч) цена БЕТСИТИ ни разу
не превысила справедливую цену Pinnacle, лучший результат -0.5%. Причина
арифметическая -- маржа одной конторы (~8%) перекрывает любое её отставание
от острой линии. Прибыльные стратегии в литературе устроены иначе: они берут
МАКСИМУМ цены по многим конторам (Kaunitz и др. 2017 -- 32 конторы, модель
не используется вовсе; Buchdahl -- 24 150 ставок, +1.81%).

Поэтому единица работы здесь -- не «контора», а «цена на исход». Каждый
адаптер обязан уметь ровно одно: вернуть свою линию по лиге ОАЭ в общем
виде. Всё остальное -- нормализация, консенсус, поиск перевеса -- делается
поверх и одинаково для всех.

СТРУКТУРА АДАПТЕРА. Модуль в src/books/ с функцией:

    def fetch() -> list[Quote]

Больше ничего не требуется. Адаптер НЕ должен: искать валуй, считать
вероятности, знать про Pinnacle, писать в файлы, падать при сетевой ошибке
тише чем исключением.

КЛЮЧ ИСХОДА (поле `sel`) -- канонический, одинаковый у всех контор:
    '1' | 'X' | '2'                     -- исход матча
    '1X' | '12' | 'X2'                  -- двойной шанс
    'O2.5' | 'U2.5'                     -- тотал больше/меньше (число как есть)
    'AH1-0.25' | 'AH2+1.5'              -- азиатская фора команде 1 или 2
    'EH1-1' | 'EH2+2'                   -- европейская (целочисленная) фора
Линия всегда записана СО СТОРОНЫ ТОЙ КОМАНДЫ, которой она даётся, со знаком.
Четвертные линии (0.25, 0.75) пишутся как есть -- расщепление делает pricing.
"""
import importlib
import os
import pkgutil
from dataclasses import dataclass, field
from typing import Optional

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')

# Конторы, в которых пользователь может реально поставить. Остальные
# источники (агрегаторы) идут только в консенсус: их цена нам недоступна,
# но она уточняет оценку истинной вероятности.
BETTABLE = {'betcity', 'fonbet', 'leon', 'olimp', 'zenit', 'marathon',
            'baltbet', 'betboom'}


@dataclass
class Quote:
    """Одна цена на один исход одного матча у одной конторы."""
    book: str                      # ключ конторы, например 'leon'
    home: str                      # каноническое английское имя, см. norm_team
    away: str
    sel: str                       # канонический ключ исхода, см. модульный docstring
    price: float                   # десятичный коэффициент
    kickoff: Optional[float] = None    # unix-секунды, если контора его отдаёт
    raw_home: str = ''             # как назвала команду сама контора (для отладки)
    raw_away: str = ''
    extra: dict = field(default_factory=dict)

    def key(self):
        return (self.home, self.away, self.sel)


# ---------------------------------------------------------------------------
#                        НОРМАЛИЗАЦИЯ НАЗВАНИЙ КОМАНД
# ---------------------------------------------------------------------------
# Канон -- имена 365scores, на них уже завязаны matches.csv и модель.
# Ключи ниже -- в нижнем регистре, без пунктуации (см. _slug).
# Один неверный маппинг = потерянная ставка или, хуже, сравнение цен на
# РАЗНЫЕ матчи, поэтому список явный и без «умного» угадывания.
_ALIASES = {
    'Al Ain': ['al ain', 'alain', 'al ain fc', 'аль айн', 'аль-айн'],
    'Al Wasl': ['al wasl', 'al wasl dubai', 'аль васл', 'аль васл дубай', 'аль-васл'],
    'Shabab Al Ahly': ['shabab al ahly', 'shabab al ahli', 'shabab al-ahli dubai',
                       'shabab al ahli dubai', 'al ahli shabab', 'шабаб аль ахли',
                       'шабаб аль ахли дубай', 'шабаб аль-ахли'],
    'Jazira Abu Dhabi': ['al jazira', 'jazira', 'al jazira abu dhabi',
                         'jazira abu dhabi', 'аль джазира', 'аль-джазира'],
    'Al-Wahda': ['al wahda', 'al wahda abu dhabi', 'wahda abu dhabi', 'wahda',
                 'аль вахда', 'аль вахда абу даби', 'аль-вахда'],
    'Sharjah SC': ['sharjah', 'al sharjah', 'sharjah sc', 'sharjah fc',
                   'шарджа', 'аль шарджа', 'аль-шарджа'],
    'Al Nasr Dubai': ['al nasr', 'al nasr dubai', 'al nasr sc', 'аль наср',
                      'аль наср дубай', 'аль-наср'],
    'Ajman Club': ['ajman', 'ajman club', 'аджман'],
    'Baniyas': ['baniyas', 'bani yas', 'baniyas sc', 'банияс', 'бани яс'],
    'Al Dhafra': ['al dhafra', 'dhafra', 'al zafra', 'аль дафра', 'аль-дафра'],
    'Khor Fakkan': ['khor fakkan', 'khorfakkan', 'khor fakkan club',
                    'хор факкан', 'хаур факкан', 'хаур-факкан'],
    'Kalba': ['kalba', 'al ittihad kalba', 'ittihad kalba', 'al orooba kalba',
              'калба', 'аль иттихад калба', 'аль-иттихад калба'],
    'Hatta Club': ['hatta', 'hatta club', 'хатта'],
    'Dubai United': ['dubai united', 'united fc', 'united', 'dubai united fc',
                     'дубай юнайтед'],
    'Al Bataeh': ['al bataeh', 'bataeh', 'al bataeh club', 'аль батаэх', 'аль-батаэх'],
    'Dibba Al Fujairah': ['dibba al fujairah', 'dibba al-fujairah', 'dibba',
                          'дибба аль фуджайра', 'дибба'],
    'Al Urooba': ['al urooba', 'urooba', 'аль уруба'],
    'Dubba Al Husun': ['dibba al hisn', 'dibba al-hisn', 'al hisn',
                       'дибба аль хисн'],
}


def _slug(s):
    s = (s or '').strip().lower().replace('ё', 'е')
    for ch in '-–—.,()"\'`':
        s = s.replace(ch, ' ')
    for junk in (' fc', ' sc', ' club', ' team'):
        if s.endswith(junk):
            s = s[: -len(junk)]
    return ' '.join(s.split())


_LOOKUP = {}
for _canon, _names in _ALIASES.items():
    _LOOKUP[_slug(_canon)] = _canon
    for _n in _names:
        _LOOKUP[_slug(_n)] = _canon


def norm_team(name):
    """
    Любое написание -> каноническое имя 365scores, либо None.

    Возврат None -- это НЕ ошибка адаптера, а сигнал «матч не из этой лиги»
    (у контор в том же турнире встречаются молодёжные и кубковые команды).
    Вызывающий обязан такие котировки отбрасывать, а не угадывать.
    """
    s = _slug(name)
    if not s:
        return None
    if s in _LOOKUP:
        return _LOOKUP[s]
    # запасной проход: точное вхождение по словам, без нечёткого сравнения --
    # 'al nasr' и 'al ain' слишком похожи, чтобы доверять расстояниям
    for k, v in _LOOKUP.items():
        if len(k) >= 6 and (k in s or s in k):
            return v
    return None


# ---------------------------------------------------------------------------
#                            КЛЮЧИ ИСХОДОВ
# ---------------------------------------------------------------------------
def sel_total(over, line):
    """-> 'O2.5' / 'U2.5'"""
    return ('O' if over else 'U') + f'{float(line):g}'


def sel_ah(team, line):
    """team = 1 или 2, line со знаком со стороны этой команды. -> 'AH1-0.25'"""
    v = float(line)
    return f'AH{int(team)}{"+" if v >= 0 else "-"}{abs(v):g}'


def sel_eh(team, line):
    v = float(line)
    return f'EH{int(team)}{"+" if v >= 0 else "-"}{abs(v):g}'


# ---------------------------------------------------------------------------
#                              РЕЕСТР АДАПТЕРОВ
# ---------------------------------------------------------------------------
def available():
    """Ключи всех модулей-адаптеров в этом пакете."""
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(m.name for m in pkgutil.iter_modules([here])
                  if not m.name.startswith('_'))


def fetch_book(key, timeout_note=''):
    """
    Снять линию одной конторы. Исключения НЕ глушим наверх молча:
    возвращаем (котировки, ошибка), чтобы сборщик мог отличить
    «контора не котирует лигу» от «мы сломались».
    """
    try:
        mod = importlib.import_module(f'{__name__}.{key}')
        qs = mod.fetch() or []
        return [q for q in qs if q.home and q.away], None
    except Exception as e:
        return [], f'{type(e).__name__}: {e}'


def fetch_all(keys=None, verbose=True):
    """-> (список Quote со всех контор, {ключ: текст ошибки})"""
    out, errs = [], {}
    for k in (keys if keys is not None else available()):
        qs, err = fetch_book(k)
        if err:
            errs[k] = err
        out.extend(qs)
        if verbose:
            n_m = len({(q.home, q.away) for q in qs})
            print(f'  {k:<12} котировок {len(qs):>5}  матчей {n_m:>2}'
                  + (f'  ОШИБКА: {err[:70]}' if err else ''))
    # ЧАСТИЧНЫЙ ответ -- тоже сбой, только тихий. Адаптер, у которого часть
    # матчей отвалилась по 429/5xx, возвращает неполную линию с err=None, и
    # консенсус для остальных матчей тихо худеет. Сверяем покрытие каждой
    # конторы с объединением матчей по всем -- это ловит любой адаптер разом.
    fixtures = {(q.home, q.away) for q in out}
    for k in (keys if keys is not None else available()):
        if k in errs:
            continue
        mine = {(q.home, q.away) for q in out
                if q.book == k or q.book.startswith(k[:2] + ':')}
        if mine and len(mine) < len(fixtures):
            errs[k] = f'частично: {len(mine)} из {len(fixtures)} матчей'
            if verbose:
                print(f'  {k:<12} ЧАСТИЧНО: {len(mine)} из {len(fixtures)} матчей')
    return out, errs
