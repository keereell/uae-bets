# -*- coding: utf-8 -*-
"""Сопоставление русских названий БЕТСИТИ с названиями 365scores."""

RU_TO_EN = {
    'Аль-Айн': 'Al Ain',
    'Аль-Васл Дубай': 'Al Wasl',
    'Аль-Васл': 'Al Wasl',
    'Шабаб Аль-Ахли Дубай': 'Shabab Al Ahly',
    'Шабаб Аль-Ахли': 'Shabab Al Ahly',
    'Аль-Джазира': 'Jazira Abu Dhabi',
    'Аль-Вахда': 'Al-Wahda',
    'Аль-Вахда Абу-Даби': 'Al-Wahda',
    'Шарджа': 'Sharjah SC',
    'Аль-Шарджа': 'Sharjah SC',
    'Аль-Наср Дубай': 'Al Nasr Dubai',
    'Аль-Наср': 'Al Nasr Dubai',
    'Аджман': 'Ajman Club',
    'Бани Яс': 'Baniyas',
    'Банияс': 'Baniyas',
    'Аль-Дафра': 'Al Dhafra',
    'Аль-Джазира Аль-Хамра': 'Al Dhafra',
    'Хаур-Факкан': 'Khor Fakkan',
    'Аль-Иттихад Калба': 'Kalba',
    'Калба': 'Kalba',
    'Хатта': 'Hatta Club',
    'Дубай Юнайтед': 'Dubai United',
    'Аль-Батаэх': 'Al Bataeh',
    'Дибба Аль-Фуджайра': 'Dibba Al Fujairah',
}


def to_en(ru):
    ru = (ru or '').strip()
    if ru in RU_TO_EN:
        return RU_TO_EN[ru]
    # мягкий поиск по вхождению
    for k, v in RU_TO_EN.items():
        if k in ru or ru in k:
            return v
    return None
