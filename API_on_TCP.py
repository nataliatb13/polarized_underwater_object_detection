#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API_on_TCP — заглушка модуля ИКИВ.

В боевом контуре — TCP/UDP-обмен с оператором/САУ АНПА; в оффлайн-
прогоне пакеты пишутся в jsonl-журнал (тот же состав данных: тип,
рамки, вероятности, курсовой угол, агрегированные решения).
"""
import json

import settings as S


class IKIV:
    def __init__(self, path=None):
        self.f = open(path or S.IKIV_LOG, 'w', encoding='utf-8')

    def send(self, packet: dict):
        self.f.write(json.dumps(packet, ensure_ascii=False) + '\n')

    def close(self):
        self.f.close()
