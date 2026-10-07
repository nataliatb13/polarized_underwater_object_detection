#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""result_aggregator — накопление решений по окну кадров (модуль
архитектуры): биномиальное правило «k из n» по каждому каналу и классу
(глава 5). Решение «объект есть» принимается, если в скользящем окне из
n последних кадров класс обнаружен не менее чем в k кадрах.
"""
from collections import deque, defaultdict

import settings as S


class KofN:
    def __init__(self, n=None, k=None):
        self.n = n or S.AGG_N
        self.k = k or S.AGG_K
        self.win = defaultdict(lambda: deque(maxlen=self.n))

    def update(self, channel, dets):
        """dets: [(cls, conf, ...)] кадра -> [(cls, hits, decision)]."""
        present = {c for c, *_ in dets}
        out = []
        for c in range(len(S.CLASSES)):
            w = self.win[(channel, c)]
            w.append(1 if c in present else 0)
            hits = sum(w)
            out.append((c, hits, int(hits >= self.k)))
        return out
