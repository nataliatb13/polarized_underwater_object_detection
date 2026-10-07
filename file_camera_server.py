#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""file_camera_server — заглушки модулей «POL Server» / «RGB Server».

Вместо камер читает сырые .npy серии 2 с диска и выдаёт пары кадров,
сопряжённые по метке времени из имени файла
('YYYY-MM-DD HH_MM_SS.ffffff', допуск PAIR_TOL_S).
"""
import os
import re
import datetime as dt
from dataclasses import dataclass

import settings as S

TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d{1,6})')


def parse_ts(name):
    m = TS_RE.search(name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1).replace('_', ':'),
                                '%Y-%m-%d %H:%M:%S.%f')


@dataclass
class FramePair:
    session: str
    ts: dt.datetime
    stem_ts: str            # компактная метка для сопряжения с разметкой
    pol_npy: str
    rgb_npy: str


def _list(folder):
    out = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith('.npy'):
            continue
        ts = parse_ts(f)
        if ts:
            out.append((ts, os.path.join(folder, f)))
    return out


def find_session_dirs(data=None, series='2'):
    """[(session_name, pol_dir, rgb_dir)]. series: '1' | '2' | 'both'."""
    data = data or S.DATA
    sessions = {}
    for dirpath, dirnames, _ in os.walk(data):
        dirnames.sort()
        for d in dirnames:
            low = d.lower()
            if 'and_rgb' in low:
                continue
            key = None
            if low.startswith('images_pol'):
                key = 'pol'
            elif low.startswith('images_rgb'):
                key = 'rgb'
            if key:
                parent = os.path.relpath(dirpath, data)
                sessions.setdefault(parent, {})[key] = \
                    os.path.join(dirpath, d)
    found = []
    for sess, dd in sorted(sessions.items()):
        sess_norm = sess.replace(os.sep, '_').replace('/', '_')
        turbid = S.TURBID_MARK in sess_norm
        if series == '1' and not turbid:
            continue
        if series == '2' and turbid:
            continue
        if 'pol' not in dd or 'rgb' not in dd:
            continue
        if not any(f.lower().endswith('.npy')
                   for f in os.listdir(dd['pol'])):
            continue
        found.append((sess, dd['pol'], dd['rgb']))
    if not found:
        print('[camera][WARN] сырые сессии (images_pol*/images_rgb* '
              'с .npy) не найдены под ' + data)
    return found


def find_session2_dirs(data=None):
    """[(session_name, pol_dir, rgb_dir)] для серии 2 (совместимость)."""
    return find_session_dirs(data, series='2')


def iter_pairs(data=None):
    """Генератор FramePair по всем сессиям серии 2."""
    for sess, pol_dir, rgb_dir in find_session2_dirs(data):
        pol = _list(pol_dir)
        rgb = _list(rgb_dir)
        j = 0
        n_pair = n_skip = 0
        for ts, ppath in pol:
            while j + 1 < len(rgb) and \
                    abs((rgb[j + 1][0] - ts).total_seconds()) <= \
                    abs((rgb[j][0] - ts).total_seconds()):
                j += 1
            if rgb and abs((rgb[j][0] - ts).total_seconds()) <= S.PAIR_TOL_S:
                n_pair += 1
                yield FramePair(sess, ts, ts.strftime('%Y%m%d_%H%M%S_%f'),
                                ppath, rgb[j][1])
            else:
                n_skip += 1
        print(f'[camera] {sess}: пар {n_pair}, без rgb-пары {n_skip}')
