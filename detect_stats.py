#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_stats — вероятность обнаружения объектов по кадрам и общая
статистика (строится в конце start_stz, можно запускать отдельно).

Источник: журнал пакетов ИКИВ (<OUT_DIR>/ikiv_packets.jsonl) — детекции
обоих каналов по каждому кадру; истина — ручная разметка
(labels_full_v1/pol для idolp, labels_full_v1/rgb для rgb).
Кадр считается обнаруженным по классу, если есть детекция этого класса
с уверенностью >= CONF; вероятность обнаружения по кадрам — скользящая
доля обнаружений в окне ROLL_W среди кадров, где класс есть в разметке.

Выход: <OUT_DIR>/detection_stats.jpg, detection_stats.csv (по кадрам),
консольная сводка: m/n, ДИ95 (Клоппер–Пирсон), средняя уверенность,
ложные обнаружения на кадр.
"""
import os
import csv
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import settings as S

ROLL_W = 15          # окно скользящей вероятности, кадров (~5 с)
CHANNELS = ('idolp', 'rgb')
LABEL_STREAM = {'idolp': 'pol', 'rgb': 'rgb'}
CLS_RU = {'diver': 'водолаз', 'submarine': 'аппарат', 'shark': 'акула'}


def clopper_pearson(m, n, alpha=0.05):
    if n == 0:
        return float('nan'), float('nan')
    try:
        from scipy.stats import beta
        lo = 0.0 if m == 0 else beta.ppf(alpha / 2, m, n - m + 1)
        hi = 1.0 if m == n else beta.ppf(1 - alpha / 2, m + 1, n - m)
        return float(lo), float(hi)
    except Exception:                       # Вилсон как запасной вариант
        p = m / n
        z = 1.96
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d


def label_index(stream):
    """ts_compact -> set(классов) по labels_full_v1/<stream> (обе серии)."""
    root = os.path.join(S.DATA, 'labels_full_v1', stream)
    idx = {}
    if not os.path.isdir(root):
        return idx
    for sess in os.listdir(root):
        d = os.path.join(root, sess)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('.txt'):
                continue
            cls = set()
            for ln in open(os.path.join(d, f), encoding='utf-8'):
                p = ln.split()
                if len(p) == 5:
                    cls.add(int(p[0]))
            idx[f[:-4].split('_', 1)[-1]] = cls
    return idx


def load_packets(path):
    out = []
    for ln in open(path, encoding='utf-8'):
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def build(jsonl_path=None, out_jpg=None):
    jsonl_path = jsonl_path or S.IKIV_LOG
    out_jpg = out_jpg or os.path.join(S.OUT_DIR, 'detection_stats.jpg')
    if not os.path.exists(jsonl_path):
        print(f'[stats] нет журнала {jsonl_path}')
        return None
    packets = load_packets(jsonl_path)
    labels = {ch: label_index(LABEL_STREAM[ch]) for ch in CHANNELS}
    n_cls = len(S.CLASSES)
    rows = []
    for p in packets:
        stem = p['stem']
        ts = stem.split('_', 1)[-1] if '_' in stem else stem
        row = {'stem': stem}
        for ch in CHANNELS:
            dets = p.get(ch) or []
            gt = labels[ch].get(ts)
            for c in range(n_cls):
                confs = [d[1] for d in dets if int(d[0]) == c
                         and float(d[1]) >= S.CONF]
                row[f'{ch}_gt_{c}'] = ('' if gt is None
                                       else int(c in gt))
                row[f'{ch}_det_{c}'] = int(bool(confs))
                row[f'{ch}_conf_{c}'] = (f'{max(confs):.3f}' if confs
                                         else '')
            row[f'{ch}_n_det'] = len([d for d in dets
                                      if float(d[1]) >= S.CONF])
        rows.append(row)
    csv_path = os.path.join(S.OUT_DIR, 'detection_stats.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.2),
                           gridspec_kw={'width_ratios': [2.2, 1]})
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    summary = []
    for i, ch in enumerate(CHANNELS):
        a = ax[i, 0]
        b = ax[i, 1]
        rates, los, his, labs = [], [], [], []
        for c in range(n_cls):
            gt = np.array([r[f'{ch}_gt_{c}'] for r in rows], dtype=object)
            det = np.array([r[f'{ch}_det_{c}'] for r in rows])
            has = np.array([g == 1 for g in gt])
            idx = np.where(has)[0]
            if len(idx) == 0:
                continue
            hit = det[idx].astype(float)
            w_ = min(ROLL_W, len(hit))
            roll = np.convolve(hit, np.ones(w_) / w_, mode='valid')
            xs = idx[w_ // 2: w_ // 2 + len(roll)]
            a.plot(xs, roll, '-', color=colors[c], lw=1.4,
                   label=f'{CLS_RU.get(S.CLASSES[c], S.CLASSES[c])} '
                         f'(в разметке {len(idx)} кадров)')
            m, n = int(hit.sum()), len(hit)
            lo, hi = clopper_pearson(m, n)
            confs = [float(r[f'{ch}_conf_{c}']) for r in rows
                     if r[f'{ch}_conf_{c}']]
            rates.append(m / n)
            los.append(lo)
            his.append(hi)
            labs.append(CLS_RU.get(S.CLASSES[c], S.CLASSES[c]))
            summary.append((ch, S.CLASSES[c], m, n, lo, hi,
                            float(np.mean(confs)) if confs else float('nan')))
        # ложные обнаружения: детекция класса, которого нет в разметке
        fp = 0
        n_lab = 0
        for r in rows:
            for c in range(n_cls):
                if r[f'{ch}_gt_{c}'] != '':
                    n_lab += 1
                    if r[f'{ch}_gt_{c}'] == 0 and r[f'{ch}_det_{c}'] == 1:
                        fp += 1
        a.set_ylim(-0.02, 1.05)
        a.set_xlabel('номер кадра (по времени)')
        a.set_ylabel(f'вероятность обнаружения (окно {ROLL_W} кадров)')
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc='lower left')
        a.set_title(f'{"а" if i == 0 else "в"}) {ch}: обнаружение по '
                    f'кадрам (порог {S.CONF})', fontsize=10)
        x = np.arange(len(rates))
        b.bar(x, rates, color=[colors[c] for c in range(len(rates))],
              alpha=0.85)
        b.errorbar(x, rates,
                   yerr=[np.array(rates) - np.array(los),
                         np.array(his) - np.array(rates)],
                   fmt='none', ecolor='k', capsize=4, lw=1)
        for xi, rr, hh in zip(x, rates, his):
            b.text(xi, min(hh + 0.02, 1.02), f'{rr:.2f}', ha='center',
                   fontsize=9)
        b.set_xticks(x, labs, fontsize=9)
        b.set_ylim(0, 1.12)
        b.set_ylabel('доля кадров с обнаружением')
        b.grid(alpha=0.3, axis='y')
        b.set_title(f'{"б" if i == 0 else "г"}) {ch}: доля кадров с '
                    f'обнаружением (ДИ95)\nложных обнаружений: {fp} на '
                    f'{len(rows)} кадров', fontsize=9)
    fig.suptitle(f'Вероятность обнаружения объектов интереса: кадров '
                 f'{len(rows)}; истина — ручная разметка соответствующего '
                 f'потока', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_jpg, dpi=140)
    plt.close(fig)
    print(f'[stats] фигура: {out_jpg}; таблица: {csv_path}')
    for ch, cls, m, n, lo, hi, mc in summary:
        print(f'[stats] {ch:5} {cls:9}: {m}/{n} = {m / n:.3f} '
              f'[ДИ95 {lo:.3f}; {hi:.3f}], средняя уверенность {mc:.2f}')
    return out_jpg


def _selftest():
    import tempfile
    import sys
    root = tempfile.mkdtemp()
    S.DATA = os.path.join(root, 'Rodnik_17062026')
    S.OUT_DIR = os.path.join(root, 'result')
    S.IKIV_LOG = os.path.join(S.OUT_DIR, 'ikiv_packets.jsonl')
    os.makedirs(S.OUT_DIR)
    for stream in ('pol', 'rgb'):
        d = os.path.join(S.DATA, 'labels_full_v1', stream, 'sA')
        os.makedirs(d)
        for i in range(40):
            open(os.path.join(d, f'{i:06d}_2026061{i:02d}_1.txt'), 'w').write(
                '1 0.5 0.5 0.1 0.1\n' + ('2 0.6 0.6 0.1 0.1\n' if i % 2 else ''))
    with open(S.IKIV_LOG, 'w', encoding='utf-8') as f:
        for i in range(40):
            dets = [[1, 0.9, 0, 0, 10, 10]]
            if i % 4 == 1:
                dets.append([2, 0.5, 0, 0, 5, 5])
            if i % 10 == 0:
                dets.append([0, 0.6, 0, 0, 5, 5])      # ложный водолаз
            f.write(json.dumps({'stem': f'{i:06d}_2026061{i:02d}_1',
                                'idolp': dets, 'rgb': dets[:1]}) + '\n')
    out = build()
    assert out and os.path.getsize(out) > 40000
    rows = list(csv.DictReader(open(os.path.join(S.OUT_DIR,
                                                 'detection_stats.csv'))))
    assert len(rows) == 40 and rows[0]['idolp_det_1'] == '1'
    lo, hi = clopper_pearson(37, 40)
    assert 0.79 < lo < 0.86 and 0.97 < hi <= 1.0
    print('detect_stats SELFTEST OK')


if __name__ == '__main__':
    if os.environ.get('UDEPTH_MOCK') == '1':
        _selftest()
    else:
        build()
