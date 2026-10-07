#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nav_pol_resnet.py — предикт ориентации по поляризационному кадру
=================================================================
Самодостаточный (не зависит от обучающих скриптов). Нужны только:
    nav_pol_resnet.py + nav_pol_resnet.pth   и   pip install torch torchvision opencv-python numpy pillow pysolar

Вход : путь к .npy (сырой кадр поляризационной камеры). Время съёмки берётся ИЗ ИМЕНИ
       файла (формат камеры: 'YYYY-MM-DD HH_MM_SS.ffffff', часы камеры — пояс из чекпоинта),
       либо --time 'YYYY-MM-DD HH:MM:SS[.ffffff]' если в имени метки нет.
Выход: Roll, Pitch, Yaw (градусы). Yaw абсолютный (mod 360), если модель обучена на полный
       угол относительно Солнца (rel_full*); для осевых моделей — ось курса mod 180.

Использование:
    python nav_pol_resnet.py "pol/2024-07-22 16_39_18.626407.npy"
    python nav_pol_resnet.py frame.npy --time "2024-07-22 16:39:18.626"
    python nav_pol_resnet.py frame.npy --json
    python nav_pol_resnet.py frame.npy --ckpt /path/nav_pol_resnet.pth
"""
import re, sys, json, argparse, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pysolar.solar import get_altitude, get_azimuth


# ===========================================================================
# cameras_utils (инлайн, идентичен репо nataliatb13/underwater)
# ===========================================================================
"""Модуль с функциями для обработки кадров с камер."""




def split_raw_pol(image: NDArray) -> NDArray:
    """Split a polarized image to 4 channels.
    
    The given image with shape `(h, w)` splits to 4 channels image
    with shape `(h/2, w/2, 4)`.

    Parameters
    ----------
    image : NDArray
        Raw polarized image.

    Returns
    -------
    NDArray
        The polarized image that is splitted to 4 channels.
    """
    image = image.squeeze()
    if len(image.shape) != 2:
        raise ValueError('Shape of polarized image must be 2D.')
    ch_90 = image[::2, ::2]
    ch_45 = image[::2, 1::2]
    ch_135 = image[1::2, ::2]
    ch_0 = image[1::2, 1::2]
    return np.stack((ch_0, ch_45, ch_90, ch_135), axis=2)


def calc_Stocks_param(
    ch_0: NDArray, ch_45: NDArray, ch_90: NDArray, ch_135: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """Calculate stokes parameters.

    S1 in range 0.0 <= S0 <= 1.0 for polarized light and
    in range 0.0 <= S0 <= 2.0 fpr unpolarized.
    S1 and S2 in range -1.0 <= S1, S2 <= 1.0
    All parameters avoid zeros by replacing them with 1e-7.

    Parameters
    ----------
    ch_0 : NDArray
        0 angle channel in float.
    ch_45 : NDArray
        45 angle channel in float.
    ch_90 : NDArray
        90 angle channel in float.
    ch_135 : NDArray
        135 angle channel in float.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Return S1, S2, S3 parameters.
    """
    s0 = ch_0 + ch_90 + 1
    # s0 = (ch_0 + ch_90 + ch_45 + ch_135) / 2
    # s0 = ch_0 + ch_90
    s1 = ch_0 - ch_90
    s2 = ch_45 - ch_135
    s0[s0 == 0.0] = 1e-7
    s1[s1 == 0.0] = 1e-7
    s2[s2 == 0.0] = 1e-7
    return s0, s1, s2


def calc_AoLP(s1: NDArray, s2: NDArray) -> NDArray:
    """Calculate angle of polarization.

    Normal values range is from -1.57pi to 1.57pi.

    Parameters
    ----------
    s1 : NDArray
        S1 stoke parameter. Expected float array in range (-1, 1).
    s2 : NDArray
        S2 stoke parameter. Expected float array in range (-1, 1).

    Returns
    -------
    NDArray
        Pixelwice angle of polarization in radians. Range is (0, 2*pi)
    """
    AoLP = 0.5 * np.arctan2(s2, s1)
    return AoLP


def pol_intensity(s1: NDArray, s2: NDArray) -> NDArray:
    """Calculate polarization intensity.

    Normal values range is from 0 to 1.

    Parameters
    ----------
    s1 : NDArray
        S1 stock parameter.
    s2 : NDArray
        S2 stock parameter.

    Returns
    -------
    NDArray
        Calculated polarization intensity.
    """
    return np.sqrt(np.square(s1) + np.square(s2) + 1)
    # return np.sqrt(np.square(s1) + np.square(s2))


def calc_DoLP(s0: NDArray, pol_int: NDArray) -> NDArray:
    """Calculate degree of linear polarization.

    Parameters
    ----------
    s0 : NDArray
        S0 stock parameter.
    pol_int : NDArray
        Polarization intensity.

    Returns
    -------
    NDArray
        Degree of linear polarization.
    """
    DoLP = pol_int / s0
    return DoLP


def hsv_pol(aolp: NDArray, dolp: NDArray, pol_int: NDArray) -> NDArray:
    h = ((aolp + np.pi / 2) * (180 / np.pi)).astype(np.uint8)
    s = (dolp / np.amax(dolp) * 255).astype(np.uint8)
    v = (pol_int / np.amax(pol_int) * 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# ===========================================================================
# ПРЕДОБРАБОТКА (идентична обучению)
# ===========================================================================
def load_pol_as_hsv_bgr(pth: str) -> np.ndarray:
    frame = np.load(pth)
    ch = split_raw_pol(frame).astype(np.float32) / 255.0
    s0, s1, s2 = calc_Stocks_param(ch[..., 0], ch[..., 1], ch[..., 2], ch[..., 3])
    pol_int = pol_intensity(s1, s2)
    aolp = calc_AoLP(s1, s2)
    dolp = calc_DoLP(s0, pol_int)
    return hsv_pol(aolp, dolp, pol_int)


def frame_to_tensor(pth: str, size: int, mean, std) -> torch.Tensor:
    rgb = cv2.cvtColor(load_pol_as_hsv_bgr(pth), cv2.COLOR_BGR2RGB)
    pil = transforms.Resize((size, size))(Image.fromarray(rgb))        # PIL-ресайз, как в кэше обучения
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return tf(np.array(pil)).unsqueeze(0)


# ===========================================================================
# ВРЕМЯ ИЗ ИМЕНИ ФАЙЛА + СОЛНЦЕ
# ===========================================================================
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d+)")   # как в скрипте синхронизации

def parse_ts_from_filename(fname: str):
    m = TS_RE.search(fname)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1).replace("_", ":"), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None

def parse_time_arg(s: str) -> dt.datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError("не разобрано время: {}".format(s))

def sun_position(t_local_naive: dt.datetime, tz: str, lat: float, lon: float) -> Tuple[float, float, dt.datetime]:
    """(sun_az [0,360) от истинного севера, sun_el, время UTC)."""
    t_utc = t_local_naive.replace(tzinfo=ZoneInfo(tz)).astimezone(dt.timezone.utc)
    return float(get_azimuth(lat, lon, t_utc)) % 360.0, float(get_altitude(lat, lon, t_utc)), t_utc


# ===========================================================================
# МОДЕЛЬ (архитектура идентична обучению)
# ===========================================================================
_BACKBONES = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}

class OrientationResNet(nn.Module):
    def __init__(self, backbone="resnet18", dropout=0.3, n_aux=0):
        super().__init__()
        net = _BACKBONES[backbone](weights=None)
        in_f = net.fc.in_features; net.fc = nn.Identity(); self.backbone = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f + n_aux, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(256, 4), nn.Tanh())
    def forward(self, x, aux):
        f = self.backbone(x)
        if aux.shape[1] > 0: f = torch.cat([f, aux], 1)
        return self.head(f)


class NavPolResNet:
    """Загружает чекпоинт и предсказывает Roll/Pitch/Yaw по кадру."""

    def __init__(self, ckpt_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu")
        self.ck = ck
        self.model = OrientationResNet(ck["backbone"], ck.get("dropout", 0.3), ck["n_aux"])
        self.model.load_state_dict(ck["model_state"]); self.model.to(self.device).eval()

    def _aux(self, sun_az: float, sun_el: float) -> torch.Tensor:
        cols = []
        for a in self.ck["aux_names"]:
            if a == "el": cols.append(sun_el / 90.0)
            elif a == "az": r = np.radians(sun_az); cols += [np.sin(r), np.cos(r)]
        return torch.tensor([cols], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, npy_path: str, when: dt.datetime = None) -> dict:
        ck = self.ck
        t = when or parse_ts_from_filename(Path(npy_path).name)
        if t is None:
            raise ValueError("В имени файла нет метки времени ('YYYY-MM-DD HH_MM_SS.ffffff') — задай --time")
        sun_az, sun_el, t_utc = sun_position(t, ck["clock_tz"], ck["site_lat"], ck["site_lon"])

        x = frame_to_tensor(npy_path, ck["img_size"], ck["imagenet_mean"], ck["imagenet_std"]).to(self.device)
        o = self.model(x, self._aux(sun_az, sun_el))[0].cpu().numpy()
        roll = float(o[0] * ck["roll_scale"]); pitch = float(o[1] * ck["pitch_scale"])
        s, c = o[2], o[3]; n = np.sqrt(s * s + c * c + 1e-8); ang = np.degrees(np.arctan2(s / n, c / n))
        decl = ck["declination_deg"]; base, k = ck["base"], ck["k"]

        if k == 1:                                   # rel_full*: полный угол относительно Солнца
            rel = ang % 360.0
            yaw = (rel + sun_az - decl) % 360.0; yaw_axis = yaw % 180.0; mode = "absolute"
        elif base == "rel_axis":                     # ось относительно Солнца -> абсолютная ось
            yaw_axis = ((ang / 2.0) % 180.0 + sun_az - decl) % 180.0; yaw = yaw_axis; mode = "axis_mod180"
        else:                                        # yaw_axis: ось, офсет записи неизвестен
            yaw_axis = (ang / 2.0) % 180.0; yaw = yaw_axis; mode = "axis_mod180_unanchored"

        return {"roll": roll, "pitch": pitch, "yaw": float(yaw), "yaw_axis": float(yaw_axis), "yaw_mode": mode,
                "sun_az": sun_az, "sun_el": sun_el, "time_utc": t_utc.isoformat(),
                "sun_below_horizon": bool(sun_el < 0), "target": ck["target"]}


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Roll/Pitch/Yaw по поляризационному .npy")
    ap.add_argument("npy", help="путь к кадру .npy (время — из имени файла)")
    ap.add_argument("--time", help="время съёмки, если его нет в имени: 'YYYY-MM-DD HH:MM:SS[.ffffff]' (часы камеры)")
    ap.add_argument("--ckpt", default=str(Path(__file__).with_name("nav_pol_resnet.pth")))
    ap.add_argument("--json", action="store_true", help="вывод в JSON")
    args = ap.parse_args()

    nav = NavPolResNet(args.ckpt)
    when = parse_time_arg(args.time) if args.time else None
    r = nav.predict(args.npy, when)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1)); return
    print("Кадр:   {}".format(args.npy))
    print("Время:  {}  (UTC) | Солнце: az {:.1f}°, el {:.1f}°{}".format(
        r["time_utc"], r["sun_az"], r["sun_el"], "  [ПОД ГОРИЗОНТОМ — компас ненадёжен]" if r["sun_below_horizon"] else ""))
    print("Roll:   {:+7.2f}°".format(r["roll"]))
    print("Pitch:  {:+7.2f}°".format(r["pitch"]))
    if r["yaw_mode"] == "absolute":
        print("Yaw:    {:7.2f}°  (абсолютный, mod 360; ось {:.2f}°)".format(r["yaw"], r["yaw_axis"]))
    elif r["yaw_mode"] == "axis_mod180":
        print("Yaw:    {:7.2f}°  (ОСЬ курса mod 180 — модель осевая, смысл 180° не определён)".format(r["yaw"]))
    else:
        print("Yaw:    {:7.2f}°  (ось mod 180 относительно записи, офсет неизвестен)".format(r["yaw"]))


if __name__ == "__main__":
    main()
