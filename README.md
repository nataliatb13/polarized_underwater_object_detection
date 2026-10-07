# detect_and_nav_project_github

Изолированный конвейер обнаружения объектов интереса и определения курса по поляризационным данным.

## Соответствие архитектуре

| Блок архитектуры                      | Модуль                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `start_stz` — запуск, оркестрация     | `start_stz.py` — оффлайн, последовательно по кадрам                         |
| POL Server / RGB Server               | `file_camera_server.py` — эмуляция камер файлами `.npy` с диска             |
| Compass Server (DCM260B)              | `compas_DCM260B.py` — заглушка, данных нет                                  |
| `pol_preproc`                         | `pol_preproc.py` + `cameras_utils_v3.py`                                    |
| `detect_pol_yolo` / `detect_rgb_yolo` | `detect_pol_yolo.py`, `detect_rgb_yolo.py` + `detect_common.py`             |
| `nav_pol_resnet`                      | `nav_pol_resnet.py` + `nav_utils.py`, `nav_crop.py`                         |
| `result_aggregator`                   | `result_aggregator.py` — правило «k из n»                                   |
| API_on_TCP (ИКИВ)                     | `API_on_TCP.py` — журнал пакетов JSONL                                      |
| Отчёты                                | `nav_report.py` — ошибка курса; `detect_stats.py` — вероятность обнаружения |

## Запуск

### 1. Подготовка данных

По умолчанию папка `17062026` ожидается по пути:

```text
../../17062026
```

относительно директории проекта.

При расположении данных в другом месте задайте переменную окружения:

```bash
export DATA=/путь/к/17062026
```

Для Windows PowerShell:

```powershell
$env:DATA="C:\путь\к\17062026"
```

### 2. Запуск конвейера

```bash
python3 start_stz.py
```
Ссылка на данные датасета:
https://zenodo.org/records/22917961

Результаты сохраняются в директории `result/`:

```text
result/
├── idolp/
├── rgb/
├── nav_session2.csv
├── aggregated.csv
├── ikiv_packets.jsonl
├── nav_yaw_vs_bearing.jpg
└── detection_stats.jpg
    detection_stats.csv
```

Каталоги `idolp/` и `rgb/` содержат кадры с визуализацией bounding box и рассчитанного курса.

### 3. Самотест без данных и весов

Для проверки запуска конвейера без реальных данных и весов:

```bash
UDEPTH_MOCK=1 python3 start_stz.py
```

## Зависимости

Основные Python-зависимости:

```text
numpy
opencv-python
pillow
matplotlib
torch
torchvision
pysolar
ultralytics
```

## Служебные и отчётные модули

Следующие модули не относятся непосредственно к основной схеме архитектуры:

| Модуль                | Назначение                                         |
| --------------------- | -------------------------------------------------- |
| `settings.py`         | Настройки                                          |
| `detect_common.py`    | Общий каркас детекторов                            |
| `cameras_utils_v3.py` | Зависимость `pol_preproc`                          |
| `nav_utils.py`        | Угол на цель из разметки и надпись на кадре        |
| `nav_report.py`       | Построение отчётных фигур                          |
| `detect_stats.py`     | Построение статистики обнаружения                  |
| `nav_crop.py`         | Используется только для чекпойнтов с кадрированием |

## Веса

В каталоге `weights/` используются следующие модели:

| Файл                        | Назначение                        |
| --------------------------- | --------------------------------- |
| `detect_idolp_intra.pt`     | YOLO11s, внутрисессионная модель  |
| `detect_rgb_intra.pt`       | YOLO11s, внутрисессионная модель  |
| `nav_pol_resnet_detect.pth` | ResNet для определения ориентации |

`detect_idolp_intra.pt` и `detect_rgb_intra.pt` — YOLO11s, внутрисессионные модели, обученные на обеих сериях с `seed=0`.

`nav_pol_resnet_detect.pth` — ResNet ориентации, дообученный с опорой на положение объекта интереса.
