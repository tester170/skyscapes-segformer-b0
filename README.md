# SegFormer-B0 для DLR SkyScapes-Dense (MMSegmentation 1.2.2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tester170/skyscapes-segformer-b0/blob/v1.0/SkyScapes_SegFormer_B0.ipynb)

Семантическая сегментация аэрофотоснимков DLR SkyScapes-Dense (20 классов) моделью SegFormer-B0 в MMSegmentation 1.2.2: обученная модель, код конвейера, точное окружение и блокнот Colab для инференса, оценки и дообучения.

## Запуск в Colab
1. Откройте блокнот кнопкой «Open in Colab» выше.
2. *Среда выполнения → Сменить среду выполнения → GPU*.
3. *Среда выполнения → Выполнить все*. Обучение включается параметром `TRAIN` в разделе 5; свой снимок можно загрузить в разделе 7, веса и результаты скачать кнопками раздела 8.

Блокнот сам клонирует этот репозиторий, скачивает файлы релиза `v1.0` и ставит окружение с точными версиями эксперимента (Python 3.11, PyTorch 2.1.0 + CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2) отдельно от окружения Colab.

## Файлы релиза `v1.0`
| Файл | Назначение |
|---|---|
| `segformer_b0_skyscapes_ep220.pth` | обученная модель (чекпойнт MMSegmentation 1.2.2, эпоха 220) |
| `mit_b0_loveda_encoder.pth` | кодировщик MiT-B0, предобученный на LoveDA (для обучения заново) |
| `mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` | официальная сборка MMCV 2.1.0 OpenMMLab для torch 2.1 / CUDA 12.1 |

Контрольные суммы — в `env/assets.json`; `scripts/fetch.py` проверяет каждый скачанный файл. Набор данных берётся с официального сервера DLR.

## Без блокнота (Linux с GPU NVIDIA)
```bash
git clone --branch v1.0 https://github.com/tester170/skyscapes-segformer-b0.git && cd skyscapes-segformer-b0
bash env/setup_env.sh /opt/py311                      # окружение Python 3.11 с точными версиями
python3 scripts/fetch.py model                        # веса модели из релиза
python3 scripts/fetch.py dataset --out /data          # набор -> /data/SS_Dense
/opt/py311/bin/python scripts/infer.py --input <снимок или папка> --out <папка> --schemes 9 6 5
/opt/py311/bin/python scripts/prepare_data.py --data /data/SS_Dense --cache <кэш>
/opt/py311/bin/python scripts/evaluate.py --cache <кэш> --out <папка>
/opt/py311/bin/python scripts/train.py --cache <кэш> --work-dir <папка запуска> --init best   # или --init encoder
```

## Модель
SegFormer-B0 (MiT-B0 + MLP-декодер, 3,7 млн параметров), 20 классов, патчи 1024, лучшая эпоха 220.
Валидационная выборка: mIoU 36,33 % (20 классов), κ = 0,736, OA 78,0 %; при укрупнении до 9 классов mIoU 62,6 %, до 6 (ISPRS) — 73,9 %, до 5 — 79,0 %. Тестовая выборка SkyScapes опубликована без эталонной разметки, поэтому качество оценено по валидационной.
Маски: коды 1…20 в порядке `splits/classes.json` (как `labels/grayscale` набора). Подробнее — `MODEL_CARD_RU.md`.

## Данные
Набор DLR SkyScapes подготовлен Германским аэрокосмическим центром (DLR): https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes. При использовании ссылайтесь на статью: S. Azimi, C. Henry, L. Sommer, A. Schaumann, E. Vig. *SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes*. ICCV 2019.

## Состав
`configs/` конфигурация · `skyscapes_mmseg/` датасет, аугментации, метрики, хуки · `scripts/` загрузка файлов (fetch.py, get_encoder.py), подготовка данных, обучение, оценка, инференс, матрицы ошибок · `env/` установка и проверка окружения, файлы релиза (`assets.json`) · `splits/` классы и схема укрупнения · `SkyScapes_SegFormer_B0.ipynb` блокнот Colab.

Автор: Александр Стрельцов.
