# VoiceClassifier

Базовый классификатор пола по голосу (2 класса: `0`/`1`) на основе простых акустических признаков и небольшой нейросети на PyTorch.

Скрипт [biometry.py](biometry.py) делает всё в одном запуске:

1) читает тренировочные данные из папки `./train`;
2) обучает модель;
3) читает тестовые `.wav` из `./test`;
4) пишет предсказания в `answers.tsv`.

## Быстрый старт

### 1) Установка

Рекомендуется Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Если `torch` не ставится через `requirements.txt`, установите его по инструкции с сайта PyTorch (CPU/CUDA зависит от вашей машины). Например для CPU:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 2) Подготовка данных

Ожидаемая структура:

```
.
├── biometry.py
├── data_converter.py
├── model.py
├── requirements.txt
├── train/
│   ├── targets.tsv
│   ├── <id1>.wav
│   ├── <id2>.wav
│   └── ...
└── test/
	├── <idA>.wav
	├── <idB>.wav
	└── ...
```

Файл `train/targets.tsv` — табличка вида:

```
<sample_id>\t<gender>
```

где:

- `sample_id` — имя wav-файла без расширения
- `gender` — метка класса `0` или `1`

Пример:

```
00001	0
00002	1
```

### 3) Запуск обучения и инференса

```bash
python biometry.py
```

На выходе появится `answers.tsv` в корне проекта.

## Формат вывода

`answers.tsv` содержит строки:

```
<sample_id>\t<predicted_gender>
```

`predicted_gender` — целое число `0` или `1`.

## Как устроена модель

Пайплайн признаков и модели (см. [data_converter.py](data_converter.py), [model.py](model.py)):

- Для каждого аудио строится mel-спектрограмма (`n_mels=128`, `n_fft=2048`, `hop_length=512`).
- Признак — конкатенация статистик по времени для каждой mel-полосы: `mean`, `std`, `max`.
- Классификатор — MLP: `input -> 512 -> 256 -> 2` + Dropout.

## Настройка

Основные параметры (число эпох, lr, batch size, доля валидации) задаются прямо в [biometry.py](biometry.py).

## Частые проблемы

- Не ставится `torch`: ставьте PyTorch отдельно под вашу платформу (CPU/CUDA) и затем остальные зависимости.
- Ошибки чтения `.wav`: иногда нужны системные аудио-зависимости. Попробуйте обновить `pip`, переустановить `librosa` и убедиться, что файлы — валидные WAV.

