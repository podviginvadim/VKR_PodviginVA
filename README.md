# Critical Infrastructure Segmentation: SpaceNet 2 + SpaceNet 3 + LoveDA

Проект для ВКР по теме сегментации зданий и объектов критической инфраструктуры на спутниковых снимках сверхвысокого разрешения.

В проекте используются **три независимых эксперимента**:

| Датасет | Назначение | Классы |
|---|---|---|
| SpaceNet 2 | специализированная сегментация зданий | `0 background`, `1 building` |
| SpaceNet 3 | специализированная сегментация дорог | `0 background`, `1 road` |
| LoveDA | дополнительный многоклассовый эксперимент | `0 background`, `1 building`, `2 road`, `3 water`, `4 barren`, `5 forest`, `6 agricultural` |

Ансамбль критической инфраструктуры используется **только** для SpaceNet:

```text
Buildings model SpaceNet 2 + Roads model SpaceNet 3 = Critical Infrastructure Map
```

LoveDA не смешивается со SpaceNet и используется как отдельный академический benchmark в главе 4.

---

## 1. Установка через venv

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Ожидаемая структура исходных данных

### SpaceNet 2

```text
data/raw/spacenet2/train/PS-RGB
data/raw/spacenet2/train/geojson_buildings
```

### SpaceNet 3

```text
data/raw/spacenet3/PS-RGB
data/raw/spacenet3/geojson_roads
```

### LoveDA

Укажи папку с распакованным LoveDA в `--raw-root`. Скрипт ищет пары `images_png/masks_png` или `images/masks`.

---

## 3. Подготовка SpaceNet 2 и SpaceNet 3

Для быстрой проверки:

```bash
python3 src/prepare_spacenet.py \
  --sn2-images data/raw/spacenet2/train/PS-RGB \
  --sn2-geojson data/raw/spacenet2/train/geojson_buildings \
  --sn3-images data/raw/spacenet3/PS-RGB \
  --sn3-geojson data/raw/spacenet3/geojson_roads \
  --out-root data \
  --max-items 50
```

Будет создано:

```text
data/spacenet_buildings/train|val|test/images|masks
data/spacenet_roads/train|val|test/images|masks
```

Без ограничения:

```bash
python3 src/prepare_spacenet.py \
  --sn2-images data/raw/spacenet2/train/PS-RGB \
  --sn2-geojson data/raw/spacenet2/train/geojson_buildings \
  --sn3-images data/raw/spacenet3/PS-RGB \
  --sn3-geojson data/raw/spacenet3/geojson_roads \
  --out-root data
```

---

## 4. Подготовка LoveDA

```bash
python3 src/prepare_loveda.py \
  --raw-root data/raw/loveda \
  --out data/loveda \
  --label-format auto
```

Для быстрой проверки:

```bash
python3 src/prepare_loveda.py \
  --raw-root data/raw/loveda \
  --out data/loveda \
  --max-items 100
```

---

## 5. Проверка, что маски не пустые

```bash
python3 - <<'PY'
import cv2, numpy as np
from pathlib import Path

for root in ['data/spacenet_buildings', 'data/spacenet_roads', 'data/loveda']:
    print('\n', root)
    files = list(Path(root, 'all' if 'spacenet' in root else 'train', 'masks').glob('*.png'))
    print('masks:', len(files))
    non_empty = 0
    for f in files[:200]:
        m = cv2.imread(str(f), 0)
        if m is not None and len(np.unique(m)) > 1:
            non_empty += 1
    print('non_empty in first 200:', non_empty)
PY
```

---

## 6. Обучение моделей

### SpaceNet 2 — здания

```bash
python3 src/train.py --dataset buildings --model unet --epochs 5 --batch-size 1 --image-size 256 --workers 0
python3 src/train.py --dataset buildings --model segformer_b0 --epochs 5 --batch-size 1 --image-size 256 --workers 0
```

### SpaceNet 3 — дороги

```bash
python3 src/train.py --dataset roads --model unet --epochs 5 --batch-size 1 --image-size 256 --workers 0
python3 src/train.py --dataset roads --model segformer_b0 --epochs 5 --batch-size 1 --image-size 256 --workers 0
```

### LoveDA — многоклассовая сегментация

```bash
python3 src/train.py --dataset loveda --model unet --epochs 5 --batch-size 1 --image-size 256 --workers 0
python3 src/train.py --dataset loveda --model segformer_b0 --epochs 5 --batch-size 1 --image-size 256 --workers 0
```

На CPU рекомендуется `unet` или `segformer_b0`. `segformer_b2` лучше запускать на GPU.

---

## 7. Оценка моделей

```bash
python3 src/evaluate.py --dataset buildings --model unet
python3 src/evaluate.py --dataset roads --model unet
python3 src/evaluate.py --dataset loveda --model unet
```

Для SegFormer:

```bash
python3 src/evaluate.py --dataset buildings --model segformer_b0
python3 src/evaluate.py --dataset roads --model segformer_b0
python3 src/evaluate.py --dataset loveda --model segformer_b0
```

---

## 8. Инференс одной модели

### Здания

```bash
python3 src/inference.py \
  --dataset buildings \
  --model unet \
  --checkpoint models/buildings/unet/best_model.pth \
  --image data/spacenet_buildings/test/images/IMAGE.png \
  --output results/buildings_prediction.png
```

### Дороги

```bash
python3 src/inference.py \
  --dataset roads \
  --model unet \
  --checkpoint models/roads/unet/best_model.pth \
  --image data/spacenet_roads/test/images/IMAGE.png \
  --output results/roads_prediction.png
```

### LoveDA

```bash
python3 src/inference.py \
  --dataset loveda \
  --model unet \
  --checkpoint models/loveda/unet/best_model.pth \
  --image data/loveda/test/images/IMAGE.png \
  --output results/loveda_prediction.png
```

---

## 9. Ансамбль критической инфраструктуры

Ансамбль объединяет только две специализированные модели SpaceNet:

```bash
python3 src/predict_critical_infrastructure.py \
  --buildings-model unet \
  --roads-model unet \
  --buildings-checkpoint models/buildings/unet/best_model.pth \
  --roads-checkpoint models/roads/unet/best_model.pth \
  --image data/spacenet_buildings/test/images/IMAGE.png \
  --output results/critical_infrastructure \
  --image-size 256
```

Итоговые классы ансамбля:

```text
0 = background
1 = building
2 = road
```

Сохраняются файлы:

```text
*_buildings_mask.png
*_roads_mask.png
*_combined_mask.png
*_combined_color.png
*_overlay.png
```

---

## 10. Как описать в ВКР

Основные эксперименты:

1. SpaceNet 2: U-Net vs SegFormer для сегментации зданий.
2. SpaceNet 3: U-Net vs SegFormer для сегментации дорог.
3. Ансамбль SpaceNet Buildings + SpaceNet Roads для получения единой карты критической инфраструктуры.

Дополнительный эксперимент:

4. LoveDA: U-Net vs SegFormer на многоклассовой сегментации земного покрова для проверки устойчивости подхода.

LoveDA не включается в ансамбль, так как это не специализированный датасет критической инфраструктуры, а общий benchmark сегментации сцены.
# VKR_PodviginVA
