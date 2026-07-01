from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path('data/loveda')
COLORS = {
    1: (180, 180, 180),
    2: (80, 80, 80),
    3: (120, 160, 220),
    4: (170, 140, 100),
    5: (60, 150, 80),
    6: (160, 200, 80),
}

def make_sample(seed: int, size: int = 512):
    rng = random.Random(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (90 + rng.randint(0, 40), 120 + rng.randint(0, 40), 90 + rng.randint(0, 30))
    mask = np.zeros((size, size), dtype=np.uint8)
    # agriculture / forest / water background patches
    for cls in [6, 5, 3, 4]:
        x1, y1 = rng.randint(0, size-150), rng.randint(0, size-150)
        x2, y2 = min(size, x1 + rng.randint(80, 220)), min(size, y1 + rng.randint(80, 220))
        cv2.rectangle(mask, (x1, y1), (x2, y2), cls, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], -1)
    # roads
    for _ in range(3):
        p1 = (rng.randint(0, size), rng.randint(0, size))
        p2 = (rng.randint(0, size), rng.randint(0, size))
        cv2.line(mask, p1, p2, 2, rng.randint(5, 14))
        cv2.line(img, p1, p2, COLORS[2], rng.randint(5, 14))
    # buildings
    for _ in range(rng.randint(10, 25)):
        x, y = rng.randint(0, size-50), rng.randint(0, size-50)
        w, h = rng.randint(15, 50), rng.randint(15, 50)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 1, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), COLORS[1], -1)
    noise = np.random.default_rng(seed).normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, mask


def main():
    for split, n in [('train', 40), ('val', 10), ('test', 10)]:
        for d in ['images', 'masks']:
            (ROOT / split / d).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, mask = make_sample(seed=1000 + i + {'train':0,'val':100,'test':200}[split])
            name = f'demo_{split}_{i:04d}.png'
            cv2.imwrite(str(ROOT / split / 'images' / name), img)
            cv2.imwrite(str(ROOT / split / 'masks' / name), mask)
    print(f'Demo LoveDA dataset created: {ROOT}')

if __name__ == '__main__':
    main()
