from pathlib import Path
import cv2
import numpy as np

for root in ["data/spacenet_buildings", "data/spacenet_roads", "data/loveda"]:
    print("\n", root)
    candidates = [Path(root) / "all" / "masks", Path(root) / "train" / "masks"]
    mask_dir = next((p for p in candidates if p.exists()), None)
    if mask_dir is None:
        print("not found")
        continue
    files = list(mask_dir.glob("*.png"))
    print("masks:", len(files))
    counts = {}
    non_empty = 0
    for f in files[:500]:
        m = cv2.imread(str(f), 0)
        if m is None:
            continue
        u, c = np.unique(m, return_counts=True)
        if len(u) > 1:
            non_empty += 1
        for cls, n in zip(u, c):
            counts[int(cls)] = counts.get(int(cls), 0) + int(n)
    print("non_empty(first 500):", non_empty)
    print("class pixels(first 500):", counts)
