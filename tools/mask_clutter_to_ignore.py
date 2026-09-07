import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ====== 你要改的路径 ======
ANN_DIR = "/home/featurize/data/vaihingen/ann_dir"  # 里面有 train/ val
SUBSETS = ["train", "val"]  # 需要改哪些子集

# ====== label 约定（你现在这套 ISPRS）======
CLUTTER_RAW_ID = 6   # 原始标注值：1..6，其中6=clutter
IGNORE_RAW_ID = 0    # 原始标注值：0=ignore

# 备份目录：会在同级生成 ann_dir_backup_YYYYMMDD_HHMMSS
BACKUP = True

def convert_one(p: Path) -> tuple[int, int]:
    """Return: (num_clutter_pixels_before, num_pixels_changed)"""
    im = Image.open(p)
    arr = np.array(im)

    before = int((arr == CLUTTER_RAW_ID).sum())
    if before == 0:
        return 0, 0

    arr2 = arr.copy()
    arr2[arr2 == CLUTTER_RAW_ID] = IGNORE_RAW_ID
    changed = int((arr != arr2).sum())

    # 保留原模式/调色板（如果是P模式）
    out = Image.fromarray(arr2)
    if im.mode == "P" and "palette" in im.info:
        out = out.convert("P")
        out.putpalette(im.getpalette())

    out.save(p)
    return before, changed

def main():
    ann_dir = Path(ANN_DIR)
    if not ann_dir.exists():
        raise FileNotFoundError(f"ANN_DIR not found: {ann_dir}")

    if BACKUP:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = ann_dir.parent / f"{ann_dir.name}_backup_{ts}"
        print(f"[Backup] Copying {ann_dir} -> {backup_dir} ...")
        shutil.copytree(ann_dir, backup_dir)
        print("[Backup] Done.")

    total_files = 0
    files_touched = 0
    total_clutter_px = 0

    for subset in SUBSETS:
        d = ann_dir / subset
        if not d.exists():
            print(f"[Skip] {d} not found")
            continue

        pngs = sorted(d.rglob("*.png"))
        print(f"[Scan] {subset}: {len(pngs)} png masks")

        for p in pngs:
            total_files += 1
            before, changed = convert_one(p)
            if before > 0:
                files_touched += 1
                total_clutter_px += before

    print("\n===== Summary =====")
    print(f"Total masks scanned: {total_files}")
    print(f"Masks changed:       {files_touched}")
    print(f"Clutter pixels->0:   {total_clutter_px}")
    print("Done.")

if __name__ == "__main__":
    main()