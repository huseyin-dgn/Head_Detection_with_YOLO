import random
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo")
DST = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo_rebalanced")

TARGET_VAL = 350
TARGET_TEST = 700
SEED = 42

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_tree():
    for split in ["train", "val", "test"]:
        ensure(DST / "images" / split)
        ensure(DST / "labels" / split)

    for split in ["train", "val", "test"]:
        for p in (SRC / "images" / split).glob("*.*"):
            shutil.copy2(p, DST / "images" / split / p.name)
        for p in (SRC / "labels" / split).glob("*.txt"):
            shutil.copy2(p, DST / "labels" / split / p.name)

def list_stems(img_dir: Path):
    return sorted([p.stem for p in img_dir.glob("*.*") if p.is_file()])

def move_pair(stem: str, from_split: str, to_split: str):
    img_from = DST / "images" / from_split
    lbl_from = DST / "labels" / from_split
    img_to = DST / "images" / to_split
    lbl_to = DST / "labels" / to_split

    img_src = None
    for p in img_from.glob(stem + ".*"):
        img_src = p
        break
    if img_src is None:
        raise FileNotFoundError(f"Image yok: {from_split}/{stem}")

    lbl_src = lbl_from / f"{stem}.txt"
    if not lbl_src.exists():
        raise FileNotFoundError(f"Label yok: {from_split}/{stem}.txt")

    shutil.move(str(img_src), str(img_to / img_src.name))
    shutil.move(str(lbl_src), str(lbl_to / lbl_src.name))

def count_split(split: str):
    img_n = len(list((DST / "images" / split).glob("*.*")))
    lbl_n = len(list((DST / "labels" / split).glob("*.txt")))
    return img_n, lbl_n

def leak_check():
    train = set(list_stems(DST / "images" / "train"))
    val = set(list_stems(DST / "images" / "val"))
    test = set(list_stems(DST / "images" / "test"))
    return len(train & val), len(train & test), len(val & test)

def main():
    random.seed(SEED)
    if DST.exists():
        raise RuntimeError(f"DST zaten var, sil veya ad değiştir: {DST}")

    copy_tree()

    val_stems = list_stems(DST / "images" / "val")
    test_stems = list_stems(DST / "images" / "test")

    if len(val_stems) < TARGET_VAL:
        raise ValueError("Val zaten hedefin altında.")
    if len(test_stems) < TARGET_TEST:
        raise ValueError("Test zaten hedefin altında.")

    move_from_val = len(val_stems) - TARGET_VAL
    move_from_test = len(test_stems) - TARGET_TEST

    val_pick = random.sample(val_stems, move_from_val)
    test_pick = random.sample(test_stems, move_from_test)

    for stem in val_pick:
        move_pair(stem, "val", "train")
    for stem in test_pick:
        move_pair(stem, "test", "train")

    for split in ["train", "val", "test"]:
        img_n, lbl_n = count_split(split)
        print(f"{split}: images={img_n} labels={lbl_n}")

    a, b, c = leak_check()
    print("leak check train∩val:", a)
    print("leak check train∩test:", b)
    print("leak check val∩test:", c)
    print("Bitti:", DST)

if __name__ == "__main__":
    main()