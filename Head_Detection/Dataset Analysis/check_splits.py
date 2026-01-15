from pathlib import Path

def read_ids(txt_path: Path) -> set[str]:
    ids = set()
    if not txt_path.exists():
        raise FileNotFoundError(f"Bulunamadı: {txt_path}")
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        ids.add(s)
    return ids

def report_intersection(a_name: str, a: set[str], b_name: str, b: set[str], limit=30):
    inter = sorted(a & b)
    print(f"{a_name} ∩ {b_name}: {len(inter)}")
    if inter:
        print("  Örnek çakışan ID'ler:", inter[:limit])

def check_one_split(split_dir: Path, tag: str):
    print(f" {tag} ")
    train = read_ids(split_dir / "train.txt")
    val   = read_ids(split_dir / "val.txt")
    test  = read_ids(split_dir / "test.txt")

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

    report_intersection("train", train, "val", val)
    report_intersection("train", train, "test", test)
    report_intersection("val", val, "test", test)

    union = train | val | test
    print(f"union(train,val,test): {len(union)}")

if __name__ == "__main__":
    PART_A_SPLIT_DIR = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Datasets\SCUT_HEAD_Part_A")
    PART_B_SPLIT_DIR = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Datasets\SCUT_HEAD_Part_B")

    check_one_split(PART_A_SPLIT_DIR, "Part A")
    check_one_split(PART_B_SPLIT_DIR, "Part B")