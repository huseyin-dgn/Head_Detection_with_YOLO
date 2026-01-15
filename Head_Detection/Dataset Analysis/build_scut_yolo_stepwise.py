import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

PART_A_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Datasets\SCUT_HEAD_Part_A")
PART_B_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Datasets\SCUT_HEAD_Part_B")
OUT_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo")

IMG_DIR = "JPEGImages"
ANN_DIR = "Annotations"
SPLIT_DIR = Path("ImageSets") / "Main"

XML_LABEL_NAME = "person"
CLASS_ID = 0
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_ids(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Split dosyası bulunamadı: {txt_path}")
    ids = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        ids.append(s)
    return ids

def find_img_path(root: Path, image_id: str) -> Path:
    base = root / IMG_DIR
    for ext in IMG_EXTS:
        p = base / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Görsel bulunamadı: {base} | id={image_id}")

def find_xml_path(root: Path, image_id: str) -> Path:
    p = root / ANN_DIR / f"{image_id}.xml"
    if not p.exists():
        raise FileNotFoundError(f"XML bulunamadı: {p}")
    return p

def get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return int(w), int(h)

def safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default

def voc_to_yolo_lines(xml_path: Path, img_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w_img = h_img = 0.0
    size = root.find("size")
    if size is not None:
        w_img = safe_float((size.findtext("width") or "0"), 0.0)
        h_img = safe_float((size.findtext("height") or "0"), 0.0)

    if w_img <= 0 or h_img <= 0:
        w, h = get_image_size(img_path)
        w_img, h_img = float(w), float(h)

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name != XML_LABEL_NAME:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        xmin = safe_float(bb.findtext("xmin") or "0")
        ymin = safe_float(bb.findtext("ymin") or "0")
        xmax = safe_float(bb.findtext("xmax") or "0")
        ymax = safe_float(bb.findtext("ymax") or "0")

        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        xmin = max(0.0, min(xmin, w_img - 1))
        xmax = max(0.0, min(xmax, w_img - 1))
        ymin = max(0.0, min(ymin, h_img - 1))
        ymax = max(0.0, min(ymax, h_img - 1))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue

        cx = (xmin + xmax) / 2.0 / w_img
        cy = (ymin + ymax) / 2.0 / h_img
        bw_n = bw / w_img
        bh_n = bh / h_img

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw_n = min(max(bw_n, 0.0), 1.0)
        bh_n = min(max(bh_n, 0.0), 1.0)

        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")

    return lines

def build_split(split_name: str, ids_a: list[str], ids_b: list[str]):
    img_out = OUT_ROOT / "images" / split_name
    lbl_out = OUT_ROOT / "labels" / split_name
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    missing = []
    written = 0

    def process_one(root: Path, image_id: str):
        nonlocal written
        img_src = find_img_path(root, image_id)
        xml_src = find_xml_path(root, image_id)

        img_dst = img_out / img_src.name
        shutil.copy2(img_src, img_dst)

        lbl_dst = lbl_out / f"{image_id}.txt"
        try:
            yolo_lines = voc_to_yolo_lines(xml_src, img_src)
            lbl_dst.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        except Exception as e:
            lbl_dst.write_text("", encoding="utf-8")
            missing.append((image_id, str(e)))
        written += 1

    for image_id in ids_a:
        process_one(PART_A_ROOT, image_id)
    for image_id in ids_b:
        process_one(PART_B_ROOT, image_id)

    img_count = len(list(img_out.glob("*.*")))
    lbl_count = len(list(lbl_out.glob("*.txt")))

    print(f"\n=== SPLIT: {split_name.upper()} ===")
    print(f"Beklenen ID sayısı: {len(ids_a) + len(ids_b)}")
    print(f"İşlenen ID sayısı: {written}")
    print(f"images/{split_name}: {img_count}")
    print(f"labels/{split_name}: {lbl_count}")
    print(f"Label üretim hatası: {len(missing)}")

    if missing:
        print("İlk 10 hata örneği:")
        for mid, msg in missing[:10]:
            print(f"- {mid}: {msg}")

def main():
    a_splits = PART_A_ROOT / SPLIT_DIR
    b_splits = PART_B_ROOT / SPLIT_DIR

    build_split("train", read_ids(a_splits / "train.txt"), read_ids(b_splits / "train.txt"))
    build_split("val", read_ids(a_splits / "val.txt"), read_ids(b_splits / "val.txt"))
    build_split("test", read_ids(a_splits / "test.txt"), read_ids(b_splits / "test.txt"))

    train_names = set(p.stem for p in (OUT_ROOT / "images" / "train").glob("*.*"))
    val_names = set(p.stem for p in (OUT_ROOT / "images" / "val").glob("*.*"))
    test_names = set(p.stem for p in (OUT_ROOT / "images" / "test").glob("*.*"))

    print("\n=== OUTPUT LEAK CHECK ===")
    print("train ∩ val :", len(train_names & val_names))
    print("train ∩ test:", len(train_names & test_names))
    print("val ∩ test  :", len(val_names & test_names))
    print("\nBitti. Çıktı klasörü:", OUT_ROOT)

if __name__ == "__main__":
    main()