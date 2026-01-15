# app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
from ultralytics import YOLO


# =========================
# PATHS
# =========================
DEFAULT_MODEL_PATH = r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\third_stage\runs\best_finetune_img896_ep30.pt"
DEFAULT_CONFIG_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Final\configs"


DEFAULT_LABELS_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo_rebalanced\labels\val"


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

def list_json_configs(config_dir: Path):
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.json"))

def read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def choose_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"

def predict_one(model: YOLO, img_pil: Image.Image, conf: float, iou: float, imgsz: int, device: str):
    img = np.array(img_pil.convert("RGB"))
    r = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    return r

def results_to_df(r) -> pd.DataFrame:
    if r.boxes is None or len(r.boxes) == 0:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "conf"])
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    df = pd.DataFrame(xyxy, columns=["x1", "y1", "x2", "y2"])
    df["conf"] = conf
    return df

def render_annotated(r, show_labels: bool, show_conf: bool):
    arr = r.plot(labels=show_labels, conf=show_conf)
    return Image.fromarray(arr)

def find_label_file(uploaded_filename: str, labels_dir: Path) -> Path | None:
    """
    uploaded_filename: örn 'IMG_000123.jpg'
    labels_dir: .../labels/val veya .../labels/test
    Aranan: IMG_000123.txt
    """
    if not labels_dir.exists():
        return None
    stem = Path(uploaded_filename).stem
    cand = labels_dir / f"{stem}.txt"
    return cand if cand.exists() else None

def count_gt_from_yolo_label(label_path: Path) -> int:
    """
    YOLO label format: her satır 1 bbox
    boş satırları ignore eder.
    """
    if label_path is None or not label_path.exists():
        return -1
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    return len(lines)


# =========================
# UI
# =========================
st.set_page_config(page_title="Head Counting – Final Demo", layout="wide")
st.title("Head Counting – Final Model Demo")

AUTO_DEVICE = choose_device()

with st.sidebar:
    st.header("Model / Config")

    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    config_dir = Path(st.text_input("Config folder (JSON)", value=DEFAULT_CONFIG_DIR))

    st.divider()
    st.subheader("GT Label Kaynağı")
    st.caption("GT count için aynı isimli YOLO label .txt dosyası gerekir.")
    labels_dir = Path(st.text_input("Labels folder (e.g., labels/val)", value=DEFAULT_LABELS_DIR))

    st.divider()
    st.subheader("Compute")
    st.write("torch.cuda.is_available():", torch.cuda.is_available())
    st.write("torch.cuda.device_count():", torch.cuda.device_count())
    st.write("Auto device:", AUTO_DEVICE)

    if AUTO_DEVICE.startswith("cuda"):
        device = st.selectbox("Device", ["cuda:0", "cpu"], index=0)
    else:
        st.warning("CUDA yok → CPU ile koşacak (yavaş ama çalışır).")
        device = "cpu"

    st.divider()
    st.subheader("Render")
    show_labels = st.checkbox("Label yazsın (HEAD)", value=False)  # default kapalı
    show_conf = st.checkbox("Confidence yazsın", value=True)

configs = list_json_configs(config_dir)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Girdi")
    uploaded = st.file_uploader("Foto yükle (.jpg/.png/.webp)", type=["jpg", "jpeg", "png", "webp"])

    st.subheader("Inference Parametreleri")
    use_json = st.checkbox("JSON config kullan", value=True if configs else False)

    if use_json and configs:
        selected = st.selectbox("Config JSON", configs, format_func=lambda p: p.name)
        cfg = read_json(selected)

        conf = float(cfg.get("conf", 0.34))
        imgsz = int(cfg.get("imgsz", 896))
        iou = float(cfg.get("nms_iou", cfg.get("diou_thr", 0.45)))

        st.write("Seçilen config:")
        st.json(cfg, expanded=False)
    else:
        conf = st.slider("conf", 0.05, 0.90, 0.34, 0.01)
        iou = st.slider("iou", 0.10, 0.90, 0.45, 0.01)
        imgsz = st.selectbox("imgsz", [640, 736, 832, 896, 960], index=3)

    run = st.button("Çalıştır", type="primary", use_container_width=True)

with right:
    st.subheader("Çıktı")

    if uploaded is None:
        st.info("Soldan foto yükle, sonra Çalıştır.")
        st.stop()

    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Input", use_container_width=True)

    if run:
        # ---- GT count (label varsa) ----
        label_path = find_label_file(uploaded.name, labels_dir)
        gt_count = count_gt_from_yolo_label(label_path)  # -1 ise yok demek

        # ---- Prediction ----
        with st.spinner(f"Inference running on: {device}"):
            model = load_model(model_path)
            r = predict_one(model=model, img_pil=img_pil, conf=conf, iou=iou, imgsz=imgsz, device=device)

        df = results_to_df(r)
        pred_count = int(len(df))

        diff = None if gt_count < 0 else (pred_count - gt_count)

        # ---- Metrics ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pred Count", pred_count)
        if gt_count >= 0:
            c2.metric("GT Count (label)", gt_count)
            c3.metric("Diff (Pred−GT)", diff)
        else:
            c2.metric("GT Count (label)", "N/A")
            c3.metric("Diff (Pred−GT)", "N/A")
        c4.metric("imgsz", str(imgsz))

        st.caption(f"conf={conf:.2f} | iou={iou:.2f} | device={device}")
        if gt_count < 0:
            st.warning("GT label bulunamadı. Labels folder doğru mu ve dosya adı görselle birebir aynı mı?")

        if label_path is not None and label_path.exists():
            st.caption(f"GT label file: {str(label_path)}")

        st.divider()

        # ---- Comparison table ----
        st.subheader("GT vs Pred Karşılaştırma")
        comp = pd.DataFrame(
            [{
                "image": uploaded.name,
                "GT_count": None if gt_count < 0 else gt_count,
                "Pred_count": pred_count,
                "Diff(Pred-GT)": None if gt_count < 0 else diff,
            }]
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.divider()

        # ---- Annotated ----
        annotated = render_annotated(r, show_labels=show_labels, show_conf=show_conf)
        st.image(annotated, caption="Predictions (annotated)", use_container_width=True)

        st.divider()

        # ---- Detections ----
        st.subheader("Detections (bbox + conf)")
        if df.empty:
            st.warning("Bu görüntüde detection yok.")
        else:
            view = df.copy()
            for c in ["x1", "y1", "x2", "y2", "conf"]:
                view[c] = view[c].astype(float).round(2)
            st.dataframe(view, use_container_width=True, hide_index=True)

            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button("Detections CSV indir", data=csv, file_name="detections.csv", mime="text/csv")
