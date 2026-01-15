import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

FINAL_DIR = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\Final")
WEIGHTS = FINAL_DIR / "weights" / "best.pt"
OUT_DIR = FINAL_DIR / "outputs" / "streamlit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.39
IMGSZ = 832
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIOU_THR = 0.51
NMS_IOU = 0.50
MAX_DET = 300


def _diou_single_to_many(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b1[0], b1[1], b1[2], b1[3]

    xx1 = torch.maximum(x1, b2[:, 0])
    yy1 = torch.maximum(y1, b2[:, 1])
    xx2 = torch.minimum(x2, b2[:, 2])
    yy2 = torch.minimum(y2, b2[:, 3])

    inter_w = torch.clamp(xx2 - xx1, min=0)
    inter_h = torch.clamp(yy2 - yy1, min=0)
    inter = inter_w * inter_h

    area1 = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area2 = torch.clamp(b2[:, 2] - b2[:, 0], min=0) * torch.clamp(b2[:, 3] - b2[:, 1], min=0)
    union = area1 + area2 - inter + 1e-9
    iou = inter / union

    c1x = (x1 + x2) * 0.5
    c1y = (y1 + y2) * 0.5
    c2x = (b2[:, 0] + b2[:, 2]) * 0.5
    c2y = (b2[:, 1] + b2[:, 3]) * 0.5
    rho2 = (c1x - c2x) ** 2 + (c1y - c2y) ** 2

    ex1 = torch.minimum(x1, b2[:, 0])
    ey1 = torch.minimum(y1, b2[:, 1])
    ex2 = torch.maximum(x2, b2[:, 2])
    ey2 = torch.maximum(y2, b2[:, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-9

    return iou - (rho2 / c2)


def diou_nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, diou_thr: float = 0.51, max_det: int = 300):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        diou_vals = _diou_single_to_many(boxes[i], boxes[rest])
        order = rest[diou_vals <= diou_thr]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def draw_boxes(pil_img, boxes, scores, ids, show_index=True, show_conf=True, line_width=3):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for box, sc, _id in zip(boxes, scores, ids):
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([x1, y1, x2, y2], width=line_width)

        parts = []
        if show_index:
            parts.append(str(int(_id)))
        if show_conf:
            parts.append(f"{float(sc):.2f}")
        label = " | ".join(parts)

        if label:
            tx, ty = x1, max(0.0, y1 - 22.0)
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 3
            draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill="black")
            draw.text((tx + pad, ty + pad), label, fill="white", font=font)

    return img


st.set_page_config(page_title="Head Counting (Final)", layout="wide")
st.title("Head Counting — Final (PyTorch + DIoU-NMS)")

uploaded = st.file_uploader("Fotoğraf yükle", type=["jpg", "jpeg", "png", "bmp", "webp"])

meta_left, meta_right = st.columns(2)
with meta_left:
    st.write(f"weights: {WEIGHTS}")
with meta_right:
    st.write(f"device: {DEVICE} | conf: {CONF} | diou_thr: {DIOU_THR} | imgsz: {IMGSZ}")

if uploaded is not None:
    t0 = time.time()
    pil = Image.open(uploaded).convert("RGB")

    model = load_model(str(WEIGHTS))
    r = model.predict(
        source=pil,
        conf=CONF,
        iou=NMS_IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )[0]

    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes.xyxy) == 0:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Original")
            st.image(pil, use_container_width=True)
        with col2:
            st.subheader("Annotated")
            st.image(pil, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw boxes", 0)
        m2.metric("Kept count", 0)
        m3.metric("Count", 0)
        m4.metric("Latency (ms)", f"{(time.time() - t0)*1000:.1f}")

    else:
        boxes = r.boxes.xyxy.detach()
        scores = r.boxes.conf.detach()

        raw_count = int(scores.shape[0])

        keep = diou_nms_xyxy(boxes, scores, diou_thr=DIOU_THR, max_det=MAX_DET)

        boxes_k = boxes[keep].cpu().numpy().tolist()
        scores_k = scores[keep].cpu().numpy().tolist()

        kept_count = len(boxes_k)
        ids = list(range(1, kept_count + 1))

        rows = []
        for _id, (b, sc) in zip(ids, zip(boxes_k, scores_k)):
            x1, y1, x2, y2 = map(float, b)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            rows.append({
                "id": int(_id),
                "conf": float(sc),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h,
                "area": w * h,
                "cx": (x1 + x2) * 0.5,
                "cy": (y1 + y2) * 0.5,
            })

        df_boxes = pd.DataFrame(rows)

        csv_bytes = df_boxes.sort_values("conf", ascending=False).to_csv(index=False).encode("utf-8")

        annotated = draw_boxes(pil, boxes_k, scores_k, ids)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Original")
            st.image(pil, use_container_width=True)
        with col2:
            st.subheader("Annotated (id | conf)")
            st.image(annotated, use_container_width=True)

        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw boxes", raw_count)
        m2.metric("Kept count", kept_count)
        m3.metric("Count", kept_count)
        m4.metric("Latency (ms)", f"{(time.time() - t0)*1000:.1f}")

        with st.expander("BBox tablosu (id sabit, conf’a göre sıralı)", expanded=True):
            st.dataframe(df_boxes.sort_values("conf", ascending=False), use_container_width=True, height=360)

        st.download_button(
            "CSV indir (bbox_listesi)",
            data=csv_bytes,
            file_name="bbox_listesi.csv",
            mime="text/csv",
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"annotated_{stamp}.jpg"
        annotated.save(out_path, quality=95)
        st.write(f"saved: {out_path}")