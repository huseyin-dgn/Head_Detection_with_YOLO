from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import torch
from ultralytics import YOLO
from src.nms import apply_nms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _save_yolo_txt_with_conf(
    txt_path: Path,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    img_w: int,
    img_h: int,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    if boxes_xyxy.numel() == 0:
        txt_path.write_text("", encoding="utf-8")
        return

    b = boxes_xyxy.clone()
    b[:, 0].clamp_(0, img_w)
    b[:, 2].clamp_(0, img_w)
    b[:, 1].clamp_(0, img_h)
    b[:, 3].clamp_(0, img_h)

    cx = (b[:, 0] + b[:, 2]) * 0.5 / float(img_w)
    cy = (b[:, 1] + b[:, 3]) * 0.5 / float(img_h)
    w  = (b[:, 2] - b[:, 0]).clamp(min=0) / float(img_w)
    h  = (b[:, 3] - b[:, 1]).clamp(min=0) / float(img_h)

    lines = []
    for i in range(b.size(0)):
        c = int(cls_ids[i].item()) if cls_ids.numel() else 0
        lines.append(
            f"{c} {cx[i].item():.6f} {cy[i].item():.6f} {w[i].item():.6f} {h[i].item():.6f} {scores[i].item():.6f}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def predict_folder(
    weights: str,
    images_dir: str,
    out_dir: str,
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 832,
    device: str = "cuda:0",
    nms_type: str = "standard",
    nms_iou: float = 0.50,
    diou_thr: float = 0.60,
    soft_sigma: float = 0.5,
    soft_score_thr: float = 0.001,
    class_agnostic: bool = True,
    max_det: int = 300,
    stats_path: Optional[str] = None,
) -> Tuple[dict, List[dict]]:
    weights = str(Path(weights).resolve())
    images_dir_p = Path(images_dir).resolve()
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    img_paths = _list_images(images_dir_p)

    total_imgs = 0
    total_raw = 0
    total_kept = 0
    total_removed = 0
    per_image = []

    for img_path in img_paths:
        r = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.90,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        total_imgs += 1

        if r.boxes is None or len(r.boxes) == 0:
            (out_dir_p / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            per_image.append({"id": img_path.stem, "raw": 0, "kept": 0, "removed": 0})
            continue

        boxes = r.boxes.xyxy.detach().cpu()
        scores = r.boxes.conf.detach().cpu()
        cls_ids = r.boxes.cls.detach().cpu()

        raw_n = int(scores.numel())

        keep = apply_nms(
            boxes=boxes,
            scores=scores,
            cls=cls_ids,
            nms_type=nms_type,
            iou_thr=nms_iou,
            diou_thr=diou_thr,
            soft_sigma=soft_sigma,
            soft_score_thr=soft_score_thr,
            class_agnostic=class_agnostic,
            max_det=max_det,
            min_score=conf,
        )

        kept_n = int(keep.numel()) if keep is not None else 0
        removed_n = raw_n - kept_n

        total_raw += raw_n
        total_kept += kept_n
        total_removed += removed_n if removed_n > 0 else 0

        per_image.append({"id": img_path.stem, "raw": raw_n, "kept": kept_n, "removed": removed_n})

        if kept_n > 0:
            boxes_k = boxes[keep]
            scores_k = scores[keep]
            cls_k = cls_ids[keep]
        else:
            boxes_k = boxes[:0]
            scores_k = scores[:0]
            cls_k = cls_ids[:0]

        img_h, img_w = r.orig_shape
        _save_yolo_txt_with_conf(
            out_dir_p / f"{img_path.stem}.txt",
            boxes_k,
            scores_k,
            cls_k,
            img_w=img_w,
            img_h=img_h,
        )

    stats = {
        "n_images": total_imgs,
        "raw_total": total_raw,
        "kept_total": total_kept,
        "removed_total": total_removed,
        "avg_raw_per_image": (total_raw / total_imgs) if total_imgs else 0.0,
        "avg_kept_per_image": (total_kept / total_imgs) if total_imgs else 0.0,
        "avg_removed_per_image": (total_removed / total_imgs) if total_imgs else 0.0,
        "nms_type": nms_type,
        "conf": conf,
        "diou_thr": diou_thr,
        "nms_iou": nms_iou,
        "imgsz": imgsz,
        "device": device,
    }

    if stats_path:
        sp = Path(stats_path).resolve()
        sp.parent.mkdir(parents=True, exist_ok=True)
        lines = ["key,value"] + [f"{k},{v}" for k, v in stats.items()]
        sp.write_text("\n".join(lines), encoding="utf-8")

    return stats, per_image