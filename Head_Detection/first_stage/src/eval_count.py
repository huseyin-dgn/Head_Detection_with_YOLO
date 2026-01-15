from pathlib import Path
import pandas as pd
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_gt_label(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    return len(lines)


def _count_pred_txt(pred_path: Path) -> int:
    if not pred_path.exists():
        return 0
    lines = [ln.strip() for ln in pred_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    return len(lines)


def eval_count(
    weights: str,
    images_dir: str,
    labels_dir: str,
    report_dir: str,
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 832,
    device: str = "cuda:0",
    limit: int | None = None,
    pred_dir: str | None = None,
):
    weights = str(Path(weights).resolve())
    images_dir = Path(images_dir).resolve()
    labels_dir = Path(labels_dir).resolve()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    pred_dir_p = Path(pred_dir).resolve() if pred_dir is not None else None
    model = None if pred_dir_p is not None else YOLO(weights)

    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    imgs = sorted(imgs)
    if limit is not None:
        imgs = imgs[: int(limit)]

    rows = []
    for img_path in imgs:
        stem = img_path.stem
        gt = _count_gt_label(labels_dir / f"{stem}.txt")

        if pred_dir_p is not None:
            pred = _count_pred_txt(pred_dir_p / f"{stem}.txt")
        else:
            r = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=False,
                verbose=False,
            )[0]
            pred = int(r.boxes.shape[0]) if r.boxes is not None else 0

        rows.append(
            {
                "id": stem,
                "image": str(img_path),
                "gt_count": gt,
                "pred_count": pred,
                "signed_error": pred - gt,
                "abs_error": abs(pred - gt),
            }
        )

    df = pd.DataFrame(rows)
    mae = float(df["abs_error"].mean()) if len(df) else 0.0
    mape = float((df["abs_error"] / (df["gt_count"].replace(0, pd.NA))).dropna().mean()) if len(df) else 0.0
    rmse = float((df["signed_error"] ** 2).mean() ** 0.5) if len(df) else 0.0

    summary = pd.DataFrame(
        [
            {"metric": "MAE", "value": mae},
            {"metric": "RMSE", "value": rmse},
            {"metric": "MAPE(ignore gt=0)", "value": mape},
            {"metric": "N_images", "value": len(df)},
            {"metric": "GT_total", "value": int(df["gt_count"].sum()) if len(df) else 0},
            {"metric": "Pred_total", "value": int(df["pred_count"].sum()) if len(df) else 0},
        ]
    )

    df.to_csv(report_dir / "count_per_image.csv", index=False, encoding="utf-8")
    summary.to_csv(report_dir / "count_summary.csv", index=False, encoding="utf-8")

    worst = df.sort_values("abs_error", ascending=False).head(30) if len(df) else df
    worst.to_csv(report_dir / "worst_30.csv", index=False, encoding="utf-8")

    return df, summary