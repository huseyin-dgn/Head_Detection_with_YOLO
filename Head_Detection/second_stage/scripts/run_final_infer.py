import json
from pathlib import Path
from src.predict import predict_folder
from src.eval_count import eval_count

def main():
    here = Path(__file__).resolve().parent
    root = here.parent

    cfg_path = root / "config" / "final_preset.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    weights = (root / cfg["weights"]).resolve()

    dataset_root = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo_rebalanced")
    test_images = dataset_root / "images" / "test"
    test_labels = dataset_root / "labels" / "test"

    out_root = root / "outputs"
    pred_out = out_root / "predictions_txt"
    rep_out = out_root / "reports"
    stats_csv = out_root / "nms_stats.csv"

    out_root.mkdir(parents=True, exist_ok=True)
    pred_out.mkdir(parents=True, exist_ok=True)
    rep_out.mkdir(parents=True, exist_ok=True)

    stats, _ = predict_folder(
        weights=str(weights),
        images_dir=str(test_images),
        out_dir=str(pred_out),
        conf=float(cfg["conf"]),
        iou=0.6,
        imgsz=int(cfg["imgsz"]),
        device=str(cfg["device"]),
        nms_type=str(cfg["nms_type"]),
        diou_thr=float(cfg["diou_thr"]),
        nms_iou=float(cfg["nms_iou"]),
        class_agnostic=bool(cfg["class_agnostic"]),
        max_det=int(cfg["max_det"]),
        stats_path=str(stats_csv),
    )

    df, summary = eval_count(
        weights=str(weights),
        images_dir=str(test_images),
        labels_dir=str(test_labels),
        report_dir=str(rep_out),
        conf=float(cfg["conf"]),
        iou=0.6,
        imgsz=int(cfg["imgsz"]),
        device=str(cfg["device"]),
        pred_dir=str(pred_out),
    )

    print("\n=== FINAL DONE ===")
    print("stats:", stats)
    print(summary)


if __name__ == "__main__":
    main()