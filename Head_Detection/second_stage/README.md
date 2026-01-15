# Head Counting Final Model

Preset:

- weights: best.pt
- conf: 0.39
- NMS: DIoU (diou_thr=0.51, nms_iou=0.50)
- imgsz: 832
- device: cuda:0

Run:
python scripts/run_final_infer.py

Outputs:

- outputs/predictions_txt
- outputs/reports (count_summary.csv, count_per_image.csv, worst_30.csv)
- outputs/nms_stats.csv
