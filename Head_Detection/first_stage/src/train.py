from pathlib import Path
import yaml
from ultralytics import YOLO

def load_cfg(cfg_path: str) -> dict:
    p = Path(cfg_path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return cfg

def ensure_abs(path_like: str, base: Path) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())

def train(cfg_path: str):
    cfg_path = Path(cfg_path).resolve()
    base = cfg_path.parent.parent

    cfg = load_cfg(str(cfg_path))

    data_yaml = Path(cfg["dataset_yaml"]).resolve()
    project_dir = ensure_abs(cfg.get("project_dir", "outputs/runs"), base)
    name = cfg.get("run_name", "baseline")

    model = YOLO(cfg.get("model", "yolov8n.pt"))

    results = model.train(
        data=str(data_yaml),
        imgsz=int(cfg.get("imgsz", 640)),
        epochs=int(cfg.get("epochs", 50)),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 4)),
        seed=int(cfg.get("seed", 42)),
        project=project_dir,
        name=name,
        patience=int(cfg.get("patience", 20)),
        save=bool(cfg.get("save", True)),
        exist_ok=bool(cfg.get("exist_ok", True)),
        pretrained=bool(cfg.get("pretrained", True)),
        amp=bool(cfg.get("amp", True)),
        optimizer=str(cfg.get("optimizer", "auto")),
        lr0=float(cfg.get("lr0", 0.01)),
        lrf=float(cfg.get("lrf", 0.01)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),
        cos_lr=bool(cfg.get("cos_lr", False)),
    )

    run_dir = Path(project_dir) / name
    best_pt = run_dir / "weights" / "best.pt"

    models_dir = (base / "models" / name).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if best_pt.exists():
        dst = models_dir / "best.pt"
        dst.write_bytes(best_pt.read_bytes())

    return str(run_dir)

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/train_baseline.yaml"
    out = train(cfg)
    print(out)