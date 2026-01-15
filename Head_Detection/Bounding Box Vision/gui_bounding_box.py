from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display, clear_output
import random

DATA_ROOT = Path(r"C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\scut_head_yolo_rebalanced")

def find_image_by_stem(img_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def read_yolo_label(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows
    for i, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            continue
        cid, xc, yc, w, h = parts
        rows.append((i, int(cid), float(xc), float(yc), float(w), float(h)))
    return rows

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return x1, y1, x2, y2

def get_stems(split: str):
    img_dir = DATA_ROOT / "images" / split
    stems = sorted([p.stem for p in img_dir.glob("*.*") if p.is_file()])
    return stems

def build_df(split: str, stem: str):
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels" / split
    img_path = find_image_by_stem(img_dir, stem)
    label_path = lbl_dir / f"{stem}.txt"

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    rows = read_yolo_label(label_path)
    df = pd.DataFrame(rows, columns=["idx", "class_id", "x_center", "y_center", "w", "h"])

    if df.empty:
        return img, img_w, img_h, df

    xyxy = df.apply(lambda r: yolo_to_xyxy(r["x_center"], r["y_center"], r["w"], r["h"], img_w, img_h), axis=1)
    df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(xyxy.tolist(), index=df.index)

    df["bbox_w_px"] = (df["x2"] - df["x1"]).round(2)
    df["bbox_h_px"] = (df["y2"] - df["y1"]).round(2)
    df["area_px2"] = (df["bbox_w_px"] * df["bbox_h_px"]).round(2)

    df["x1"] = df["x1"].round(2)
    df["y1"] = df["y1"].round(2)
    df["x2"] = df["x2"].round(2)
    df["y2"] = df["y2"].round(2)

    return img, img_w, img_h, df

def apply_filters_and_sort(df: pd.DataFrame, min_area, min_w, min_h, sort_mode, max_boxes):
    if df.empty:
        return df

    f = df.copy()
    f = f[(f["area_px2"] >= min_area) & (f["bbox_w_px"] >= min_w) & (f["bbox_h_px"] >= min_h)]

    if sort_mode == "Area ↓ (büyükten küçüğe)":
        f = f.sort_values("area_px2", ascending=False)
    elif sort_mode == "Area ↑ (küçükten büyüğe)":
        f = f.sort_values("area_px2", ascending=True)
    else:
        f = f.sort_values("idx", ascending=True)

    if max_boxes is not None and max_boxes > 0:
        f = f.head(max_boxes)

    return f

def draw_image_with_boxes(img, df_show: pd.DataFrame, title: str, show_indices=True, linewidth=2):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.imshow(img)

    for _, r in df_show.iterrows():
        rect = patches.Rectangle(
            (r["x1"], r["y1"]),
            r["x2"] - r["x1"],
            r["y2"] - r["y1"],
            fill=False,
            linewidth=linewidth
        )
        ax.add_patch(rect)

        if show_indices:
            ax.text(
                r["x1"],
                max(0, r["y1"] - 2),
                str(int(r["idx"])),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.6)
            )

    ax.set_title(title)
    ax.axis("off")
    plt.show()


split_dd = widgets.Dropdown(options=["train", "val", "test"], value="train", description="Split:")
idx_slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description="Index:")
prev_btn = widgets.Button(description="Prev", button_style="")
next_btn = widgets.Button(description="Next", button_style="")
rand_btn = widgets.Button(description="Random", button_style="")

search_txt = widgets.Text(value="", description="ID:", placeholder="Örn: PartB_00042")
go_btn = widgets.Button(description="Go", button_style="")

min_area = widgets.IntSlider(value=0, min=0, max=2000, step=10, description="Min area(px²):")
min_w = widgets.IntSlider(value=0, min=0, max=200, step=1, description="Min w(px):")
min_h = widgets.IntSlider(value=0, min=0, max=200, step=1, description="Min h(px):")
max_boxes = widgets.IntSlider(value=200, min=1, max=500, step=1, description="Max boxes:")
sort_mode = widgets.Dropdown(
    options=["Original (idx)", "Area ↓ (büyükten küçüğe)", "Area ↑ (küçükten büyüğe)"],
    value="Original (idx)",
    description="Sort:"
)

show_idx_chk = widgets.Checkbox(value=True, description="Bbox index göster")
lw_slider = widgets.IntSlider(value=2, min=1, max=6, step=1, description="LineWidth:")

stats_html = widgets.HTML(value="")
out = widgets.Output()

_state = {"stems": []}

def refresh_stems():
    stems = get_stems(split_dd.value)
    _state["stems"] = stems
    idx_slider.max = max(0, len(stems) - 1)
    if idx_slider.value > idx_slider.max:
        idx_slider.value = 0

def set_index(i):
    stems = _state["stems"]
    if not stems:
        return
    idx_slider.value = max(0, min(i, len(stems) - 1))

def current_stem():
    stems = _state["stems"]
    if not stems:
        return None
    return stems[idx_slider.value]

def update_stats(total_boxes, shown_boxes):
    if total_boxes == 0:
        ratio = 0.0
    else:
        ratio = shown_boxes / total_boxes * 100.0
    stats_html.value = (
        f"<b>Toplam bbox:</b> {total_boxes} &nbsp; | &nbsp; "
        f"<b>Gösterilen bbox:</b> {shown_boxes} &nbsp; | &nbsp; "
        f"<b>Filtre sonrası oran:</b> {ratio:.1f}%"
    )

def render():
    refresh_stems()
    stem = current_stem()
    if stem is None:
        return

    img, img_w, img_h, df = build_df(split_dd.value, stem)

    df_show = apply_filters_and_sort(
        df,
        min_area=min_area.value,
        min_w=min_w.value,
        min_h=min_h.value,
        sort_mode=sort_mode.value,
        max_boxes=max_boxes.value
    )

    title = f"{split_dd.value} | {stem} | {img_w}x{img_h}"
    update_stats(total_boxes=len(df), shown_boxes=len(df_show))

    with out:
        clear_output(wait=True)
        draw_image_with_boxes(
            img,
            df_show,
            title=title,
            show_indices=show_idx_chk.value,
            linewidth=lw_slider.value
        )
        if df_show.empty:
            display(pd.DataFrame(columns=["idx","class_id","x_center","y_center","w","h","x1","y1","x2","y2","bbox_w_px","bbox_h_px","area_px2"]))
        else:
            display(df_show[["idx","class_id","x_center","y_center","w","h","x1","y1","x2","y2","bbox_w_px","bbox_h_px","area_px2"]])

def on_change(_):
    render()

def on_prev(_):
    set_index(idx_slider.value - 1)

def on_next(_):
    set_index(idx_slider.value + 1)

def on_rand(_):
    stems = _state["stems"]
    if stems:
        set_index(random.randint(0, len(stems) - 1))

def on_go(_):
    refresh_stems()
    target = search_txt.value.strip()
    if not target:
        return
    stems = _state["stems"]
    if target in stems:
        set_index(stems.index(target))
    else:
        with out:
            print(f"Bulunamadı: {target}")

split_dd.observe(on_change, names="value")
idx_slider.observe(on_change, names="value")
min_area.observe(on_change, names="value")
min_w.observe(on_change, names="value")
min_h.observe(on_change, names="value")
max_boxes.observe(on_change, names="value")
sort_mode.observe(on_change, names="value")
show_idx_chk.observe(on_change, names="value")
lw_slider.observe(on_change, names="value")

prev_btn.on_click(on_prev)
next_btn.on_click(on_next)
rand_btn.on_click(on_rand)
go_btn.on_click(on_go)

refresh_stems()

top_row = widgets.HBox([split_dd, idx_slider, prev_btn, next_btn, rand_btn])
search_row = widgets.HBox([search_txt, go_btn, stats_html])
filter_row1 = widgets.HBox([min_area, sort_mode, max_boxes])
filter_row2 = widgets.HBox([min_w, min_h, show_idx_chk, lw_slider])

display(top_row)
display(search_row)
display(filter_row1)
display(filter_row2)
display(out)

render()