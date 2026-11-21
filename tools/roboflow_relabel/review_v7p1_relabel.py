from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_annotation import detection


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/review_v7p1_seed85_relabel_181126"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
COLORS = {
    "insulator": "#00d4ff", "insulator_broken": "#ff3b30", "insulator_flashover": "#ff00ff",
    "bird_nest": "#34c759", "foreign_object": "#ff9500", "broken_strand": "#ffd60a",
}


st.set_page_config(page_title="V7.1 six-class box editor", layout="wide")
st.title("V7.1 · Seed85 relabel review")
st.caption("Sửa class và tọa độ pixel trực tiếp; có thể thêm/xóa box. Label nguồn không bị ghi đè.")


@st.cache_data
def manifest():
    return pd.read_csv(BUNDLE/"manifest.csv")


def decision_path(stem):
    return BUNDLE/"decisions"/f"{stem}.json"


def status(stem):
    p = decision_path(stem)
    return json.loads(p.read_text(encoding="utf-8")).get("status", "reviewed") if p.is_file() else "pending"


def read_boxes(path: Path, size):
    width, height = size; rows = []
    if not path.is_file(): return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        p = line.split()
        if len(p) != 5: continue
        cls = int(float(p[0])); x, y, w, h = map(float, p[1:])
        rows.append({"class": NAMES[cls], "x1": round((x-w/2)*width, 1), "y1": round((y-h/2)*height, 1),
                     "x2": round((x+w/2)*width, 1), "y2": round((y+h/2)*height, 1), "delete": False})
    return rows


def draw_boxes(image_path, boxes, before=None):
    image = Image.open(image_path).convert("RGB"); draw = ImageDraw.Draw(image)
    if before:
        for b in before:
            draw.rectangle((b["x1"], b["y1"], b["x2"], b["y2"]), outline="#8e8e93", width=2)
    for n, b in enumerate(boxes, 1):
        if bool(b.get("delete", False)): continue
        if any(b.get(k) is None or pd.isna(b.get(k)) for k in ("x1", "y1", "x2", "y2")):
            continue
        color = COLORS.get(str(b.get("class", "")), "white")
        xy = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
        draw.rectangle(xy, outline=color, width=4)
        label = f"{n} {b['class']}"
        draw.rectangle((xy[0], max(0, xy[1]-18), xy[0]+8*len(label), xy[1]), fill=color)
        draw.text((xy[0]+2, max(0, xy[1]-17)), label, fill="black")
    return image


def validate_and_save(rows, image_size, label_path):
    width, height = image_size; lines = []
    for index, row in enumerate(rows, 1):
        if bool(row.get("delete", False)): continue
        cls_name = str(row.get("class", ""))
        if cls_name not in NAMES: raise ValueError(f"Box {index}: class không hợp lệ")
        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1: raise ValueError(f"Box {index}: x2/y2 phải lớn hơn x1/y1")
        xc, yc, bw, bh = (x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height
        lines.append(f"{NAMES.index(cls_name)} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text(("\n".join(lines)+"\n") if lines else "", encoding="utf-8")
    return len(lines)


queue = manifest().copy()
queue["stem"] = queue["image"].map(lambda x: Path(x).stem)
queue["status"] = queue["stem"].map(status)
with st.sidebar:
    st.metric("Reviewed", int((queue.status != "pending").sum()), f"/ {len(queue)}")
    show = st.radio("Queue", ["pending", "all", "reviewed"], horizontal=True)
    keyword = st.text_input("Filter reason/class")
    if show == "pending": queue = queue[queue.status == "pending"]
    elif show == "reviewed": queue = queue[queue.status != "pending"]
    if keyword: queue = queue[queue.reasons.str.contains(keyword, case=False, na=False)]
    if queue.empty:
        st.success("Không còn ảnh trong filter này."); st.stop()
    options = queue.index.tolist()
    selected = st.selectbox("Image", options, format_func=lambda i: f"{int(queue.loc[i,'index']):03d} · {queue.loc[i,'status']} · {queue.loc[i,'image'][:45]}")

record = queue.loc[selected]; image_path = BUNDLE/"images"/record.image; stem = record.stem
with Image.open(image_path) as im: size = im.size
before = read_boxes(BUNDLE/"labels_before"/record.label, size)
reviewed_path = BUNDLE/"reviewed_labels"/record.label
source_path = reviewed_path if reviewed_path.is_file() else BUNDLE/"labels_seed85"/record.label
current = read_boxes(source_path, size)

st.info(record.reasons)
st.write(f"**{record.image}** · {size[0]}×{size[1]} · status: `{record.status}`")
st.caption("Chọn class trong thanh công cụ rồi kéo chuột để vẽ box. Click box để chọn, kéo góc để resize, kéo giữa để di chuyển; dùng nút xóa của widget để bỏ box.")

initial_bboxes = [[b["x1"], b["y1"], b["x2"]-b["x1"], b["y2"]-b["y1"]] for b in current]
initial_labels = [NAMES.index(b["class"]) for b in current]
canvas_value = detection(
    str(image_path), NAMES, bboxes=initial_bboxes, labels=initial_labels,
    height=760, width=1180, line_width=4, use_space=True, key=f"canvas_{stem}",
)
if canvas_value is None:
    canvas_rows = current
else:
    canvas_rows = []
    for item in canvas_value:
        x, y, w, h = item["bbox"]
        canvas_rows.append({"class": item["label"], "x1": x, "y1": y, "x2": x+w, "y2": y+h, "delete": False})

controls, old_view = st.columns([1, 1])
with controls:
    st.write(f"Box hiện tại trên canvas: **{len(canvas_rows)}**")
    note = st.text_area("Note", key=f"note_{stem}")
    c1, c2 = st.columns(2)
    if c1.button("Save canvas", type="primary", use_container_width=True):
        try:
            count = validate_and_save(canvas_rows, size, reviewed_path)
            decision_path(stem).write_text(json.dumps({"status":"edited", "boxes":count, "note":note}, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Đã lưu {count} box"); st.rerun()
        except Exception as exc: st.error(str(exc))
    if c2.button("Approve seed85", use_container_width=True):
        count = validate_and_save(current, size, reviewed_path)
        decision_path(stem).write_text(json.dumps({"status":"approved", "boxes":count, "note":note}, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success("Đã duyệt"); st.rerun()
with old_view:
    with st.expander("So sánh label v2 trước sửa"):
        st.image(draw_boxes(image_path, before), use_container_width=True)

with st.expander("Bảng tọa độ canvas (chỉ để kiểm tra)"):
    st.dataframe(pd.DataFrame(canvas_rows), use_container_width=True)
