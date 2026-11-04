"""Streamlit review UI for val_clean/test cross-label proposals."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from core import Box, read_yolo_boxes


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "val_clean": {
        "review": ROOT / "artifacts/roboflow_relabel/val_clean_serverless/review_items.json",
        "decisions": ROOT / "artifacts/roboflow_relabel/val_clean_review_decisions.json",
        "labels": ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined/labels/val_clean",
    },
    "test": {
        "review": ROOT / "artifacts/roboflow_relabel/test_serverless/review_items.json",
        "decisions": ROOT / "artifacts/roboflow_relabel/test_review_decisions.json",
        "labels": ROOT / "data/working/test/labels/test",
    },
}
COLORS = {0: "#2f80ed", 1: "#eb5757", 2: "#f2994a", 3: "#9b51e0", 4: "#56ccf2", 5: "#f2c94c"}


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def save_decision(path: Path, target: str, total: int, item: dict, decision: str, note: str) -> None:
    document = load_json(path, {"package": f"{target}_cross_label_review", "target": target,
                                "total_items": total, "decisions": {}})
    document["total_items"] = total
    document["exported_at"] = datetime.now(timezone.utc).isoformat()
    document["decisions"][item["key"]] = {
        "decision": decision, "image": Path(item["image"]).name, "kind": item["kind"],
        "class_name": item["box"].get("class_name"), "source": item["source"],
        "source_class": item.get("source_class"), "box": item["box"], "note": note,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def render(item: dict, label_dir: Path) -> tuple[Image.Image, Image.Image]:
    image_path = Path(item["image"])
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    label = label_dir / f"{image_path.stem}.txt"
    for old in read_yolo_boxes(label, image.size):
        draw.rectangle((old.x1, old.y1, old.x2, old.y2), outline=COLORS.get(old.class_id, "white"), width=3)
    box = Box(**item["box"])
    draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#00ff66", width=7)
    text = f"PROPOSED {box.class_name} {box.confidence or 0:.2f} | {item['source']}:{item.get('source_class', '')}"
    text_box = draw.textbbox((max(0, box.x1), max(0, box.y1 - 25)), text, font=font, stroke_width=2)
    draw.rectangle(text_box, fill="black")
    draw.text((max(0, box.x1), max(0, box.y1 - 25)), text, fill="#00ff66", font=font)
    margin_x = max(120, box.width * 1.3)
    margin_y = max(120, box.height * 1.3)
    crop_box = (
        max(0, int(box.x1 - margin_x)), max(0, int(box.y1 - margin_y)),
        min(image.width, int(box.x2 + margin_x)), min(image.height, int(box.y2 + margin_y)),
    )
    return image, image.crop(crop_box)


st.set_page_config(page_title="Finding Nemo label review", layout="wide")
st.title("Val/Test cross-label review")
st.caption("Green = proposed box · other colors = existing labels. Decisions are saved separately; sources are read-only.")

val_package = load_json(TARGETS["val_clean"]["review"], {"items": []})
val_saved = load_json(TARGETS["val_clean"]["decisions"], {"decisions": {}}).get("decisions", {})
test_package = load_json(TARGETS["test"]["review"], {"items": []})
test_saved = load_json(TARGETS["test"]["decisions"], {"decisions": {}}).get("decisions", {})
default_target = 1 if len(val_saved) == len(val_package["items"]) and len(test_saved) < len(test_package["items"]) else 0
if default_target == 1:
    st.warning(f"val_clean complete ({len(val_saved)}/{len(val_package['items'])}). Test still pending ({len(test_saved)}/{len(test_package['items'])}).")

with st.sidebar:
    target = st.radio("Dataset", list(TARGETS), index=default_target, horizontal=True,
                      key="dataset_selector_v2")
    config = TARGETS[target]
    package = load_json(config["review"], {"items": []})
    items = package["items"]
    decision_doc = load_json(config["decisions"], {"decisions": {}})
    decisions = decision_doc.get("decisions", {})
    st.progress(len(decisions) / max(1, len(items)))
    st.write(f"Reviewed **{len(decisions):,}/{len(items):,}**")
    status_filter = st.selectbox("Show", ["pending", "all", "accept", "reject", "unsure"])
    source_filter = st.selectbox("Source", ["all", "YOLO26m", "Roboflow"])
    class_values = sorted({item["box"].get("class_name", "") for item in items})
    class_filter = st.selectbox("Class", ["all", *class_values])

filtered = []
for item in items:
    saved = decisions.get(item["key"], {}).get("decision")
    if status_filter == "pending" and saved is not None:
        continue
    if status_filter not in {"pending", "all"} and saved != status_filter:
        continue
    if source_filter != "all" and item["source"] != source_filter:
        continue
    if class_filter != "all" and item["box"].get("class_name") != class_filter:
        continue
    filtered.append(item)

if not filtered:
    st.success("No items match this filter.")
    st.stop()

index_key = f"review_index_{target}_{status_filter}_{source_filter}_{class_filter}"
if index_key not in st.session_state:
    st.session_state[index_key] = 0
st.session_state[index_key] = min(st.session_state[index_key], len(filtered) - 1)
index = st.number_input("Item", min_value=1, max_value=len(filtered),
                        value=st.session_state[index_key] + 1, step=1,
                        label_visibility="collapsed") - 1
st.session_state[index_key] = index
item = filtered[index]
current = decisions.get(item["key"], {})

box = item["box"]
st.markdown(
    f"### {index + 1:,}/{len(filtered):,} · `{box.get('class_name')}` · "
    f"**{float(box.get('confidence') or 0):.2f}** · {item['source']} / {item.get('source_class', '')}"
)
note = st.text_input("Optional note", value=current.get("note", ""),
                     key=f"note_{item['key']}", placeholder="Ghi chú nếu cần…")
actions = st.columns([2, 2, 2, 1, 1])
choice = None
if actions[0].button("✅ ACCEPT + NEXT", width="stretch", type="primary"):
    choice = "accept"
if actions[1].button("❌ REJECT + NEXT", width="stretch"):
    choice = "reject"
if actions[2].button("❓ UNSURE + NEXT", width="stretch"):
    choice = "unsure"
if actions[3].button("←", width="stretch", help="Previous"):
    st.session_state[index_key] = max(0, index - 1)
    st.rerun()
if actions[4].button("→", width="stretch", help="Skip / next"):
    st.session_state[index_key] = min(len(filtered) - 1, index + 1)
    st.rerun()
if choice:
    save_decision(config["decisions"], target, len(items), item, choice, note)
    if status_filter != "pending":
        st.session_state[index_key] = min(index + 1, len(filtered) - 1)
    st.rerun()

full_image, crop_image = render(item, config["labels"])
focus, context = st.columns([2, 1])
with focus:
    st.caption("🔍 Proposal crop")
    st.image(crop_image, width="stretch")
with context:
    st.caption("🗺️ Full image")
    st.image(full_image, width="stretch")

if current:
    st.info(f"Saved: **{current.get('decision')}**")
with st.expander("Proposal details"):
    st.code(item["key"])
    st.json({"kind": item["kind"], "source": item["source"],
             "source_class": item.get("source_class"), "proposal": item["box"]})

st.caption(f"Decision file: {config['decisions']}")
