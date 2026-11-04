from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from core import (
    Box,
    default_output_dir,
    draw_review,
    export_staged_labels,
    environment_defaults,
    load_jsonl,
    read_yolo_boxes,
    run_workflow_batch,
    save_decision,
    scan_candidates,
)


load_dotenv()
st.set_page_config(page_title="Roboflow Insulator Box Review", layout="wide")
st.title("Roboflow insulator_broken box review")
st.caption("Red = class 1 insulator_broken · Blue = existing class 0 insulator · Green = proposed class 0 insulator")


def latest_proposals() -> str:
    paths = list(Path("artifacts/roboflow_relabel").glob("*/proposals.jsonl"))
    return str(max(paths, key=lambda path: path.stat().st_mtime)) if paths else ""

defaults = environment_defaults()
with st.sidebar:
    dataset_root = Path(st.text_input("Dataset root", "data/cache/base_extract/curated_v6"))
    split = st.selectbox("Split", ["train", "val"], index=0)
    api_url = st.text_input("Inference server", defaults["api_url"])
    workspace_name = st.text_input("Workspace", defaults["workspace_name"])
    workflow_id = st.text_input("Workflow", defaults["workflow_id"])
    allowed_classes_text = st.text_input("Full-insulator class names", defaults["allowed_classes"])
    min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.50, 0.05)
    min_area_ratio = st.number_input("Minimum area expansion", 1.0, 100.0, 1.10, 0.10)
    max_area_ratio = st.number_input("Maximum area expansion", 1.0, 100.0, 20.0, 0.50)
    dry_run_size = st.number_input("Dry-run images", 1, 100, 20)

if st.button("Scan class-1 candidates"):
    with st.spinner("Scanning labels..."):
        st.session_state["candidates"] = scan_candidates(dataset_root, split, class_id=1)

candidates = st.session_state.get("candidates", [])
if candidates:
    st.success(f"Found {len(candidates):,} images containing insulator_broken")

api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
if not api_key:
    st.warning("ROBOFLOW_API_KEY is missing. Add it to .env, then restart Streamlit.")

if st.button("Run safe dry-run", disabled=not candidates or not api_key):
    output_dir = default_output_dir()
    with st.spinner(f"Running first {dry_run_size} candidate images..."):
        proposals_path = run_workflow_batch(
            candidates[: int(dry_run_size)],
            output_dir,
            api_url,
            api_key,
            workspace_name,
            workflow_id,
            {item.strip() for item in allowed_classes_text.split(",")},
            min_confidence,
            min_area_ratio,
            max_area_ratio,
            resume=False,
            workers=1,
        )
    st.session_state["proposals_path"] = str(proposals_path)
    st.success(f"Saved proposals to {proposals_path}")

proposal_input = st.text_input(
    "Proposal JSONL",
    st.session_state.get("proposals_path", latest_proposals()),
    placeholder="artifacts/roboflow_relabel/.../proposals.jsonl",
)
records = load_jsonl(Path(proposal_input)) if proposal_input else []
if records:
    proposal_path = Path(proposal_input)
    decisions_path = proposal_path.parent / "decisions.csv"
    index = st.number_input("Review image", 1, len(records), 1) - 1
    record = records[int(index)]
    image_path = Path(record["image_path"])
    workflow_boxes = [Box(**item) for item in record.get("predictions", [])]
    with Image.open(image_path) as source:
        current_boxes = read_yolo_boxes(Path(record["label_path"]), source.size)
    preview = draw_review(
        image_path,
        current_boxes,
        record.get("proposals"),
        record.get("additions"),
    )
    left, right = st.columns([2, 1])
    with left:
        st.image(preview, caption=image_path.name, use_container_width=True)
    with right:
        statuses = [item["status"] for item in record.get("additions", record.get("proposals", []))]
        st.write({"proposal_statuses": statuses, "workflow_predictions": len(workflow_boxes)})
        decision = st.radio("Decision", ["reject", "accept", "manual"], horizontal=True)
        note = st.text_area("Review note")
        if st.button("Save decision"):
            save_decision(decisions_path, str(image_path), decision, note)
            st.success(f"Saved to {decisions_path}")

    if st.button("Export class-0 additions to staging labels"):
        staging_dir = proposal_path.parent / "staged_labels"
        summary = export_staged_labels(records, staging_dir)
        st.success(f"Staged only; source unchanged. {summary} → {staging_dir}")

