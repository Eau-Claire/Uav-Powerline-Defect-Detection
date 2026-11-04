"""Streamlit UI for reviewing the targeted NMS audit pack."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "results/finding_nemo/targeted_audit_test_main_v1_nms50"
VERDICTS = [
    "model_error",
    "missing_gt",
    "duplicate_gt",
    "incorrect_gt_box",
    "incorrect_gt_class",
    "valid_unlabelled_object_but_out_of_scope",
    "unsure",
]


def load_rows(pack: Path) -> pd.DataFrame:
    source = pack / "selected_for_review.csv"
    frame = pd.read_csv(source, encoding="utf-8-sig", dtype=str).fillna("")
    progress = pack / "review_progress.csv"
    if progress.is_file():
        saved = pd.read_csv(progress, encoding="utf-8-sig", dtype=str).fillna("")
        saved = saved[["audit_class", "selection", "rank", "review_verdict", "review_note"]]
        frame = frame.drop(columns=["review_verdict", "review_note"], errors="ignore").merge(
            saved, on=["audit_class", "selection", "rank"], how="left"
        ).fillna("")
    return frame


def save_rows(pack: Path, frame: pd.DataFrame) -> None:
    columns = ["audit_class", "selection", "rank", "image", "review_verdict", "review_note"]
    frame[columns].to_csv(pack / "review_progress.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def preview_path(pack: Path, row: pd.Series) -> Path | None:
    folder = pack / "previews" / row["audit_class"] / row["selection"]
    matches = list(folder.glob(f"{int(row['rank']):03d}_*.jpg"))
    return matches[0] if matches else None


st.set_page_config(page_title="Targeted GT Audit", layout="wide")
st.title("Targeted audit — test_main_v1")
pack = Path(st.sidebar.text_input("Audit pack", str(DEFAULT_PACK))).resolve()
if not (pack / "selected_for_review.csv").is_file():
    st.error(f"Audit pack not found: {pack}")
    st.stop()

df = load_rows(pack)
audit_class = st.sidebar.selectbox("Class", ["insulator", "broken_strand"])
selection = st.sidebar.selectbox("Group", ["error", "tp_control", "all"])
review_filter = st.sidebar.selectbox("Status", ["unreviewed", "reviewed", "all"])
mask = df["audit_class"].eq(audit_class)
if selection != "all":
    mask &= df["selection"].eq(selection)
if review_filter == "unreviewed":
    mask &= df["review_verdict"].eq("")
elif review_filter == "reviewed":
    mask &= df["review_verdict"].ne("")
indices = df.index[mask].tolist()

reviewed = int(df["review_verdict"].ne("").sum())
st.sidebar.metric("Reviewed", f"{reviewed}/{len(df)}")
st.sidebar.progress(reviewed / max(1, len(df)))
if not indices:
    st.success("No images match this filter.")
    st.stop()

if "position" not in st.session_state:
    st.session_state.position = 0
st.session_state.position = min(st.session_state.position, len(indices) - 1)
position = st.sidebar.number_input("Position", 1, len(indices), st.session_state.position + 1) - 1
st.session_state.position = int(position)
idx = indices[st.session_state.position]
row = df.loc[idx]
preview = preview_path(pack, row)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"{row['audit_class']} · {row['selection']} · rank {row['rank']}")
    if preview:
        st.image(str(preview), use_container_width=True)
    else:
        st.error("Preview missing")
    st.caption(row["image"])

with right:
    fields = [
        "insulator_severity", "insulator_TP", "insulator_FP", "insulator_FN",
        "insulator_LOCALIZATION", "broken_strand_severity", "broken_strand_TP",
        "broken_strand_FP", "broken_strand_FN", "broken_strand_LOCALIZATION",
    ]
    st.dataframe(pd.DataFrame({"metric": fields, "value": [row.get(k, "") for k in fields]}), hide_index=True)
    current = row.get("review_verdict", "")
    verdict = st.selectbox("Verdict", [""] + VERDICTS, index=([""] + VERDICTS).index(current) if current in VERDICTS else 0)
    note = st.text_area("Note", value=row.get("review_note", ""), height=100)
    c1, c2, c3 = st.columns(3)
    if c1.button("← Previous", use_container_width=True):
        st.session_state.position = max(0, st.session_state.position - 1)
        st.rerun()
    if c2.button("Save", type="primary", use_container_width=True):
        df.at[idx, "review_verdict"] = verdict
        df.at[idx, "review_note"] = note
        save_rows(pack, df)
        st.toast("Saved")
    if c3.button("Save + Next", type="primary", use_container_width=True):
        df.at[idx, "review_verdict"] = verdict
        df.at[idx, "review_note"] = note
        save_rows(pack, df)
        st.session_state.position = min(len(indices) - 1, st.session_state.position + 1)
        st.rerun()

st.divider()
counts = df[df["review_verdict"].ne("")].groupby(["audit_class", "review_verdict"]).size().reset_index(name="count")
st.subheader("Review totals")
st.dataframe(counts, hide_index=True, use_container_width=True)
