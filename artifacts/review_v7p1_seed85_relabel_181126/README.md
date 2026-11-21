# V7.1 seed85 relabel review bundle

Queue: 474 unique train images.

- `labels_before`: labels from v2 before repair
- `labels_seed85`: conservative repaired labels proposed for v3
- `reviewed_labels`: reviewer-approved/edited labels
- `decisions`: one JSON decision per reviewed image
- `manifest.csv`: queue and reasons

Run: `.venv-roboflow\Scripts\python.exe -m streamlit run tools/roboflow_relabel/review_v7p1_relabel.py`
