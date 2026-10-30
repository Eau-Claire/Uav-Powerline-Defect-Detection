# Continue the v3 cross-label audit on another computer

Dataset: v6p5 v3, 17,903 train / 2,272 val / 2,271 val_clean.
Only train labels are supplemented. Outputs are unreviewed pseudo-labels.

Download the transfer archive:

```sh
camber stash cp stash://bunpmc/projects/yolov11sdi/transfers/cross_v3_resume_v1.tar cross_v3_resume_v1.tar
tar -xf cross_v3_resume_v1.tar
```

Run inside the extracted folder, or extract it into the Git checkout. The bundle
contains runnable code, the original dataset ZIP, YOLO26m weights, and complete
checkpoint records. It contains no API keys or virtual environments.

Create a Python 3.10+ environment and install:

```sh
python -m venv .venv-roboflow
# Windows: .venv-roboflow\Scripts\activate
# Linux: source .venv-roboflow/bin/activate
python -m pip install -r tools/roboflow_relabel/requirements-handoff.txt
```

For GPU inference, install the matching CUDA-enabled PyTorch build for the host.
Set `CROSS_DEVICE=0`; the default is CPU.

Create `.env` in the extracted root using your own Roboflow key:

```dotenv
ROBOFLOW_API_KEY=YOUR_KEY
ROBOFLOW_API_URL=http://localhost:9001
ROBOFLOW_WORKSPACE=my-workspace-j75em
ROBOFLOW_WORKFLOW_ID=evn-object-detection
```

Start Docker Desktop / Docker Engine, then:

```sh
inference server start --env-file .env --image roboflow/roboflow-inference-server-cpu:latest
python -u tools/roboflow_relabel/run_v3.py
```

The runner extracts the dataset once, skips completed images, continues Roboflow,
and exports `data/working/v6p5_v3_cross_labeled/data.yaml` plus
`data_val_clean.yaml`. Detailed changes, conflicts, and summary are saved under
`artifacts/roboflow_relabel/cross_v3/`.

Mapping: Insulator → insulator (0); damage_insulator, critical, CenterDamage,
EdgeSmallDamage, EdgeBigDamage, cautious, low → insulator_broken (1).
YOLO class names map directly to the original six dataset classes.

Confidence ≥0.6 is required for additions. Same-class overlap IoU ≥0.3 is skipped;
different-class overlap IoU ≥0.5 goes to review (except context insulator vs
insulator_broken, which may overlap). Existing labels are preserved.
