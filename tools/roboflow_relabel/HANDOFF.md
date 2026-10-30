# Run or resume the v3 cross-label audit

Dataset: v6p5 v3, 17,903 train / 2,272 val / 2,271 val_clean. Only TRAIN labels
are supplemented. Existing labels are preserved. Every output is an unreviewed
pseudo-label until a human accepts it.

The process has two resumable inference phases, followed by one export step:

1. curated YOLO26m predicts all 17,903 TRAIN images and appends one complete JSON
   record per image to `artifacts/roboflow_relabel/cross_v3/yolo.jsonl`;
2. the local Roboflow workflow processes all TRAIN images and writes
   `artifacts/roboflow_relabel/cross_v3/roboflow/proposals.jsonl`.

If those files are absent, the run starts from image 1. If they contain complete
records, rerunning skips those images. A truncated last line caused by a crash is
ignored; other corrupt lines fail loudly.

## Get the inputs

First clone/pull the current Git checkout. Then download the transfer bundle. The
bundle may contain an older copy of the scripts, so extract only data, weights and
optional progress into the current checkout:

```sh
camber stash cp stash://bunpmc/projects/yolov11sdi/transfers/cross_v3_resume_v1.tar cross_v3_resume_v1.tar
tar -xf cross_v3_resume_v1.tar \
  data/transfer/curated_v6p5_v3.zip \
  weights/curated_yolo26m/best.pt \
  artifacts/roboflow_relabel/cross_v3/yolo.jsonl \
  artifacts/roboflow_relabel/cross_v3/roboflow/proposals.jsonl
```

If either optional JSONL member is absent, `tar` may report it as missing; extract
the two mandatory files separately and proceed from image 1. Do not overwrite the
current `tools/roboflow_relabel` folder with the archived copy.

Run from the Git root containing `tools`, `data` and `weights`. The progress JSONL
files are optional. These two inputs are mandatory:

- `data/transfer/curated_v6p5_v3.zip`
- `weights/curated_yolo26m/best.pt`

## Linux

Requirements: Python 3.10+, Docker Engine/Desktop and enough free disk space.

```sh
chmod +x tools/roboflow_relabel/setup_linux.sh
tools/roboflow_relabel/setup_linux.sh
```

Edit the generated `.env` and insert a newly issued Roboflow key:

```dotenv
ROBOFLOW_API_KEY=replace_with_a_new_key
ROBOFLOW_API_URL=http://localhost:9001
ROBOFLOW_WORKSPACE=my-workspace-j75em
ROBOFLOW_WORKFLOW_ID=evn-object-detection
V3_ARCHIVE=data/transfer/curated_v6p5_v3.zip
CROSS_MODEL=weights/curated_yolo26m/best.pt
CROSS_DEVICE=cpu
```

Run YOLO26m first; this phase does not need Docker or a Roboflow key:

```sh
source .venv-roboflow/bin/activate
python tools/roboflow_relabel/doctor.py --phase yolo
python -u tools/roboflow_relabel/run_v3.py --phase yolo
```

After YOLO finishes, start the Roboflow server in terminal 1:

```sh
source .venv-roboflow/bin/activate
inference server start --env-file .env \
  --image roboflow/roboflow-inference-server-cpu:latest
```

Continue in terminal 2:

```sh
source .venv-roboflow/bin/activate
python tools/roboflow_relabel/doctor.py --phase roboflow --require-server
python -u tools/roboflow_relabel/run_v3.py --phase roboflow
python -u tools/roboflow_relabel/run_v3.py --phase export
```

## Windows

Requirements: Python 3.10/3.11, Docker Desktop running Linux containers, and
PowerShell. From the extracted root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\tools\roboflow_relabel\setup_windows.ps1
```

Edit `.env` as above. Relative paths work on Windows. For custom absolute paths,
prefer forward slashes:

```dotenv
V3_ARCHIVE=D:/datasets/curated_v6p5_v3.zip
CROSS_MODEL=D:/models/curated_yolo26m/best.pt
```

Run YOLO26m first; this phase does not need Docker or a Roboflow key:

```powershell
.\.venv-roboflow\Scripts\Activate.ps1
python tools\roboflow_relabel\doctor.py --phase yolo
python -u tools\roboflow_relabel\run_v3.py --phase yolo
```

After YOLO finishes, start the Roboflow server in PowerShell 1:

```powershell
.\.venv-roboflow\Scripts\Activate.ps1
inference server start --env-file .env `
  --image roboflow/roboflow-inference-server-cpu:latest
```

Continue in PowerShell 2:

```powershell
.\.venv-roboflow\Scripts\Activate.ps1
python tools\roboflow_relabel\doctor.py --phase roboflow --require-server
python -u tools\roboflow_relabel\run_v3.py --phase roboflow
python -u tools\roboflow_relabel\run_v3.py --phase export
```

If activation is restricted, call `.venv-roboflow\Scripts\python.exe` directly.

## GPU teacher inference

CPU works but 17,903 YOLO predictions will be slow. Install the CUDA-enabled
PyTorch build matching the host inside `.venv-roboflow`, verify
`torch.cuda.is_available()` is `True`, then set `CROSS_DEVICE=0`. This accelerates
the YOLO teacher; the documented Roboflow container remains CPU.

## Outputs and rules

Final export:

- `data/working/v6p5_v3_cross_labeled/data.yaml`
- `data/working/v6p5_v3_cross_labeled/data_val_clean.yaml`
- `artifacts/roboflow_relabel/cross_v3/changes.json`
- `artifacts/roboflow_relabel/cross_v3/conflicts.json`
- `artifacts/roboflow_relabel/cross_v3/summary.json`

Mapping: Insulator → insulator (0); damage_insulator, critical, CenterDamage,
EdgeSmallDamage, EdgeBigDamage, cautious and low → insulator_broken (1). YOLO
teacher class names map directly to the original six classes.

Confidence ≥ 0.6 is required. Same-class overlap IoU ≥ 0.3 is skipped.
Different-class overlap IoU ≥ 0.5 goes to conflict review, except contextual
insulator versus insulator_broken overlap. VAL and VAL_CLEAN are never modified.

Do not delete `yolo.jsonl` or `roboflow/proposals.jsonl` merely to restart the
shell. To intentionally run independently from image 1, use another progress
folder:

```sh
python -u tools/roboflow_relabel/run_v3.py \
  --output-root artifacts/roboflow_relabel/cross_v3_fresh --phase yolo
```

`--phase all` remains available and performs YOLO → Roboflow → export in one
command. Separate phases are recommended for a fresh long run because each phase
can be observed and backed up independently. Reversing YOLO and Roboflow does not
reduce the total number of images; both are required before export.

The final dataset export path is shared, so finish and review one experiment at a
time. Back up the checkpoint JSONL files periodically during a long run.
