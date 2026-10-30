# Roboflow review-first relabeling

This tool uses Roboflow's external `Insulator` detections to append dataset
class `0` (`insulator`) boxes to images that already contain class `1`
(`insulator_broken`). Existing class-1 boxes are retained exactly. The external
capitalized name is never written into the YOLO taxonomy.

## Local setup

Add these values to the repository `.env` (which is git-ignored):

```dotenv
ROBOFLOW_API_KEY=
ROBOFLOW_API_URL=http://localhost:9001
ROBOFLOW_WORKSPACE=my-workspace-j75em
ROBOFLOW_WORKFLOW_ID=evn-object-detection
ROBOFLOW_INSULATOR_CLASSES=insulator
```

Start the local server in one terminal:

```powershell
$env:DOCKER_CONFIG=(Resolve-Path ".docker-anon")
.\.venv-roboflow\Scripts\inference.exe server start --env-file .env
```

The isolated Docker config avoids a broken global `credsStore` helper and is
enough for pulling Roboflow's public server image. It does not alter the user's
global Docker configuration.

For a small validation run when the GPU image cannot be pulled, use the public
CPU image without changing model outputs:

```powershell
$env:DOCKER_CONFIG=(Resolve-Path ".docker-anon")
$env:ROBOFLOW_SERVER_IMAGE="roboflow/roboflow-inference-server-cpu:latest"
.\.venv-roboflow\Scripts\python.exe tools\roboflow_relabel\pull_image.py
.\.venv-roboflow\Scripts\inference.exe server start --env-file .env --image $env:ROBOFLOW_SERVER_IMAGE --use-local-images
```

Start the reviewer in another terminal:

```powershell
.\tools\roboflow_relabel\run_review.ps1
```

Or run the default 20-image proposal batch without the UI:

```powershell
.\.venv-roboflow\Scripts\python.exe tools\roboflow_relabel\dry_run.py --limit 20
```

The full train run is resumable and prints one progress line per image:

```powershell
.\.venv-roboflow\Scripts\python.exe tools\roboflow_relabel\dry_run.py `
  --limit 0 `
  --output-dir artifacts\roboflow_relabel\full_train `
  --resume `
  --workers 4
```

Build a complete training dataset without duplicating image storage (images are
hardlinked when source and destination are on the same volume):

```powershell
.\.venv-roboflow\Scripts\python.exe tools\roboflow_relabel\export_dataset.py `
  --source data\cache\base_extract\curated_v6 `
  --proposals artifacts\roboflow_relabel\full_train\proposals.jsonl `
  --output data\working\curated_v6_roboflow_insulator_v1
```

Run 20 images first. Verify that the workflow emits a class representing the
whole insulator. Proposals and review decisions are written below
`artifacts/roboflow_relabel/`; train/validation labels remain unchanged.

