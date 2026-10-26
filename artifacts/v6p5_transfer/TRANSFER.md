# V6.5-12 transfer bundle

Remote root:

`stash://bunpmc/projects/yolov11sdi/datasets/v6p5_12_transfer/20261010`

## Files

- `v6p5_roboflow_ablation.tar`
  - 5.4 GiB, 41,222 TAR entries.
  - SHA256: `abfe19fba5cf70d4ca6f822417c2dc9eecb9da053688fa66735dc3040236743f`
- `v6p5_12_resume_pack.tar.gz`
  - SHA256: `9e1cbc24b91b0866979e3f67c774973cf7c6a6399f7920f7fa6399fb8a195aea`
  - Contains `yolo11n.pt`, the MuSGD comparison `best.pt`, AdamW
    `last.pt`/`best.pt` through epoch 29, training metadata, and the V6.5-12
    notebook.

## Restore on a new WSL machine

Download both archives from the remote root. From the repository root:

```bash
mkdir -p /home/intern/datasets
tar -xf v6p5_roboflow_ablation.tar -C /home/intern/datasets

mkdir -p /tmp/v6p5_12_restore
tar -xzf v6p5_12_resume_pack.tar.gz -C /tmp/v6p5_12_restore

cp /tmp/v6p5_12_restore/checkpoints/yolo11n.pt \
  legacy/notebooks/v6p5/yolo11n.pt

mkdir -p artifacts/v6p5_train_local/20261009T024019Z_roboflow/runs/yolo11n_v6p5_roboflow_ablation_640/weights
cp /tmp/v6p5_12_restore/checkpoints/roboflow_musgd_best.pt \
  artifacts/v6p5_train_local/20261009T024019Z_roboflow/runs/yolo11n_v6p5_roboflow_ablation_640/weights/best.pt

mkdir -p artifacts/v6p5_train_local/20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640/weights
cp /tmp/v6p5_12_restore/checkpoints/adamw_last_epoch29.pt \
  artifacts/v6p5_train_local/20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640/weights/last.pt
cp /tmp/v6p5_12_restore/checkpoints/adamw_best_through_epoch29.pt \
  artifacts/v6p5_train_local/20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640/weights/best.pt
cp /tmp/v6p5_12_restore/checkpoints/adamw_results_through_epoch29.csv \
  artifacts/v6p5_train_local/20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640/results.csv
cp /tmp/v6p5_12_restore/checkpoints/adamw_args.yaml \
  artifacts/v6p5_train_local/20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640/args.yaml
```

Verify downloads:

```bash
sha256sum v6p5_roboflow_ablation.tar v6p5_12_resume_pack.tar.gz
```

In V6.5-12 CONFIG, set:

```python
RUN_ID = "20261009T131324Z_adamw"
```

Then run CONFIG, GPU/DATASET GATE, and TRAIN. It resumes from epoch 30.
