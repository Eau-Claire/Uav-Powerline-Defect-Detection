#!/usr/bin/env python
"""(Re)generate notebooks/: ONLY the ordered steps needed to train on the newest dataset.

    python scripts/build_notebooks.py

01 setup -> 02 parent -> 03 external stage -> 04 review+compose -> 05 train -> 06 evaluate.
Each notebook starts with the Camber datasets it reads/writes. Notebooks hold no
business logic; edit src/yolov11sdi instead.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from yolov11sdi.config import load_config
from yolov11sdi.nbgen import LOCAL_BOOTSTRAP, check_compiles, code, md, notebook, write_notebook

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config("yolo11n_ablation", ROOT / "configs")
R = CFG.get("v6p5.remote")
DS = CFG.get("v6p2e.root")
TAG = CFG.get("v6p2e.tag")
LIN = CFG.get("v6p2e.expected_lineage")
OLD_NOTEBOOKS = ["00_environment_and_status", "10_v6p2e_parent_prepare", "20_v6p5_external_stage",
                 "30_v6p5_compose_qa", "40_yolo11n_640_data_ablation", "50_evaluate_and_compare",
                 "60_freeze_v6p5_candidate", "70_yolo11m_1024", "80_rtdetr_r50_b0", "99_pipeline_status"]

DATASETS = f'''\
# ===== DATASET TRÊN CAMBER — điền tag nếu muốn chọn tay, None = tự động =====
PARENT_DATASET = "{TAG}"   # V6.2E: cố định, KHÔNG đổi
EXTERNAL_STAGE = None   # vd "v6p5_ext_0123456789ab"     (tạo ở bước 03)
COMPOSE        = None   # vd "v6p5_compose_0123456789ab" (tạo ở bước 04)
REQUIRE_CAMBER = True   # lưu chính trên Camber -> dừng nếu chưa đăng nhập

from yolov11sdi.camber import CamberClient
from yolov11sdi.environment import load_camber_api_key
from yolov11sdi.reporting import status
cfg = pipeline.load(stage_tag=EXTERNAL_STAGE, compose_tag=COMPOSE, parent_tag=PARENT_DATASET)
load_camber_api_key(paths.env)
cam = CamberClient(paths, cfg.get("camber.stash_prefix"), cfg.runtime("camber_bin"))
if REQUIRE_CAMBER and not cam.available():
    raise RuntimeError("Camber chưa sẵn sàng: chạy `camber login` hoặc export CAMBER_API_KEY "
                       "(CAMBER_BIN=... nếu CLI không nằm trong PATH). Chỉ chạy local: REQUIRE_CAMBER=False")
display(status.dataset_card(cfg, paths, cam))'''


def run_cell(stage: str) -> str:
    return (f'outcome = pipeline.run_stage("{stage}", cfg, force=FORCE_REBUILD, dry_run=DRY_RUN)\n'
            'print(outcome)\n'
            'display(pd.Series(outcome.summary, dtype=object).to_frame("value"))')


def verify_cell(*stages: str) -> str:
    lines = ["# Kiểm tra output đã thật sự nằm trên Camber chưa"]
    lines += [f'display(pipeline.verify_on_camber("{s}", cfg))' for s in stages]
    return "\n".join(lines)


def io_table(inputs: list[str], outputs: list[str]) -> str:
    rows = ["| | Camber Stash |", "|---|---|"]
    rows += [f"| **Input** | `{p}` |" for p in inputs]
    rows += [f"| **Output** | `{p}` |" for p in outputs]
    return "\n".join(rows)


def header(step: str, title: str, body: str, inputs: list[str], outputs: list[str], nxt: str) -> dict:
    return md(f"""# Bước {step} — {title}

{body}

{io_table(inputs, outputs)}

Chạy lại notebook bất cứ lúc nào: bước đã xong + đúng config sẽ được **reuse**, bước dở dang sẽ **resume**.
Bước tiếp theo: **{nxt}**""")


def nb01():
    return [
        header("01/06", "Setup + kiểm tra", "Kiểm tra môi trường, thư viện, đăng nhập Camber và trạng thái pipeline. "
               "Không tải / không sửa dữ liệu.", [f"{DS}/v6p2e/{TAG}/"], ["—"], "02_prepare_parent_v6p2e"),
        code(LOCAL_BOOTSTRAP),
        code('''\
        import importlib
        mods = {"yaml": "core", "pandas": "core", "PIL": "core", "imagehash": "stage", "cv2": "stage",
                "fiftyone": "stage", "gdown": "stage", "huggingface_hub": "stage", "ultralytics": "train", "torch": "train"}
        rows = []
        for m, extra in mods.items():
            try:
                mod = importlib.import_module(m); rows.append((m, extra, getattr(mod, "__version__", "ok")))
            except Exception as e:
                rows.append((m, extra, f"MISSING -> pip install -e '.[{extra}]'"))
        display(pd.DataFrame(rows, columns=["module", "extra", "version"]))'''),
        code(DATASETS),
        code('''\
        display(status.stage_table(cfg, paths))
        ov = status.overview(cfg, paths)
        display(pd.Series({k: v for k, v in ov.items() if k != "disk"}, dtype=object).to_frame("value"))
        d = ov["disk"]; print(f"disk free {d['free_gb']:.1f} / {d['total_gb']:.1f} GB at {d['path']}")'''),
    ]


def nb02():
    return [
        header("02/06", "Chuẩn bị parent V6.2E (low disk)",
               "V6.2E **bất biến** (17,483 / 2,272 / 2,012). Tải freeze + split manifest, đối chiếu config "
               "(lệch là dừng), stream ≤1200 ảnh TRAIN có insulator làm background, pHash VAL+TEST một lần, "
               "rồi xoá archive lớn. pHash cache được lưu lên Camber để Kaggle/local khác không phải tính lại.",
               [f"{DS}/v6p2e/{TAG}/v6p2e.freeze.json", f"{DS}/v6p2e/{TAG}/v6p2e_split_manifest.csv",
                f"{DS}/v6p2e/{TAG}/v6p2e_reconciled_labels.zip",
                f"{DS}/curated_v6_rs1280_{LIN['base_tag']}.tar",
                f"{DS}/v6p1/{LIN['parent_dataset_tag']}/v6p1_fotl_overlay_{LIN['overlay_tag']}.tar.gz"],
               [f"{R['stage_root']}/parent_cache/{TAG}/parent_val_test_phash.csv.gz",
                f"{R['stage_root']}/parent_cache/{TAG}/parent_train_background_pool.csv",
                "(local) data/parent/train_background_pool/"],
               "03_build_external_stage_v6p5"),
        code(LOCAL_BOOTSTRAP),
        code(DATASETS),
        code(run_cell("parent")),
        code(verify_cell("parent")),
        code('''\
        pool = pd.read_csv(paths.resolve(store.load("parent").output("bg_pool_csv")["path"]))
        for p in pool["image"].head(3):
            display(IPImage(filename=str(paths.resolve(p)), width=480))'''),
    ]


def nb03():
    return [
        header("03/06", "Tạo V6.5 external stage → `v6p5_ext_<12hex>`",
               "Open Images V7 TRAIN (Balloon/Kite/Plastic bag/Bird/Person, detections-only + GrabCut, 250/class) · "
               "MPCD (broken → broken_strand) · Power Equipment (dx_dg/yw/nw, dx_sg chỉ review) · synthetic "
               "foreign_object 100/subtype trên TRAIN background · exact-SHA + pHash QA · review manifest · "
               "đóng gói + freeze + upload. Checkpoint theo class / subtype.",
               ["Open Images V7 (internet)", "MPCD Google Drive 1KyciMmwL2_p_-mwckHFqG5fkBA1jzATv",
                "HF sxiong/Power-equipment-image-dataset", "kết quả bước 02"],
               [f"{R['stage_root']}/v6p5_ext_<12hex>/v6p5_ext_<12hex>.freeze.json",
                f"{R['stage_root']}/v6p5_ext_<12hex>/v6p5_ext_<12hex>_overlay.tar.gz",
                f"{R['stage_root']}/v6p5_ext_<12hex>/v6p5_ext_<12hex>_qa.zip",
                R["mpcd_direct_archive"], R["power_equipment_direct_archive"]],
               "04_review_and_compose_v6p5"),
        code(LOCAL_BOOTSTRAP),
        code(DATASETS),
        code('''\
        for s in ["parent", "openimages", "mpcd", "power_equipment", "synthetic", "dedup"]:
            print(pipeline.run_stage(s, cfg, force=FORCE_REBUILD, dry_run=DRY_RUN))
        d = store.load("dedup").summary if store.load("dedup") else {}
        print("exact SHA trùng V6.2E (HARD gate):", d.get("exact_duplicates"))
        print("pHash gần trùng VAL/TEST (cần review):", d.get("phash_candidates"))'''),
        code(run_cell("external_freeze")),
        code(verify_cell("external_freeze")),
        code('''\
        contacts = sorted((paths.qa / "v6p5_external" / "contacts").glob("*.jpg"))
        print(len(contacts), "contact sheets")
        for p in contacts[:12]:
            print(p.name); display(IPImage(filename=str(p), width=640))'''),
        md("Ghi lại tag `v6p5_ext_...` ở bảng trên — bước 04 tự lấy, hoặc điền vào `EXTERNAL_STAGE`."),
    ]


def nb04():
    return [
        header("04/06", "Review KEEP/DROP + compose → `v6p5_compose_<12hex>`",
               "**Không auto-promotion.** Mỗi ứng viên TRAIN cần `KEEP`/`DROP` trong "
               "`artifacts/qa/v6p5_review_decisions.csv` (trống = pending; file này được đồng bộ lên Camber). "
               "Còn pending → dừng ở `needs_review`. Exact-SHA trùng là hard gate; pHash phải review.\n\n"
               "Quy trình: chạy → mở CSV điền decision/reviewer/comment → chạy lại.",
               [f"{R['stage_root']}/v6p5_ext_<12hex>/ (EXTERNAL_STAGE)"],
               [f"{R['compose_root']}/review_decisions/v6p5_ext_<12hex>/v6p5_review_decisions.csv",
                f"{R['compose_root']}/v6p5_compose_<12hex>/v6p5_compose_<12hex>.index.json  (field synthetic_archive)",
                f"{R['compose_root']}/v6p5_compose_<12hex>/v6p5_compose_<12hex>.freeze.json",
                f"{R['compose_root']}/v6p5_compose_<12hex>/v6p5_compose_<12hex>_synthetic_approved.tar.gz",
                f"{R['compose_root']}/v6p5_compose_<12hex>/*.review.resolved.csv, *.direct_approved_manifest.csv"],
               "05_train_yolo11n_640_ablation"),
        code(LOCAL_BOOTSTRAP),
        code(DATASETS),
        code(run_cell("compose")),
        code('''\
        dec_path = paths.resolve(cfg.get("compose.decisions_file"))
        print("file review:", dec_path)
        if dec_path.exists():
            dec = pd.read_csv(dec_path, keep_default_na=False)
            display(dec.assign(decision=dec["decision"].replace("", "PENDING"))
                    .groupby(["candidate_type", "source", "target_split", "decision"]).size().rename("n").reset_index())
        for name in ["pending_review.csv", "phash_candidates_to_review.csv"]:
            p = paths.compose / name
            if p.exists():
                print(name); display(pd.read_csv(p, keep_default_na=False).head(50))
        for p in sorted((paths.working / "compose_qa_extract").glob("*/contacts/*.jpg"))[:cfg.get("compose.max_contact_sheets")]:
            print(p.name); display(IPImage(filename=str(p), width=640))'''),
        code(verify_cell("compose")),
        md("Nếu trạng thái là `needs_review`: **dừng ở đây**, điền CSV rồi chạy lại notebook này."),
    ]


def nb05():
    return [
        header("05/06", "Train YOLO11n@640, 50 epoch (data ablation)",
               "**Data ablation, KHÔNG phải benchmark V6.5 vs V6.4 cuối.** V6.2E TRAIN + MPCD/PowerEquipment TRAIN "
               "đã KEEP + synthetic đã duyệt; đánh giá trên **VAL V6.2E đóng băng**; TEST không bao giờ được tạo. "
               "`yolo11n.pt`, 640, 50 ep, batch -1, seed 20261007. Cần GPU CUDA (hoặc dùng launcher Kaggle).\n\n"
               f"Experiment ID: `{CFG.get('training.experiment')}` · checkpoint sync lên Camber mỗi "
               f"{CFG.get('training.sync_every_epochs')} epoch · mất máy/kernel → chạy lại là resume từ `last.pt`.",
               [f"{R['compose_root']}/v6p5_compose_<12hex>/ (COMPOSE)", f"{DS}/v6p2e/{TAG}/ + archive base/overlay",
                R["mpcd_direct_archive"], R["power_equipment_direct_archive"]],
               [f"{R['train_root']}/{CFG.get('training.remote_tag_prefix')}<12hex>/" + "{last.pt, best.pt, run_state.json, results.csv, args.yaml}",
                f"(local) runs/{CFG.get('training.run_family')}/{CFG.get('training.experiment')}__v6p5_compose_<12hex>/"],
               "06_evaluate_vs_v6p2e_baseline"),
        code(LOCAL_BOOTSTRAP),
        code(DATASETS),
        code(f'pipeline.run_stage("yolo11n_ablation", cfg, dry_run=True)'),
        code(run_cell("yolo11n_ablation")),
        code(verify_cell(CFG.get("training.experiment"))),
        code(f'''\
        from yolov11sdi.reporting.status import latest_checkpoint
        print(latest_checkpoint(paths))
        runs = sorted(paths.runs.glob("{CFG.get('training.run_family')}/*/results.csv"))
        if runs:
            r = pd.read_csv(runs[-1]); r.columns = [c.strip() for c in r.columns]
            display(r.tail())
            r.plot(x="epoch", y=[c for c in r.columns if "mAP50" in c], figsize=(8, 4))'''),
    ]


def nb06():
    return [
        header("06/06", "Đánh giá vs baseline V6.2E YOLO11n@640",
               "Baseline: mAP50 0.768 / mAP50:95 0.413 / gap 0.355 (foreign .537/.295, strand .749/.356).\n\n"
               "Gate: mAP50:95 toàn cục giảm ≤ 1 điểm; foreign_object AP50 +2 và AP50:95 +1; broken_strand AP50:95 +1. "
               "**Pass chỉ mở quyền review/promotion bằng tay** → freeze V6.5 canonical → YOLO11m@1024 → RT-DETR B0.",
               [f"{R['train_root']}/{CFG.get('training.remote_tag_prefix')}<12hex>/v6p5_yolo11n_data_ablation_report.json"],
               [f"{R['train_root']}/{CFG.get('training.remote_tag_prefix')}<12hex>/{CFG.get('training.experiment')}_vs_baseline.csv"],
               "quyết định promotion (người làm)"),
        code(LOCAL_BOOTSTRAP),
        code(DATASETS),
        code(run_cell("evaluate")),
        code('''\
        cmp_csv = paths.exports / f"{cfg.get('training.experiment')}_vs_baseline.csv"
        if cmp_csv.exists():
            display(pd.read_csv(cmp_csv))'''),
        md("Nếu foreign_object tăng nhưng broken_strand giảm → tách ablation theo nguồn (`abl02` chỉ foreign, "
           "`abl03` chỉ strand) trước khi freeze bất cứ thứ gì."),
    ]


NOTEBOOKS = {
    "01_setup_and_check.ipynb": nb01,
    "02_prepare_parent_v6p2e.ipynb": nb02,
    "03_build_external_stage_v6p5.ipynb": nb03,
    "04_review_and_compose_v6p5.ipynb": nb04,
    "05_train_yolo11n_640_ablation.ipynb": nb05,
    "06_evaluate_vs_v6p2e_baseline.ipynb": nb06,
}


def main() -> None:
    out = ROOT / "notebooks"
    for old in OLD_NOTEBOOKS:
        (out / f"{old}.ipynb").unlink(missing_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = notebook(fn())
        check_compiles(nb, name)
        print(write_notebook(out / name, nb).relative_to(ROOT))


if __name__ == "__main__":
    main()
