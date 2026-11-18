"""Regenerate every paper figure into figures/out/ (PDF + 300 dpi PNG).

    .venv/bin/python figures/make_all.py
    .venv/bin/python figures/make_all.py --with-train-sizes   # also scan 17,903 train labels

A figure whose source data is missing is skipped with a message; the rest still run.
"""
import argparse
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

JOBS = [
    ("fig01_dataset_distribution", {}),
    ("fig02_curation_actions", {}),
    ("fig03_object_size", {"split": "test_main_v1"}),
    ("fig03_object_size", {"split": "curated_val_clean"}),
    ("fig04_raw_vs_curated", {}),          # also writes fig05
    ("fig06_seed36_raw_vs_curated", {}),
    ("fig07_benchmark", {}),               # fig07/08/09/12 for every protocol
    ("fig10_val_test_gap", {}),
    ("fig11_training_curves", {}),
    ("fig13_test_protocols", {}),          # also writes fig14
    ("fig15_curation_ablation", {}),       # also writes fig16
    ("fig17_hardware", {}),
    ("fig18_v6p5_ablation", {}),           # fig18/19/20
    ("diagrams", {}),                      # diag01-05 method diagrams
    ("diagrams_project", {}),              # diag06-07 project pipeline + fig21 dataset versions
]


def main(with_train_sizes=False):
    jobs = list(JOBS)
    if with_train_sizes:
        jobs.append(("fig03_object_size", {"split": "curated_train"}))
    failed = []
    for name, kwargs in jobs:
        print(f"[{name}] {kwargs or ''}")
        try:
            importlib.import_module(name).main(**kwargs)
        except FileNotFoundError as e:
            print(f"  skip: missing input {e.filename}")
        except Exception:  # keep going; report at the end
            traceback.print_exc()
            failed.append(name)
    out = Path(__file__).resolve().parent / "out"
    print(f"\n{len(list(out.rglob('*.pdf')))} figures in {out}")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-train-sizes", action="store_true")
    main(ap.parse_args().with_train_sizes)
