
from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, zipfile, tarfile, io
import tempfile

COMPOSE_TAG = 'v6p5_compose_4b9a452e24e2'
COMPOSE_REMOTE = 'stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage/compose_qa/v6p5_compose_4b9a452e24e2'
TRAIN_ROOT = 'stash://bunpmc/projects/yolov11sdi/checkpoints/v6p5_data_ablation'
STAGE_ROOT = 'stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage'

V6ROOT = "stash://bunpmc/projects/yolov11sdi/datasets"
V6TAG = "v6p2e_23562a45b343"

MODEL = 'yolo11n.pt'
IMGSZ = 640
EPOCHS = 50
BATCH = -1
DEVICE = 0
VAL_BATCH = 8
SEED = 20261007

ROOT = Path(tempfile.mkdtemp(prefix=f"v65_train_{COMPOSE_TAG}_"))
CACHE = ROOT/"cache"
DATASET = ROOT/"dataset"
RUNS = ROOT/"runs"

for p in [CACHE,DATASET,RUNS]:
    p.mkdir(parents=True,exist_ok=True)

subprocess.check_call([
    sys.executable,"-m","pip","install","-q",
    "ultralytics==8.4.172","pandas","numpy","pyyaml","pillow"
])

import pandas as pd
import numpy as np
import yaml
from ultralytics import YOLO

def ensure_camber():
    c=shutil.which("camber") or str(Path.home()/".camber/bin/camber")
    if not Path(c).exists():
        subprocess.check_call(
            "curl -sL https://cli.cambercloud.com/install-v2.sh | bash",
            shell=True,
        )
        os.environ["PATH"]=f"{Path.home()}/.camber/bin:"+os.environ.get("PATH","")
        c=shutil.which("camber") or str(Path.home()/".camber/bin/camber")
    assert Path(c).exists()
    return c

CAMBER=ensure_camber()

def run(cmd):
    r=subprocess.run(cmd,text=True,capture_output=True,env=os.environ.copy())
    if r.returncode!=0:
        raise RuntimeError((r.stderr or r.stdout)[:10000])
    return r

def get(remote,local):
    local=Path(local)
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists():
        run([CAMBER,"stash","cp",remote,str(local)])
    return local

def put(local,remote):
    run([CAMBER,"stash","cp",str(local),remote])

def norm_names(names):
    if isinstance(names,dict):
        return [names[k] for k in sorted(names,key=lambda x:int(x))]
    return list(names or [])

def safe_link_or_copy(src,dst):
    src=Path(src); dst=Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src,dst)
    except Exception:
        shutil.copy2(src,dst)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

# =============================================================================
# A. RECONSTRUCT V6.2E BASE
# =============================================================================
index_local=get(f"{COMPOSE_REMOTE}/{COMPOSE_TAG}.index.json",CACHE/"compose.index.json")
compose_index=json.loads(index_local.read_text())
if compose_index.get("compose_tag") != COMPOSE_TAG:
    raise RuntimeError("Compose index tag mismatch")
compose_freeze_path=get(f"{COMPOSE_REMOTE}/{COMPOSE_TAG}.freeze.json",CACHE/"compose.freeze.json")
compose_freeze=json.loads(compose_freeze_path.read_text())
if compose_freeze.get("tag") != COMPOSE_TAG:
    raise RuntimeError("Compose freeze tag mismatch")

freeze_path=get(
    f"{V6ROOT}/v6p2e/{V6TAG}/v6p2e.freeze.json",
    CACHE/"v6p2e.freeze.json",
)
freeze=json.loads(freeze_path.read_text())

base_tar=get(
    f"{V6ROOT}/curated_v6_rs1280_{freeze['base_tag']}.tar",
    CACHE/"base.tar",
)
overlay_tar=get(
    f"{V6ROOT}/v6p1/{freeze['parent_dataset_tag']}/v6p1_fotl_overlay_{freeze['overlay_tag']}.tar.gz",
    CACHE/"overlay.tar.gz",
)
labels_zip=get(
    f"{V6ROOT}/v6p2e/{V6TAG}/v6p2e_reconciled_labels.zip",
    CACHE/"labels.zip",
)
manifest_csv=get(
    f"{V6ROOT}/v6p2e/{V6TAG}/v6p2e_split_manifest.csv",
    CACHE/"manifest.csv",
)

BASE_EX=CACHE/"base_ex"
OV_EX=CACHE/"overlay_ex"
RECON=CACHE/"recon"
for p in [BASE_EX,OV_EX,RECON]:
    p.mkdir(parents=True,exist_ok=True)

def find_sixclass_root(root):
    target=[
        "insulator",
        "insulator_broken",
        "insulator_flashover",
        "bird_nest",
        "foreign_object",
        "broken_strand",
    ]
    for yp in Path(root).rglob("data.yaml"):
        try:
            cfg=yaml.safe_load(yp.read_text())
            if norm_names(cfg.get("names"))==target:
                return yp.parent
        except Exception:
            pass
    raise RuntimeError(f"Six-class root not found under {root}")

if not any(BASE_EX.rglob("data.yaml")):
    with tarfile.open(base_tar,"r:*") as t:
        t.extractall(BASE_EX,filter="data")

if not any(OV_EX.rglob("data.yaml")):
    with tarfile.open(overlay_tar,"r:*") as t:
        t.extractall(OV_EX,filter="data")

if not any(RECON.rglob("*.txt")):
    with zipfile.ZipFile(labels_zip) as z:
        z.extractall(RECON)

BASE_ROOT=find_sixclass_root(BASE_EX)
OV_ROOT=find_sixclass_root(OV_EX)
manifest=pd.read_csv(manifest_csv)

def locate(row):
    old=str(row["image_path_source"]).replace("\\","/")
    oldsplit=str(row["previous_v6p1_split"])
    marker=f"/images/{oldsplit}/"
    suffix=old.split(marker,1)[1]

    if suffix.startswith("base/"):
        root=BASE_ROOT; rel=suffix[len("base/"):]
    elif suffix.startswith("fotl/"):
        root=OV_ROOT; rel=suffix[len("fotl/"):]
    else:
        raise RuntimeError(suffix)

    return (
        root/"images"/oldsplit/rel,
        root/"labels"/oldsplit/Path(rel).with_suffix(".txt"),
    )

def label_for(row,lp0):
    rp=RECON/f"{row['sha256']}.txt"
    if str(row.get("resolution","")).startswith("merged") and rp.exists():
        return rp
    return lp0

for s in ["train","val"]:
    (DATASET/"images"/s).mkdir(parents=True,exist_ok=True)
    (DATASET/"labels"/s).mkdir(parents=True,exist_ok=True)

unresolved=[]

for i,row in manifest[manifest["split"].isin(["train","val"])].iterrows():
    split=str(row["split"])
    ip,lp0=locate(row)
    lp=label_for(row,lp0)

    if not ip.exists() or not lp.exists():
        unresolved.append((int(i),str(ip),str(lp)))
        continue

    name=f"{str(row['sha256'])[:16]}_{i:06d}{ip.suffix.lower()}"
    safe_link_or_copy(ip,DATASET/"images"/split/name)
    safe_link_or_copy(lp,DATASET/"labels"/split/Path(name).with_suffix(".txt"))

if unresolved:
    pd.DataFrame(unresolved,columns=["row","image","label"]).to_csv(
        ROOT/"unresolved.csv",index=False
    )
    raise RuntimeError(f"V6.2E unresolved files: {len(unresolved)}")

base_train_count=len(list((DATASET/"images/train").glob("*")))
base_val_count=len(list((DATASET/"images/val").glob("*")))

# =============================================================================
# B. ADD DIRECT V6.5 TRAIN OVERLAYS
# =============================================================================
overlay_sources=[
    (
        compose_index["direct_archives"]["mpcd"],
        CACHE/"mpcd.tar.gz",
        "mpcd_broken_strand",
    ),
    (
        compose_index["direct_archives"]["power_equipment"],
        CACHE/"pe.tar.gz",
        "power_equipment_stage",
    ),
]

added_direct=0

for remote,local,rootname in overlay_sources:
    get(remote,local)

    with tarfile.open(local,"r:gz") as t:
        # Stream only train images/labels to avoid materializing external_eval.
        members={m.name:m for m in t.getmembers() if m.isfile()}

        image_members=[
            n for n in members
            if f"{rootname}/images/train/" in n
        ]

        for n in image_members:
            m=members[n]
            raw=t.extractfile(m).read()

            basename=Path(n).name
            source_prefix="mpcd" if "mpcd" in rootname else "pe"
            name=f"v65_{source_prefix}_{basename}"

            di=DATASET/"images/train"/name
            di.write_bytes(raw)

            label_name=str(
                Path(n.replace("/images/train/","/labels/train/"))
                .with_suffix(".txt")
            )

            lm=members.get(label_name)
            if lm is None:
                raise RuntimeError(f"Missing overlay label: {label_name}")

            dl=DATASET/"labels/train"/Path(name).with_suffix(".txt")
            dl.write_bytes(t.extractfile(lm).read())
            added_direct+=1

    local.unlink(missing_ok=True)

# =============================================================================
# C. ADD SYNTHETIC V6.5 TRAIN OVERLAY
# =============================================================================
# Discover exact synthetic archive filename from compose index.
index_remote=f"{COMPOSE_REMOTE}/{COMPOSE_TAG}.index.json"
index_local=get(index_remote,CACHE/"compose.index.json")
compose_index=json.loads(index_local.read_text())

synthetic_remote=compose_index["synthetic_archive"]
synthetic_local=get(synthetic_remote,CACHE/"synthetic.tar.gz")

expected_synthetic_sha=compose_freeze["artifacts"]["synthetic_archive"]["sha256"]
if sha256_file(synthetic_local) != expected_synthetic_sha:
    raise RuntimeError("Synthetic archive SHA256 mismatch")

added_synth=0

with tarfile.open(synthetic_local,"r:gz") as t:
    members={m.name:m for m in t.getmembers() if m.isfile()}

    image_members=[
        n for n in members
        if "synthetic_foreign/images/train/" in n
    ]

    for n in image_members:
        m=members[n]
        basename=Path(n).name
        name=f"v65_syn_{basename}"

        (DATASET/"images/train"/name).write_bytes(
            t.extractfile(m).read()
        )

        label_name=str(
            Path(n.replace("/images/train/","/labels/train/"))
            .with_suffix(".txt")
        )

        lm=members.get(label_name)
        if lm is None:
            raise RuntimeError(f"Missing synthetic label: {label_name}")

        (DATASET/"labels/train"/Path(name).with_suffix(".txt")).write_bytes(
            t.extractfile(lm).read()
        )

        added_synth+=1

if added_synth != int(compose_freeze["counts"]["approved_synthetic"]):
    raise RuntimeError("Synthetic image count does not match compose freeze")
synthetic_local.unlink(missing_ok=True)

# =============================================================================
# D. WRITE DATA YAML
# =============================================================================
names=[
    "insulator",
    "insulator_broken",
    "insulator_flashover",
    "bird_nest",
    "foreign_object",
    "broken_strand",
]

data_yaml=DATASET/"data.yaml"
data_yaml.write_text(yaml.safe_dump({
    "path":str(DATASET),
    "train":"images/train",
    "val":"images/val",
    "names":{i:n for i,n in enumerate(names)},
},sort_keys=False))

train_count=len(list((DATASET/"images/train").glob("*")))
val_count=len(list((DATASET/"images/val").glob("*")))

# =============================================================================
# E. TRAIN SAME-SCALE YOLO11n@640 / 50 epochs
# =============================================================================
model=YOLO(MODEL)

result=model.train(
    data=str(data_yaml),
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    workers=8,
    pretrained=True,
    seed=SEED,
    project=str(RUNS),
    name="yolo11n_v6p5_data_ablation_640",
    exist_ok=True,
    plots=False,
    verbose=True,
)

run_dir=RUNS/"yolo11n_v6p5_data_ablation_640"
best=run_dir/"weights/best.pt"
assert best.exists(), best

# =============================================================================
# F. STRICT VAL EVALUATION
# =============================================================================
best_model=YOLO(str(best))

metrics=best_model.val(
    data=str(data_yaml),
    split="val",
    imgsz=IMGSZ,
    batch=VAL_BATCH,
    device=DEVICE,
    plots=False,
    verbose=False,
)

m=metrics.box
all_ap=np.asarray(m.all_ap,dtype=float)
ap50=np.asarray(m.ap50,dtype=float)
class_ids=[int(x) for x in np.asarray(m.ap_class_index).tolist()]

if all_ap.ndim==1:
    all_ap=all_ap[:,None]

thresholds=0.50+0.05*np.arange(all_ap.shape[1])

def col_for(t):
    return int(np.argmin(np.abs(thresholds-t)))

per_class=[]

for ri,cid in enumerate(class_ids):
    vals=all_ap[ri]
    row={
        "cid":cid,
        "class":names[cid],
        "AP50":float(ap50[ri]),
        "AP50_95":float(np.mean(vals)),
        "gap":float(ap50[ri]-np.mean(vals)),
        "AP75":float(vals[col_for(0.75)]) if all_ap.shape[1]>1 else None,
        "AP90":float(vals[col_for(0.90)]) if all_ap.shape[1]>1 else None,
        "AP95":float(vals[col_for(0.95)]) if all_ap.shape[1]>1 else None,
    }
    per_class.append(row)

# Old V6.2E YOLO11n@640 reference
baseline={
    "mAP50":0.768,
    "mAP50_95":0.413,
    "gap":0.355,
    "per_class":{
        "insulator":{"AP50":0.934,"AP50_95":0.638},
        "insulator_broken":{"AP50":0.800,"AP50_95":0.407},
        "insulator_flashover":{"AP50":0.755,"AP50_95":0.326},
        "bird_nest":{"AP50":0.831,"AP50_95":0.455},
        "foreign_object":{"AP50":0.537,"AP50_95":0.295},
        "broken_strand":{"AP50":0.749,"AP50_95":0.356},
    },
}

report={
    "experiment":"V6.2E + V6.5 external candidate data ablation",
    "compose_tag":COMPOSE_TAG,
    "model":"YOLO11n",
    "imgsz":IMGSZ,
    "epochs":EPOCHS,
    "base_train_images":base_train_count,
    "added_direct_train_images":added_direct,
    "added_synthetic_train_images":added_synth,
    "final_train_images":train_count,
    "val_images":val_count,
    "mAP50":float(m.map50),
    "mAP50_95":float(m.map),
    "gap":float(m.map50-m.map),
    "precision":float(m.mp),
    "recall":float(m.mr),
    "per_class":per_class,
    "reference_v6p2e_yolo11n":baseline,
}

report["delta_vs_v6p2e"]={
    "mAP50":report["mAP50"]-baseline["mAP50"],
    "mAP50_95":report["mAP50_95"]-baseline["mAP50_95"],
    "gap":report["gap"]-baseline["gap"],
}

for row in report["per_class"]:
    b=baseline["per_class"].get(row["class"])
    if b:
        row["delta_AP50_vs_v6p2e"]=row["AP50"]-b["AP50"]
        row["delta_AP50_95_vs_v6p2e"]=row["AP50_95"]-b["AP50_95"]

report_path=ROOT/"v6p5_yolo11n_data_ablation_report.json"
report_path.write_text(json.dumps(report,indent=2))

pd.DataFrame(per_class).to_csv(
    ROOT/"v6p5_yolo11n_strict_iou_per_class.csv",
    index=False,
)

# Practical promotion signal for the NEW sources:
# foreign AP50/AP50:95 should improve, strand strict metric should improve,
# and global AP50:95 must not materially regress.
pc={r["class"]:r for r in per_class}

promotion={
    "global_map5095_not_regress_gt_1pt":
        report["mAP50_95"] >= baseline["mAP50_95"]-0.01,
    "foreign_AP50_improves_2pt":
        pc.get("foreign_object",{}).get("AP50",0) >= baseline["per_class"]["foreign_object"]["AP50"]+0.02,
    "foreign_AP5095_improves_1pt":
        pc.get("foreign_object",{}).get("AP50_95",0) >= baseline["per_class"]["foreign_object"]["AP50_95"]+0.01,
    "strand_AP5095_improves_1pt":
        pc.get("broken_strand",{}).get("AP50_95",0) >= baseline["per_class"]["broken_strand"]["AP50_95"]+0.01,
}

promotion["candidate_pass"]=all(promotion.values())
report["promotion_gate"]=promotion
report_path.write_text(json.dumps(report,indent=2))

# =============================================================================
# G. PACKAGE + UPLOAD TO CAMBER
# =============================================================================
tag_input=json.dumps({
    "compose_tag":COMPOSE_TAG,
    "model":"YOLO11n",
    "imgsz":IMGSZ,
    "epochs":EPOCHS,
    "train_count":train_count,
},sort_keys=True).encode()

tag="v6p5_y11n_ablation_"+hashlib.sha256(tag_input).hexdigest()[:12]
remote_root=f"{TRAIN_ROOT}/{tag}"

put(best,f"{remote_root}/best.pt")
put(report_path,f"{remote_root}/{report_path.name}")
put(ROOT/"v6p5_yolo11n_strict_iou_per_class.csv",
    f"{remote_root}/v6p5_yolo11n_strict_iou_per_class.csv")

results_csv=run_dir/"results.csv"
if results_csv.exists():
    put(results_csv,f"{remote_root}/results.csv")

done=ROOT/"done.json"
done.write_text(json.dumps({
    "tag":tag,
    "remote_root":remote_root,
    "promotion_gate":promotion,
},indent=2))
put(done,f"{remote_root}/done.json")

print(json.dumps(report,indent=2))
print("REMOTE:",remote_root)

shutil.rmtree(ROOT,ignore_errors=True)
