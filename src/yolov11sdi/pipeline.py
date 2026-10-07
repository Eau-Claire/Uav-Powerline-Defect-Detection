"""Stage registry + orchestration. Notebooks and scripts/run_stage.py call `run_stage`.

Each stage:
  1. computes a stage-specific config hash and its input fingerprints;
  2. reuses a complete, matching previous result (default) or runs;
  3. writes state/<stage>.json throughout (running/complete/failed/needs_review).
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import __version__
from .camber import CamberClient
from .config import Config, load_config
from .environment import detect_environment, disk_usage, git_commit, hostname, load_camber_api_key
from .hashing import sha256_file
from .io_utils import atomic_write_csv, atomic_write_json, read_csv_safe, read_json, read_yolo_label, yolo_to_xyxy
from .logging_utils import setup_stage_logger
from .paths import ProjectPaths
from .state import FileRecord, StageRun, StateStore, check_reusable

DEFAULT_CONFIG = "yolo11n_ablation"


# =============================================================== context

@dataclass
class StageOutcome:
    stage: str
    status: str            # reused | complete | needs_review | dry_run | failed
    message: str = ""
    summary: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.stage}] {self.status}: {self.message}"


@dataclass
class Context:
    cfg: Config
    paths: ProjectPaths
    store: StateStore
    camber: CamberClient
    log: logging.Logger
    force: bool = False
    dry_run: bool = False
    commit: str | None = None

    # ---------------------------------------------------------- helpers
    def meta(self) -> dict:
        return {"git_commit": self.commit, "host": hostname(), "environment": self.paths.env.value}

    def rec(self, name: str, path: Path, remote: str | None = None, with_sha: bool = True) -> FileRecord:
        return FileRecord.of(name, path, self.paths.rel(path), remote=remote, with_sha=with_sha)

    def output_path(self, stage: str, name: str) -> Path:
        st = self.store.load(stage)
        o = st.output(name) if st else None
        if not st or st.status != "complete" or not o:
            raise RuntimeError(f"Stage '{stage}' has no complete output '{name}'. Run: "
                               f"python scripts/run_stage.py {stage}")
        return self.paths.resolve(o["path"])

    def summary(self, stage: str) -> dict:
        st = self.store.load(stage)
        return st.summary if st else {}

    def input_from(self, stage: str, name: str) -> dict:
        p = self.output_path(stage, name)
        st = self.store.load(stage)
        o = st.output(name)
        return {"name": f"{stage}:{name}", "path": o["path"], "sha256": o.get("sha256") or sha256_file(p)}

    def section_hash(self, keys) -> str:
        return self.cfg.section_hash(keys)

    def compatible_partial(self, stage: str, config_hash: str, inputs: list[dict]) -> bool:
        """True when partial checkpoints from a previous attempt may be resumed."""
        prev = self.store.load(stage)
        if prev is None:
            return True
        same_inputs = {i["name"]: i.get("sha256") for i in prev.inputs} == {i["name"]: i.get("sha256") for i in inputs}
        return prev.config_hash == config_hash and same_inputs

    def begin(self, stage: str, keys, inputs: list[dict]) -> tuple[StageRun | None, StageOutcome | None]:
        h = self.section_hash(keys)
        prev = self.store.load(stage)
        if not self.force:
            ok, reason = check_reusable(prev, h, inputs, self.paths.resolve)
            if ok:
                self.log.info("REUSE %s (%s)", stage, reason)
                return None, StageOutcome(stage, "reused", reason, prev.summary)
            self.log.info("RUN %s (%s)", stage, reason)
        else:
            self.log.info("RUN %s (force rebuild)", stage)
        if self.dry_run:
            return None, StageOutcome(stage, "dry_run", f"would run (config_hash={h[:12]})")
        return StageRun(self.store, stage, h, inputs, self.meta(),
                        float(self.cfg.runtime("progress_interval_s", 15))), None

    def upload_enabled(self) -> bool:
        if not self.cfg.runtime("upload_camber", True):
            return False
        if not self.camber.available():
            self.log.warning("Camber unavailable (CLI/key/login); outputs stay local. Sync later with "
                             "scripts/sync_camber.py")
            if self.cfg.runtime("camber_required", False):
                raise RuntimeError("runtime.camber_required=true but Camber is unavailable")
            return False
        return True

    def disk_check(self, where: str) -> None:
        du = disk_usage(self.paths.data_root)
        self.log.info("disk %s: free=%.2f GB (%s)", where, du["free_gb"], du["path"])
        if du["free_gb"] < float(self.cfg.runtime("min_free_gb_warn", 3.0)):
            self.log.warning("free space < %.1f GB", float(self.cfg.runtime("min_free_gb_warn", 3.0)))


def make_context(config: str | Config = DEFAULT_CONFIG, stage: str = "pipeline", force: bool = False,
                 dry_run: bool = False) -> Context:
    cfg = config if isinstance(config, Config) else load_config(config)
    paths = ProjectPaths.detect().ensure()
    load_camber_api_key(paths.env)
    log = setup_stage_logger(stage, paths.logs, cfg.runtime("log_level", "INFO"), bool(cfg.runtime("debug", False)))
    camber = CamberClient(paths, cfg.get("camber.stash_prefix"), cfg.runtime("camber_bin"), dry_run=dry_run)
    return Context(cfg=cfg, paths=paths, store=StateStore(paths.state), camber=camber, log=log,
                   force=force, dry_run=dry_run, commit=git_commit(paths.project_root))


# ================================================================ layout

class Layout:
    """Working locations derived from ProjectPaths (never hard-coded elsewhere)."""

    def __init__(self, paths: ProjectPaths):
        self.p = paths

    overlay = property(lambda s: s.p.working / "v6p5_overlay")
    synthetic_root = property(lambda s: s.overlay / "synthetic_foreign")
    synthetic_manifests = property(lambda s: s.p.working / "synthetic_manifests")
    qa_external = property(lambda s: s.p.qa / "v6p5_external")
    bg_pool = property(lambda s: s.p.parent / "train_background_pool")
    bg_pool_csv = property(lambda s: s.p.manifests / "parent_train_background_pool.csv")
    phash_partial = property(lambda s: s.p.parent / "parent_val_test_phash.partial.csv")
    oi_crops = property(lambda s: s.p.external / "openimages" / "crops")
    oi_manifest = property(lambda s: s.p.manifests / "open_images_crop_manifest.csv")
    mpcd_root = property(lambda s: s.p.external / "mpcd")
    mpcd_manifest = property(lambda s: s.p.manifests / "mpcd_direct_manifest.csv")
    pe_root = property(lambda s: s.p.external / "power_equipment")
    pe_manifest = property(lambda s: s.p.manifests / "power_equipment_direct_manifest.csv")
    pe_review = property(lambda s: s.p.manifests / "power_equipment_review_only_dx_sg.csv")
    synth_manifest = property(lambda s: s.p.manifests / "synthetic_foreign_manifest.csv")
    exact_csv = property(lambda s: s.p.manifests / "external_vs_internal_exact_duplicates.csv")
    phash_csv = property(lambda s: s.p.manifests / "external_vs_internal_phash_candidates.csv")

    def phash_cache(self, cfg) -> Path:
        return self.p.parent / cfg.get("parent_prepare.phash_cache_name")


# ================================================================ stages

PARENT_KEYS = ["taxonomy", "seed", "v6p2e", "parent_prepare"]


def stage_parent(ctx: Context) -> StageOutcome:
    from .dataset import v6p2e

    cfg, L = ctx.cfg, Layout(ctx.paths)
    inputs = [{"name": "v6p2e_tag", "path": v6p2e.parent_dir_remote(cfg), "sha256": cfg.get("v6p2e.tag")}]
    run, out = ctx.begin("parent", PARENT_KEYS, inputs)
    if out:
        return out
    try:
        remotes = v6p2e.metadata_remotes(cfg)
        freeze_p = ctx.camber.fetch(remotes["freeze"])
        manifest_p = ctx.camber.fetch(remotes["manifest"])
        freeze = json.loads(freeze_p.read_text())
        manifest = v6p2e.load_manifest(manifest_p)
        conflicts = v6p2e.validate_parent(cfg, freeze, manifest)
        if conflicts:
            raise v6p2e.ParentConflict("Frozen V6.2E artifacts conflict with config (artifact is "
                                       "authoritative; fix the config or investigate):\n- " + "\n- ".join(conflicts))
        ph = cfg.section_hash(PARENT_KEYS)
        if not ctx.compatible_partial("parent", ph, inputs):
            ctx.log.info("config changed: resetting partial parent checkpoints")
            shutil.rmtree(L.bg_pool, ignore_errors=True)
            L.phash_partial.unlink(missing_ok=True)
            L.phash_cache(cfg).unlink(missing_ok=True)

        arch = v6p2e.archive_remotes(cfg, freeze)
        base_p = ctx.camber.fetch(arch["base_archive"], bulk=True)
        overlay_p = ctx.camber.fetch(arch["overlay_archive"], bulk=True)
        recon_p = ctx.camber.fetch(remotes["reconciled_labels"], bulk=True)
        ctx.disk_check("after parent download")

        with v6p2e.ParentReader(base_p, overlay_p, recon_p) as reader:
            run.add_summary(stage_step="background_pool")
            pool, fails = v6p2e.build_background_pool(
                reader, manifest, cfg, L.bg_pool, ctx.paths.rel,
                tick=lambda n, t: run.progress(n, t, step="background_pool"))
            if len(pool) < int(cfg.get("parent_prepare.min_bg_pool")):
                raise RuntimeError(f"Too few structural backgrounds: {len(pool)}")
            atomic_write_csv(L.bg_pool_csv, pool)
            if len(fails):
                atomic_write_csv(ctx.paths.manifests / "parent_stream_failures.csv", fails)
            run.progress(len(pool), int(cfg.get("parent_prepare.bg_pool_max")), force=True, step="background_pool")

            ph_fails = []
            ph_remote = _phash_remote(cfg)
            if not L.phash_cache(cfg).exists() and ctx.camber.available() and ctx.camber.exists(ph_remote):
                ctx.log.info("VAL/TEST pHash cache found on Camber; downloading instead of recomputing")
                ctx.camber.fetch(ph_remote, L.phash_cache(cfg))
            if L.phash_cache(cfg).exists():
                ph_df = v6p2e.load_phash_cache(L.phash_cache(cfg))
                ctx.log.info("reuse VAL/TEST pHash cache (%d rows)", len(ph_df))
            else:
                ph_df, ph_fails = v6p2e.compute_parent_phash(
                    reader, manifest, cfg, L.phash_cache(cfg), L.phash_partial,
                    tick=lambda n, t: run.progress(n, t, step="val_test_phash"))
            if ph_fails:
                atomic_write_csv(ctx.paths.manifests / "parent_phash_failures.csv", pd.DataFrame(ph_fails))
        uploaded = 0
        if ctx.upload_enabled():
            uploaded = len(ctx.camber.put_many([(L.phash_cache(cfg), _phash_remote(cfg)),
                                                (L.bg_pool_csv, _phash_remote(cfg).rsplit("/", 1)[0]
                                                 + "/parent_train_background_pool.csv")]))

        if ctx.cfg.runtime("clean_parent_archives_after_qa", True):
            for p in (base_p, overlay_p, recon_p):
                if p.exists():
                    ctx.log.info("delete large parent archive: %s", ctx.paths.rel(p))
                    p.unlink()
        counts = manifest["split"].value_counts().to_dict()
        return _done(run, ctx, [
            ctx.rec("freeze", freeze_p, remotes["freeze"]),
            ctx.rec("manifest", manifest_p, remotes["manifest"]),
            ctx.rec("bg_pool_csv", L.bg_pool_csv),
            ctx.rec("phash_cache", L.phash_cache(cfg), _phash_remote(cfg)),
        ], tag=cfg.get("v6p2e.tag"), uploaded=uploaded, split_counts={k: int(v) for k, v in counts.items()},
            bg_pool=int(len(pool)), bg_pool_failures=int(len(fails)),
            phash_rows=int(len(ph_df)), phash_failures=int(len(ph_fails)))
    except BaseException as e:
        run.fail(e)
        raise


OI_KEYS = ["seed", "openimages", "synthetic.require_mask"]


def stage_openimages(ctx: Context) -> StageOutcome:
    from .dataset import openimages

    cfg, L = ctx.cfg, Layout(ctx.paths)
    run, out = ctx.begin("openimages", OI_KEYS, [])
    if out:
        return out
    try:
        if not ctx.compatible_partial("openimages", cfg.section_hash(OI_KEYS), []):
            shutil.rmtree(L.oi_crops, ignore_errors=True)
        classes = list(cfg.get("openimages.classes"))
        done = []

        def on_done(c):
            done.append(c)
            run.progress(len(done), len(classes), force=True, step="class", last=c)

        df = openimages.build_openimages_crops(cfg, ctx.paths.working, L.oi_crops, ctx.paths.rel, on_done)
        atomic_write_csv(L.oi_manifest, df)
        by = df.groupby("foreign_subtype").size().to_dict()
        return _done(run, ctx, [ctx.rec("crop_manifest", L.oi_manifest)],
                     crops=int(len(df)), crops_by_subtype={k: int(v) for k, v in by.items()},
                     annotation_download=cfg.get("openimages.annotation_download"),
                     segmentation_zip_downloaded=False)
    except BaseException as e:
        run.fail(e)
        raise


def _direct_stage(ctx: Context, stage: str, keys: list[str]) -> StageOutcome:
    from .dataset import direct_overlay, mpcd, power_equipment

    cfg, L = ctx.cfg, Layout(ctx.paths)
    run, out = ctx.begin(stage, keys, [])
    if out:
        return out
    try:
        extra_outputs, summary = [], {}
        if stage == "mpcd":
            L.mpcd_root.mkdir(parents=True, exist_ok=True)
            src_dir, zsha = mpcd.download_and_extract(cfg, L.mpcd_root)
            _provenance(L.mpcd_root, "MPCD.zip", zsha)
            records, source, manifest_path = mpcd.parse_mpcd(cfg, src_dir), mpcd.SOURCE, L.mpcd_manifest
        else:
            L.pe_root.mkdir(parents=True, exist_ok=True)
            src_dir, zsha = power_equipment.download_and_extract(cfg, L.pe_root, ctx.paths.cache / "hf_power_equipment")
            _provenance(L.pe_root, cfg.get("power_equipment.filename"), zsha)
            records, review = power_equipment.parse_power_equipment(cfg, src_dir)
            source, manifest_path = power_equipment.SOURCE, L.pe_manifest
            atomic_write_csv(L.pe_review, pd.DataFrame(review, columns=[
                "source", "source_image", "source_xml", "source_class", "review_role", "bbox_xyxy"]))
            extra_outputs.append(ctx.rec("review_only_dx_sg", L.pe_review))
            summary["dx_sg_review_only_boxes"] = len(review)
        direct_overlay.clear_source(L.overlay, source)
        df = direct_overlay.add_direct_records(records, L.overlay, cfg.names, ctx.paths.rel)
        atomic_write_csv(manifest_path, df)
        by = df.groupby(["target_split", "mapped_class_names"]).size()
        summary["images_by_split_class"] = {f"{a}/{b}": int(n) for (a, b), n in by.items()}
        summary["archive_sha256"] = read_json(_prov_path(src_dir.parent), {}).get("archive_sha256")
        return _done(run, ctx, [ctx.rec("direct_manifest", manifest_path), *extra_outputs],
                     images=int(len(df)), train=int((df["target_split"] == "train").sum()),
                     external_eval=int((df["target_split"] == "external_eval").sum()), **summary)
    except BaseException as e:
        run.fail(e)
        raise


def stage_mpcd(ctx: Context) -> StageOutcome:
    return _direct_stage(ctx, "mpcd", ["taxonomy", "mpcd"])


def stage_power_equipment(ctx: Context) -> StageOutcome:
    return _direct_stage(ctx, "power_equipment", ["taxonomy", "power_equipment"])


SYN_KEYS = ["seed", "taxonomy", "synthetic", "openimages", "parent_prepare", "v6p2e.tag"]


def stage_synthetic(ctx: Context) -> StageOutcome:
    from .dataset import synthetic

    cfg, L = ctx.cfg, Layout(ctx.paths)
    inputs = [ctx.input_from("parent", "bg_pool_csv"), ctx.input_from("openimages", "crop_manifest")]
    run, out = ctx.begin("synthetic", SYN_KEYS, inputs)
    if out:
        return out
    try:
        if not ctx.compatible_partial("synthetic", cfg.section_hash(SYN_KEYS), inputs) or ctx.force:
            synthetic.reset_synthetic(L.synthetic_root, L.synthetic_manifests)
        pool = read_csv_safe(ctx.output_path("parent", "bg_pool_csv"))
        crops = read_csv_safe(ctx.output_path("openimages", "crop_manifest"))
        anchors = synthetic.find_anchors(pool, int(cfg.get("synthetic.anchor_class")), ctx.paths.resolve)
        subtypes = sorted(crops["foreign_subtype"].unique())
        done = []

        def on_done(s):
            done.append(s)
            run.progress(len(done), len(subtypes), force=True, step="subtype", last=s)

        df = synthetic.build_synthetic(cfg, crops, anchors, L.synthetic_root, L.synthetic_manifests,
                                       ctx.paths.resolve, ctx.paths.rel, on_done)
        atomic_write_csv(L.synth_manifest, df)
        return _done(run, ctx, [ctx.rec("synthetic_manifest", L.synth_manifest)],
                     synthetic=int(len(df)), anchors=int(len(anchors)),
                     anchor_backgrounds=int(anchors["image"].nunique()),
                     by_subtype={k: int(v) for k, v in df["foreign_subtype"].value_counts().sort_index().items()},
                     train_only=True)
    except BaseException as e:
        run.fail(e)
        raise


DEDUP_KEYS = ["dedup", "parent_prepare.phash_splits", "parent_prepare.phash_hash_size", "v6p2e.tag"]


def stage_dedup(ctx: Context) -> StageOutcome:
    from .dataset import dedup, v6p2e

    cfg, L = ctx.cfg, Layout(ctx.paths)
    inputs = [ctx.input_from("parent", "manifest"), ctx.input_from("parent", "phash_cache"),
              ctx.input_from("mpcd", "direct_manifest"), ctx.input_from("power_equipment", "direct_manifest")]
    run, out = ctx.begin("dedup", DEDUP_KEYS, inputs)
    if out:
        return out
    try:
        direct = pd.concat([read_csv_safe(ctx.output_path("mpcd", "direct_manifest")),
                            read_csv_safe(ctx.output_path("power_equipment", "direct_manifest"))], ignore_index=True)
        images = [(r, ctx.paths.resolve(r)) for r in direct["overlay_image"]]
        manifest = v6p2e.load_manifest(ctx.output_path("parent", "manifest"))
        exact = dedup.exact_sha_audit(images, manifest)
        atomic_write_csv(L.exact_csv, exact)
        near = pd.DataFrame(columns=dedup.PHASH_COLUMNS)
        if cfg.get("dedup.run_phash"):
            near = dedup.phash_audit(images, v6p2e.load_phash_cache(ctx.output_path("parent", "phash_cache")),
                                     int(cfg.get("dedup.phash_max_hamming")), cfg.get("dedup.phash_band_widths"),
                                     int(cfg.get("parent_prepare.phash_hash_size")))
        atomic_write_csv(L.phash_csv, near)
        if len(exact):
            ctx.log.error("HARD GATE will fail at compose: %d exact SHA duplicate(s)", len(exact))
        if len(near):
            ctx.log.warning("%d pHash candidate(s) vs VAL/TEST must be reviewed", len(near))
        return _done(run, ctx, [ctx.rec("exact_duplicates", L.exact_csv), ctx.rec("phash_candidates", L.phash_csv)],
                     direct_images_audited=len(images), exact_duplicates=int(len(exact)),
                     phash_candidates=int(len(near)),
                     note="synthetic images use internal TRAIN backgrounds by design; not audited as leakage")
    except BaseException as e:
        run.fail(e)
        raise


EXT_KEYS = ["taxonomy", "seed", "v6p2e.tag", "v6p2e.expected_lineage", "parent_prepare", "openimages", "mpcd",
            "power_equipment", "synthetic", "dedup", "qa", "v6p5.remote"]


def stage_external_freeze(ctx: Context) -> StageOutcome:
    from .archive_utils import verify_yolo_tar
    from .dataset import external_stage as es
    from .qa import contacts, review as rv
    from .reporting.manifests import register_freeze

    cfg, L = ctx.cfg, Layout(ctx.paths)
    inputs = [ctx.input_from("parent", "bg_pool_csv"), ctx.input_from("openimages", "crop_manifest"),
              ctx.input_from("mpcd", "direct_manifest"), ctx.input_from("power_equipment", "direct_manifest"),
              ctx.input_from("power_equipment", "review_only_dx_sg"), ctx.input_from("synthetic", "synthetic_manifest"),
              ctx.input_from("dedup", "exact_duplicates"), ctx.input_from("dedup", "phash_candidates")]
    run, out = ctx.begin("external_freeze", EXT_KEYS, inputs)
    if out:
        return out
    try:
        direct = pd.concat([read_csv_safe(ctx.output_path("mpcd", "direct_manifest")),
                            read_csv_safe(ctx.output_path("power_equipment", "direct_manifest"))], ignore_index=True)
        synth = read_csv_safe(ctx.output_path("synthetic", "synthetic_manifest"))
        exact = read_csv_safe(L.exact_csv)
        near = read_csv_safe(L.phash_csv)
        _check_overlay_matches(ctx, direct, synth)

        qa = L.qa_external
        shutil.rmtree(qa, ignore_errors=True)
        qa.mkdir(parents=True)
        review = rv.build_review_manifest(direct, synth)
        rv.check_review_manifest(review)
        for name, src in [("direct_overlay_manifest.csv", None), ("synthetic_foreign_manifest.csv", None),
                          ("open_images_crop_manifest.csv", L.oi_manifest),
                          ("parent_train_background_pool.csv", L.bg_pool_csv),
                          ("power_equipment_review_only_dx_sg.csv", L.pe_review),
                          ("external_vs_internal_exact_duplicates.csv", L.exact_csv),
                          ("external_vs_internal_phash_candidates.csv", L.phash_csv)]:
            if src is not None:
                shutil.copyfile(src, qa / name)
        atomic_write_csv(qa / "direct_overlay_manifest.csv", direct)
        atomic_write_csv(qa / "synthetic_foreign_manifest.csv", synth)
        atomic_write_csv(qa / "v6p5_candidate_review_manifest.csv", review)
        _make_contacts(ctx, direct, synth, qa / "contacts", contacts.make_contact)

        tag, digest = es.overlay_tag(L.overlay)
        remote = es.StageRemote(cfg.get("v6p5.remote.stage_root"), tag)
        out_dir = ctx.camber.mirror_path(remote.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = es.package_stage(L.overlay, qa, out_dir, tag)

        direct_remote = {"mpcd": cfg.get("v6p5.remote.mpcd_direct_archive"),
                         "power_equipment": cfg.get("v6p5.remote.power_equipment_direct_archive")}
        direct_files = {}
        for key, source in (("mpcd", "MPCD"), ("power_equipment", "PowerEquipment")):
            fixed = ctx.camber.mirror_path(direct_remote[key])
            es.build_direct_archive(L.overlay, direct, source, fixed, ctx.paths.resolve)
            tagged = out_dir / "direct" / Path(direct_remote[key]).name
            tagged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed, tagged)
            verify_yolo_tar(fixed, len(cfg.names)).raise_if_bad(fixed.name)
            direct_files[key] = (fixed, tagged)

        provenance = {"mpcd_zip_sha256": ctx.summary("mpcd").get("archive_sha256"),
                      "power_equipment_zip_sha256": ctx.summary("power_equipment").get("archive_sha256")}
        freeze = es.build_freeze(cfg, tag, digest, direct, synth, review, exact, near, provenance,
                                 cfg.section_hash(EXT_KEYS), ctx.commit)
        freeze["artifacts"] = {
            "overlay": {"remote": remote.overlay, "sha256": sha256_file(files["overlay"])},
            "qa": {"remote": remote.qa, "sha256": sha256_file(files["qa"])},
            "direct_archives": {k: {"remote": direct_remote[k], "tagged_remote": f"{remote.dir}/direct/{v[1].name}",
                                    "sha256": sha256_file(v[0])} for k, v in direct_files.items()},
        }
        freeze_path = out_dir / f"{tag}.freeze.json"
        if freeze_path.exists():
            # Immutable: same tag == same overlay content. Keep the first freeze.
            ctx.log.info("freeze for %s already exists; kept unchanged", tag)
            freeze = read_json(freeze_path)
        else:
            atomic_write_json(freeze_path, freeze)
        register_freeze(freeze_path, ctx.paths.freezes)

        uploaded = []
        if ctx.upload_enabled():
            pairs = [(files["overlay"], remote.overlay), (files["qa"], remote.qa), (freeze_path, remote.freeze)]
            for key, (fixed, tagged) in direct_files.items():
                pairs += [(tagged, f"{remote.dir}/direct/{tagged.name}"), (fixed, direct_remote[key])]
            uploaded = ctx.camber.put_many(pairs, required=False)

        outputs = [ctx.rec("freeze", freeze_path, remote.freeze), ctx.rec("overlay", files["overlay"], remote.overlay),
                   ctx.rec("qa_zip", files["qa"], remote.qa), ctx.rec("review_manifest", qa / "v6p5_candidate_review_manifest.csv")]
        outputs += [ctx.rec(f"direct_{k}", v[1], f"{remote.dir}/direct/{v[1].name}") for k, v in direct_files.items()]
        return _done(run, ctx, outputs, tag=tag, overlay_sha256=digest, counts=freeze["counts"],
                     review_candidates=int(len(review)), exact_duplicates=int(len(exact)),
                     phash_candidates=int(len(near)), uploaded=len(uploaded), remote=remote.dir)
    except BaseException as e:
        run.fail(e)
        raise


COMPOSE_KEYS = ["taxonomy", "compose.auto_approve_recommended", "compose.allow_phash_candidates",
                "compose.allow_exact_duplicates", "v6p5.remote"]


def stage_compose(ctx: Context) -> StageOutcome:
    from .archive_utils import verify_yolo_tar
    from .dataset import compose as cp
    from .dataset.external_stage import StageRemote
    from .qa import review as rv
    from .reporting.manifests import register_freeze

    cfg = ctx.cfg
    stage_tag = _select_stage_tag(ctx)
    srem = StageRemote(cfg.get("v6p5.remote.stage_root"), stage_tag)
    decisions_path = ctx.paths.resolve(cfg.get("compose.decisions_file"))
    _pull_decisions(ctx, stage_tag, decisions_path)
    inputs = [{"name": "stage_tag", "sha256": stage_tag},
              {"name": "decisions", "path": ctx.paths.rel(decisions_path),
               "sha256": sha256_file(decisions_path) if decisions_path.exists() else None}]
    run, out = ctx.begin("compose", COMPOSE_KEYS, inputs)
    if out:
        return out
    try:
        freeze_p = ctx.camber.fetch(srem.freeze)
        qa_p = ctx.camber.fetch(srem.qa)
        overlay_p = ctx.camber.fetch(srem.overlay, bulk=False)
        stage_freeze = json.loads(freeze_p.read_text())
        extract = ctx.paths.working / "compose_qa_extract" / stage_tag
        shutil.rmtree(extract, ignore_errors=True)
        import zipfile

        with zipfile.ZipFile(qa_p) as z:
            z.extractall(extract)
        need = ["v6p5_candidate_review_manifest.csv", "direct_overlay_manifest.csv", "synthetic_foreign_manifest.csv"]
        missing = [n for n in need if not (extract / n).exists()]
        if missing:
            raise RuntimeError("Stage QA ZIP missing: " + ", ".join(missing))
        review = read_csv_safe(extract / "v6p5_candidate_review_manifest.csv")
        exact = read_csv_safe(extract / "external_vs_internal_exact_duplicates.csv")
        near = read_csv_safe(extract / "external_vs_internal_phash_candidates.csv")
        rv.check_review_manifest(review)
        _ensure_direct_archives(ctx, stage_freeze, srem)

        try:
            cp.hard_gates(stage_freeze, cfg.names, exact, near, bool(cfg.get("compose.allow_exact_duplicates")),
                          bool(cfg.get("compose.allow_phash_candidates")))
        except cp.PHashReviewRequired as e:
            atomic_write_csv(ctx.paths.compose / "phash_candidates_to_review.csv", near)
            _write_decisions_template(ctx, review, decisions_path)
            _push_decisions(ctx, stage_tag, decisions_path)
            run.needs_review(pending_reason="phash", phash_candidates=e.n, stage_tag=stage_tag,
                             exact_duplicates=int(len(exact)))
            return StageOutcome("compose", "needs_review", str(e), run.state.summary)

        decisions = _write_decisions_template(ctx, review, decisions_path)
        _push_decisions(ctx, stage_tag, decisions_path)
        res = rv.resolve_decisions(review, decisions, bool(cfg.get("compose.auto_approve_recommended")))
        if not res.complete:
            atomic_write_csv(ctx.paths.compose / "pending_review.csv", res.pending_train)
            run.needs_review(pending_reason="decisions", pending_train=int(len(res.pending_train)),
                             review_total=int(len(review)), stage_tag=stage_tag,
                             decisions_file=ctx.paths.rel(decisions_path),
                             exact_duplicates=int(len(exact)), phash_candidates=int(len(near)))
            return StageOutcome("compose", "needs_review",
                                f"{len(res.pending_train)} TRAIN candidate(s) without KEEP/DROP in "
                                f"{ctx.paths.rel(decisions_path)}", run.state.summary)
        resolved = res.resolved
        is_train = resolved["target_split"].astype(str).str.lower().eq("train")
        keep = resolved[is_train & resolved["decision"].eq("KEEP")]
        keep_synth = keep[keep["candidate_type"] == "synthetic_relation"]
        keep_direct = keep[keep["candidate_type"] == "direct_powerline"]
        if len(keep_synth) == 0:
            raise RuntimeError("No synthetic_relation TRAIN candidate was KEEP.")

        tmp_dir = ctx.paths.working / "compose_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_arch = tmp_dir / "approved_synthetic_train.tar.gz"
        included = cp.build_approved_synthetic(overlay_p, keep_synth, tmp_arch)
        seed = cp.compose_seed(stage_tag, stage_freeze, resolved, tmp_arch, cfg.names)
        tag = cp.compose_tag_from_seed(seed)
        crem = cp.ComposeRemote(cfg.get("v6p5.remote.compose_root"), tag)
        out_dir = ctx.camber.mirror_path(crem.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        syn_final = out_dir / Path(crem.synthetic).name
        shutil.move(str(tmp_arch), syn_final)
        rep = verify_yolo_tar(syn_final, len(cfg.names))
        rep.raise_if_bad(syn_final.name)
        if rep.images != len(keep_synth) or set(rep.splits) != {"train"}:
            raise RuntimeError(f"Synthetic archive check failed: images={rep.images} splits={rep.splits}")

        resolved_p = out_dir / Path(crem.resolved).name
        syn_manifest_p = out_dir / Path(crem.synthetic_manifest).name
        direct_manifest_p = out_dir / Path(crem.direct_manifest).name
        atomic_write_csv(resolved_p, resolved)
        atomic_write_csv(syn_manifest_p, included)
        dkeep = keep_direct.assign(overlay_basename=keep_direct["overlay_image"].map(lambda s: Path(str(s)).name))
        atomic_write_csv(direct_manifest_p, dkeep[["review_id", "source", "overlay_basename", "mapped_class_names"]])

        train_rows = resolved[is_train]
        qa_summary = {
            "compose_tag": tag, "source_stage_tag": stage_tag, "review_total": int(len(resolved)),
            "train_candidates": int(len(train_rows)), "train_keep": int(len(keep)),
            "train_drop": int((train_rows["decision"] == "DROP").sum()),
            "approved_synthetic": int(len(keep_synth)), "approved_direct": int(len(keep_direct)),
            "exact_duplicate_candidates": int(len(exact)), "phash_candidates": int(len(near)),
            "allow_phash_candidates": bool(cfg.get("compose.allow_phash_candidates")),
            "auto_approve_recommended": bool(cfg.get("compose.auto_approve_recommended")),
        }
        qa_summary_p = atomic_write_json(out_dir / Path(crem.qa_summary).name, qa_summary)
        direct_archives = {"mpcd": cfg.get("v6p5.remote.mpcd_direct_archive"),
                           "power_equipment": cfg.get("v6p5.remote.power_equipment_direct_archive")}
        index = cp.build_index(tag, stage_tag, srem, crem, direct_archives)
        tagged = (stage_freeze.get("artifacts") or {}).get("direct_archives") or {}
        index["direct_archives_tagged"] = {k: v.get("tagged_remote") for k, v in tagged.items()}
        cp.validate_index(index)
        index_p = atomic_write_json(out_dir / Path(crem.index).name, index)
        freeze = {
            "tag": tag, "type": "v6p5_compose_qa", "source_stage_tag": stage_tag,
            "source_stage_freeze": stage_freeze, "taxonomy": {i: n for i, n in enumerate(cfg.names)},
            "counts": qa_summary,
            "policy": {
                "test_modified": False, "val_modified": False, "external_eval_in_training": False,
                "synthetic_train_only": True, "review_resolved": True,
                "auto_approve_recommended": bool(cfg.get("compose.auto_approve_recommended")),
                "phash_candidates_explicitly_allowed": bool(cfg.get("compose.allow_phash_candidates")),
                "direct_rows_filtered_by_review": True,
            },
            "artifacts": {
                "synthetic_archive": {"remote": crem.synthetic, "sha256": sha256_file(syn_final)},
                "resolved_review_manifest": {"remote": crem.resolved, "sha256": sha256_file(resolved_p)},
                "approved_synthetic_manifest": {"remote": crem.synthetic_manifest, "sha256": sha256_file(syn_manifest_p)},
                "approved_direct_manifest": {"remote": crem.direct_manifest, "sha256": sha256_file(direct_manifest_p)},
                "qa_summary": {"remote": crem.qa_summary, "sha256": sha256_file(qa_summary_p)},
                "index": {"remote": crem.index, "sha256": sha256_file(index_p)},
            },
            "compose_seed": seed,
            "config_hash": cfg.section_hash(COMPOSE_KEYS),
            "code_commit": ctx.commit,
        }
        freeze_out = out_dir / Path(crem.freeze).name
        if freeze_out.exists():
            ctx.log.info("compose freeze %s already exists; kept unchanged", tag)
        else:
            atomic_write_json(freeze_out, freeze)
        register_freeze(freeze_out, ctx.paths.freezes)

        uploaded = 0
        if ctx.upload_enabled():
            existing = ctx.camber.discover_tags(cfg.get("v6p5.remote.compose_root"), cp.COMPOSE_PREFIX)
            if tag in existing and not cfg.get("compose.allow_existing_compose_tag"):
                ctx.log.info("%s already on Camber (content-addressed); not re-uploading", tag)
            else:
                pairs = [(syn_final, crem.synthetic), (resolved_p, crem.resolved), (syn_manifest_p, crem.synthetic_manifest),
                         (direct_manifest_p, crem.direct_manifest), (qa_summary_p, crem.qa_summary),
                         (index_p, crem.index), (freeze_out, crem.freeze)]
                uploaded = len(ctx.camber.put_many(pairs, required=False))
                listing = ctx.camber.ls(crem.dir, recursive=True)
                miss = [Path(x).name for x in (crem.index, crem.freeze, crem.synthetic) if Path(x).name not in listing.stdout]
                if miss:
                    ctx.log.warning("server-side verify: not listed yet: %s", miss)
        outs = [ctx.rec("index", index_p, crem.index), ctx.rec("freeze", freeze_out, crem.freeze),
                ctx.rec("synthetic_archive", syn_final, crem.synthetic),
                ctx.rec("resolved_review", resolved_p, crem.resolved),
                ctx.rec("approved_direct_manifest", direct_manifest_p, crem.direct_manifest)]
        return _done(run, ctx, outs, tag=tag, stage_tag=stage_tag, uploaded=uploaded, **qa_summary)
    except BaseException as e:
        run.fail(e)
        raise


TRAIN_KEYS = ["taxonomy", "training", "v6p2e.tag", "v6p5.remote.train_root"]


def stage_train(ctx: Context) -> StageOutcome:
    from .dataset import v6p2e
    from .dataset.compose import ComposeRemote, validate_index
    from .reporting.compare import add_deltas, promotion_gate
    from .reporting.manifests import experiment_manifest, write_once_json
    from .training import dataset_builder as db
    from .training import metrics, ultralytics_runner as ur
    from .training.checkpoint import RunState, train_hash

    cfg = ctx.cfg
    t = cfg.get("training")
    if cfg.get("future.blocked_until", None):
        raise RuntimeError(f"Config '{cfg.name}' is blocked until: {cfg.get('future.blocked_until')}")
    compose_tag = _select_compose_tag(ctx)
    inputs = [{"name": "compose_tag", "sha256": compose_tag}]
    run, out = ctx.begin(t["experiment"], TRAIN_KEYS, inputs)
    if out:
        return out
    try:
        crem = ComposeRemote(cfg.get("v6p5.remote.compose_root"), compose_tag)
        index = json.loads(ctx.camber.fetch(crem.index).read_text())
        validate_index(index)

        # ---- dataset
        ds_root = ctx.paths.working / "datasets" / f"{t['experiment']}__{compose_tag}"
        if ctx.force and ds_root.exists():
            shutil.rmtree(ds_root)
        meta = v6p2e.metadata_remotes(cfg)
        manifest = v6p2e.load_manifest(ctx.camber.fetch(meta["manifest"]))
        freeze = json.loads(ctx.camber.fetch(meta["freeze"]).read_text())
        conflicts = v6p2e.validate_parent(cfg, freeze, manifest)
        if conflicts:
            raise v6p2e.ParentConflict("\n".join(conflicts))
        arch = v6p2e.archive_remotes(cfg, freeze)
        run.add_summary(step="dataset")
        records = []
        parent_done = (ds_root / ".parent_complete").exists()
        if not parent_done:
            base_p = ctx.camber.fetch(arch["base_archive"], bulk=True)
            overlay_p = ctx.camber.fetch(arch["overlay_archive"], bulk=True)
            recon_p = ctx.camber.fetch(meta["reconciled_labels"], bulk=True)
            with v6p2e.ParentReader(base_p, overlay_p, recon_p) as reader:
                records += db.add_parent_rows(reader, manifest, t["splits"], ds_root,
                                              tick=lambda n, tot: run.progress(n, tot, step="dataset_parent"))
            atomic_write_csv(ds_root / "parent_records.csv", pd.DataFrame(records))
            (ds_root / ".parent_complete").write_text("ok\n")
            if ctx.cfg.runtime("clean_parent_archives_after_qa", True):
                for p in (base_p, overlay_p, recon_p):
                    p.unlink(missing_ok=True)
        else:
            records += read_csv_safe(ds_root / "parent_records.csv").to_dict("records")

        keep_names = None
        if t["direct_filter_by_review"]:
            dm = read_csv_safe(ctx.camber.fetch(index["approved_direct_manifest"]))
            keep_names = set(dm["overlay_basename"].astype(str)) if len(dm) else set()
        tagged = index.get("direct_archives_tagged") or {}
        for key, arc_root in (("mpcd", "mpcd_broken_strand"), ("power_equipment", "power_equipment_stage")):
            remote = tagged.get(key) or index["direct_archives"][key]
            records += db.add_direct_archive(ctx.camber.fetch(remote), arc_root, ds_root, keep_names)
        records += db.add_synthetic_archive(ctx.camber.fetch(index[t["compose_index_synthetic_field"]]), ds_root)
        dataset = db.finalize(ds_root, records, manifest, cfg.names, compose_tag)
        run.add_summary(dataset_tag=dataset.tag, dataset_counts=dataset.counts)
        ctx.log.info("dataset %s: %s", dataset.tag, dataset.counts)

        # ---- training
        rtag = ur.remote_run_tag(t, compose_tag, dataset.counts["train_total"])
        remote_root = f"{cfg.get('v6p5.remote.train_root')}/{rtag}"
        run_name = f"{t['experiment']}__{compose_tag}"
        run_dir = ctx.paths.runs / t["run_family"] / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        th = train_hash(t, dataset.tag, compose_tag)
        write_once_json(run_dir / "experiment_manifest.json",
                        experiment_manifest(cfg, ctx.paths, dataset, compose_tag, run_name, th, remote_root, ctx.commit))
        run.add_summary(step="train", run_dir=ctx.paths.rel(run_dir), remote_root=remote_root)
        run.progress(0, int(t["epochs"]), force=True, step="train")
        camber = ctx.camber if ctx.upload_enabled() else None
        state = ur.train(cfg, dataset.data_yaml, run_dir, dataset.tag, compose_tag, remote_root, camber, ctx.force,
                         session={"code_commit": ctx.commit, "code_version": __version__, **ctx.meta()})

        # ---- strict VAL evaluation (TEST never)
        ev = metrics.evaluate_weights(run_dir / "weights/best.pt", dataset.data_yaml, int(t["imgsz"]),
                                      int(t["eval_batch"]), t["device"], cfg.names)
        report = {"experiment": "V6.2E + V6.5 external candidate data ablation", "compose_tag": compose_tag,
                  "dataset_tag": dataset.tag, "model": t["model_label"], "imgsz": t["imgsz"], "epochs": t["epochs"],
                  "base_train_images": dataset.counts.get("train/v6p2e", 0),
                  "added_direct_train_images": dataset.counts.get("train/v6p5_direct", 0),
                  "added_synthetic_train_images": dataset.counts.get("train/v6p5_synthetic", 0),
                  "final_train_images": dataset.counts["train_total"], "val_images": dataset.counts["val_total"],
                  **ev}
        report = add_deltas(report, cfg.get("baseline"))
        report["promotion_gate"] = promotion_gate(report, cfg.get("baseline"), cfg.get("promotion_gate"))
        report_p = ur.write_report(run_dir, report)
        state = RunState.load(run_dir / "run_state.json")
        state.status = "evaluated"
        state.save(run_dir / "run_state.json")
        done = atomic_write_json(run_dir / "done.json", {"tag": rtag, "remote_root": remote_root,
                                                          "promotion_gate": report["promotion_gate"]})
        if camber:
            ur.sync_run(camber, run_dir, remote_root, ur.SYNC_FILES + (
                report_p.name, "v6p5_yolo11n_strict_iou_per_class.csv", "done.json"))
            ur.sync_run(camber, run_dir, remote_root, ("weights/best.pt",))
        return _done(run, ctx, [ctx.rec("report", report_p), ctx.rec("best", run_dir / "weights/best.pt", f"{remote_root}/best.pt"),
                                ctx.rec("done", done)],
                     mAP50=report["mAP50"], mAP50_95=report["mAP50_95"], gap=report["gap"],
                     promotion_gate=report["promotion_gate"], remote_root=remote_root, run_tag=rtag,
                     last_completed_epoch=state.last_completed_epoch)
    except BaseException as e:
        run.fail(e)
        raise


def stage_evaluate(ctx: Context) -> StageOutcome:
    from .reporting.compare import comparison_table

    t = ctx.cfg.get("training")
    run, out = ctx.begin("evaluate", ["baseline", "promotion_gate", "training.experiment"],
                         [ctx.input_from(t["experiment"], "report")])
    if out:
        return out
    try:
        report = read_json(ctx.output_path(t["experiment"], "report"))
        table = comparison_table(report, ctx.cfg.get("baseline"))
        out_csv = atomic_write_csv(ctx.paths.exports / f"{t['experiment']}_vs_baseline.csv", table)
        remote_root = ctx.summary(t["experiment"]).get("remote_root")
        if remote_root and ctx.upload_enabled():
            ctx.camber.put_many([(out_csv, f"{remote_root}/{out_csv.name}")])
        gate = report.get("promotion_gate", {})
        return _done(run, ctx, [ctx.rec("comparison", out_csv)], promotion_gate=gate,
                     candidate_pass=bool(gate.get("candidate_pass")),
                     note="Gate pass only allows human review/promotion; this is a data ablation, not a final benchmark.")
    except BaseException as e:
        run.fail(e)
        raise


# ============================================================== helpers

def _done(run: StageRun, ctx: Context, outputs, **summary) -> StageOutcome:
    st = run.complete(outputs, **summary)
    return StageOutcome(st.stage, "complete", "ok", st.summary)


def _prov_path(root: Path) -> Path:
    return root / "PROVENANCE.json"


def _provenance(root: Path, archive: str, sha: str | None) -> None:
    if sha:
        atomic_write_json(_prov_path(root), {"archive": archive, "archive_sha256": sha})


def _check_overlay_matches(ctx: Context, direct: pd.DataFrame, synth: pd.DataFrame) -> None:
    """The overlay dir must contain exactly the manifest files (it defines the tag)."""
    L = Layout(ctx.paths)
    expected = set()
    for col in ("overlay_image", "overlay_label"):
        expected |= {ctx.paths.resolve(p).resolve() for p in direct[col]}
        expected |= {ctx.paths.resolve(p).resolve() for p in synth[col]}
    actual = {p.resolve() for p in L.overlay.rglob("*") if p.is_file()}
    if expected != actual:
        raise RuntimeError(f"Overlay dir out of sync with manifests: missing={len(expected - actual)} "
                           f"stray={len(actual - expected)} (e.g. {sorted(map(str, actual - expected))[:5]}). "
                           "Re-run the source stages with --force.")


def _make_contacts(ctx: Context, direct: pd.DataFrame, synth: pd.DataFrame, out: Path, make) -> None:
    names = ctx.cfg.names
    per_src = int(ctx.cfg.get("qa.max_contacts_per_source"))
    for source in sorted(direct["source"].unique()):
        for _, r in direct[direct["source"] == source].head(per_src).iterrows():
            rows = read_yolo_label(ctx.paths.resolve(r["overlay_label"]))
            make(ctx.paths.resolve(r["overlay_image"]), [yolo_to_xyxy(*x[1:]) for x in rows],
                 [names[x[0]] for x in rows], out / f"{source}_{Path(r['overlay_image']).stem}.jpg",
                 f"{source} | {r['target_split']} | {r['mapped_class_names']}")
    for _, r in synth.head(int(ctx.cfg.get("qa.max_synth_contacts"))).iterrows():
        rows = [x for x in read_yolo_label(ctx.paths.resolve(r["overlay_label"])) if x[0] == int(ctx.cfg.get("synthetic.foreign_class"))]
        make(ctx.paths.resolve(r["overlay_image"]), [yolo_to_xyxy(*x[1:]) for x in rows],
             [f"foreign_object:{r['foreign_subtype']}"] * len(rows),
             out / f"synthetic_{Path(r['overlay_image']).stem}.jpg", f"synthetic relation | {r['foreign_subtype']}")


def _select_stage_tag(ctx: Context) -> str:
    from .dataset.external_stage import EXT_PREFIX

    explicit = ctx.cfg.get("compose.stage_tag", None)
    if explicit:
        return str(explicit)
    local = ctx.summary("external_freeze").get("tag")
    st = ctx.store.load("external_freeze")
    if local and st and st.status == "complete":
        return local
    if not ctx.camber.available():
        raise RuntimeError("No local external_freeze result and Camber is unavailable to discover v6p5_ext_*.")
    tags = ctx.camber.discover_tags(ctx.cfg.get("v6p5.remote.stage_root"), EXT_PREFIX)
    if len(tags) != 1:
        raise RuntimeError(f"Expected exactly one v6p5_ext_* on Camber, found {tags}. Set compose.stage_tag.")
    return tags[0]


def _select_compose_tag(ctx: Context) -> str:
    from .dataset.compose import COMPOSE_PREFIX

    explicit = ctx.cfg.get("training.compose_tag", None)
    if explicit:
        return str(explicit)
    st = ctx.store.load("compose")
    if st and st.status == "complete" and st.summary.get("tag"):
        return st.summary["tag"]
    if not ctx.camber.available():
        raise RuntimeError("No complete local compose stage and Camber unavailable. Run compose first "
                           "(do not invent a compose tag).")
    tags = ctx.camber.discover_tags(ctx.cfg.get("v6p5.remote.compose_root"), COMPOSE_PREFIX)
    if len(tags) != 1:
        raise RuntimeError(f"Compose candidates: {tags}. Need exactly one, or set training.compose_tag. "
                           "Run the compose stage first if empty.")
    return tags[0]


def _write_decisions_template(ctx: Context, review: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Create/extend the editable decisions CSV without touching existing decisions."""
    from .qa import review as rv

    existing = read_csv_safe(path)
    if len(existing) and "overlay_image" in existing.columns:
        want = dict(zip(review["review_id"].astype(str), review["overlay_image"].map(lambda s: Path(str(s)).name)))
        got = dict(zip(existing["review_id"].astype(str), existing["overlay_image"].map(lambda s: Path(str(s)).name)))
        clash = [k for k, v in got.items() if k in want and want[k] != v]
        if clash:
            raise RuntimeError(f"{path} belongs to a different external stage (review_id/image mismatch, "
                               f"e.g. {clash[:3]}). Move it aside before reviewing this stage.")
    tpl = rv.decisions_template(review, existing if len(existing) else None)
    if not path.exists() or len(tpl) != len(existing):
        atomic_write_csv(path, tpl)
        ctx.log.info("decisions file ready for review: %s", ctx.paths.rel(path))
    return read_csv_safe(path)


def _phash_remote(cfg) -> str:
    return (f"{cfg.get('v6p5.remote.stage_root')}/parent_cache/{cfg.get('v6p2e.tag')}/"
            f"{cfg.get('parent_prepare.phash_cache_name')}")


def _decisions_remote(cfg, stage_tag: str) -> str:
    return f"{cfg.get('v6p5.remote.compose_root')}/review_decisions/{stage_tag}/v6p5_review_decisions.csv"


def _pull_decisions(ctx: Context, stage_tag: str, path: Path) -> None:
    """Local file is where you edit; Camber holds the shared copy (Kaggle <-> local)."""
    if path.exists() or not ctx.camber.available():
        return
    remote = _decisions_remote(ctx.cfg, stage_tag)
    if ctx.camber.exists(remote):
        ctx.camber.fetch(remote, path, reuse=False)
        ctx.log.info("review decisions restored from Camber: %s", remote)


def _push_decisions(ctx: Context, stage_tag: str, path: Path) -> None:
    if path.exists() and ctx.upload_enabled():
        ctx.camber.put_many([(path, _decisions_remote(ctx.cfg, stage_tag))])


def _ensure_direct_archives(ctx: Context, stage_freeze: dict, srem) -> None:
    """Legacy repair step: fixed direct archives must exist for the train worker."""
    for key in ("mpcd_direct_archive", "power_equipment_direct_archive"):
        remote = ctx.cfg.get(f"v6p5.remote.{key}")
        local = ctx.camber.mirror_path(remote)
        if local.exists():
            continue
        if ctx.camber.available() and ctx.camber.exists(remote):
            continue
        raise RuntimeError(f"Direct archive missing locally and remotely: {remote}. "
                           "Re-run external_freeze (it rebuilds them from the stage overlay).")


# ============================================================= registry

@dataclass(frozen=True)
class StageSpec:
    name: str
    title: str
    func: Callable[[Context], StageOutcome]
    depends: tuple[str, ...] = ()


STAGES: dict[str, StageSpec] = {s.name: s for s in [
    StageSpec("parent", "V6.2E parent (pool + VAL/TEST pHash)", stage_parent),
    StageSpec("openimages", "Open Images V7 masked crops", stage_openimages),
    StageSpec("mpcd", "MPCD direct stage", stage_mpcd),
    StageSpec("power_equipment", "PowerEquipment direct stage", stage_power_equipment),
    StageSpec("synthetic", "Relation-aware synthetic foreign objects", stage_synthetic, ("parent", "openimages")),
    StageSpec("dedup", "Exact SHA + pHash leakage QA", stage_dedup, ("parent", "mpcd", "power_equipment")),
    StageSpec("external_freeze", "V6.5 external freeze (v6p5_ext_*)", stage_external_freeze,
              ("parent", "openimages", "mpcd", "power_equipment", "synthetic", "dedup")),
    StageSpec("compose", "Compose QA review gate (v6p5_compose_*)", stage_compose),
    StageSpec("yolo11n_ablation", "YOLO11n@640 data ablation", stage_train),
    StageSpec("evaluate", "Evaluate vs V6.2E baseline", stage_evaluate),
]}

STAGE_CONFIG = {"yolo11n_ablation": "yolo11n_ablation", "evaluate": "yolo11n_ablation"}


def load(stage_tag: str | None = None, compose_tag: str | None = None, parent_tag: str | None = None,
         config: str = DEFAULT_CONFIG) -> Config:
    """Config with the dataset names a notebook selected (None = automatic)."""
    cfg = load_config(config)
    if parent_tag and parent_tag != cfg.get("v6p2e.tag"):
        raise ValueError(f"Parent dataset is immutable: {cfg.get('v6p2e.tag')} (got {parent_tag})")
    overrides: dict = {}
    if stage_tag:
        overrides.setdefault("compose", {})["stage_tag"] = stage_tag
    if compose_tag:
        overrides.setdefault("training", {})["compose_tag"] = compose_tag
    return load_config(config, overrides=overrides) if overrides else cfg


def verify_on_camber(stage: str, config: str | Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Which outputs of a stage are really stored on Camber (read-only check)."""
    ctx = make_context(config, stage="verify")
    st = ctx.store.load(stage)
    rows = []
    for o in (st.outputs if st else []):
        if not o.get("remote"):
            continue
        rows.append({"output": o["name"], "camber": o["remote"],
                     "on_camber": ctx.camber.exists(o["remote"]) if ctx.camber.available() else None})
    return pd.DataFrame(rows, columns=["output", "camber", "on_camber"])


def run_stage(name: str, config: str | Config | None = None, force: bool = False,
              dry_run: bool = False) -> StageOutcome:
    if name not in STAGES:
        raise KeyError(f"Unknown stage {name!r}. Known: {list(STAGES)}")
    ctx = make_context(config or STAGE_CONFIG.get(name, DEFAULT_CONFIG), stage=name, force=force, dry_run=dry_run)
    ctx.log.info("stage=%s env=%s home=%s config=%s", name, ctx.paths.env.value, ctx.paths.home, ctx.cfg.name)
    outcome = STAGES[name].func(ctx)
    ctx.log.info("%s", outcome)
    return outcome


def run_external_pipeline(config: str | Config | None = None, force: bool = False, dry_run: bool = False) -> list[StageOutcome]:
    """parent -> sources -> synthetic -> dedup -> external_freeze, stopping on first failure."""
    order = ["parent", "openimages", "mpcd", "power_equipment", "synthetic", "dedup", "external_freeze"]
    return [run_stage(s, config, force=force, dry_run=dry_run) for s in order]
