"""Analyze the frozen CSL fixed40/local first20 comparison without refitting.

The primary statistic and clustered bootstrap are executed from the exact
preregistered source snapshot.  This adapter adds fail-closed provenance,
fixed20 coverage/diversity diagnostics, an independent pre-analysis gate, and
a separate-process scientific repeat.  It never launches a recognizer.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict, defaultdict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "outputs/revision/nonhuman_closure_20260907"
ANALYSIS_DIR = REVISION / "csl_full_method_paired_analysis_v10"
LOCAL_DIR = (
    REVISION
    / "csl_full_method_local_comparator_recovery_v9/local_comparator"
)
CANDIDATE_DIR = REVISION / "csl_full_method_dev_candidates_v2"

PREREG = REVISION / "csl_full_method_prereg_v2/preregistration.json"
PREREG_SEAL = REVISION / "csl_full_method_prereg_v2/seal.json"
METHOD_SOURCE = (
    REVISION / "csl_full_method_resource_gate_v3/source_v2_audit.py"
)
METRIC_HELPER = ROOT / "scripts/reduced_learned_csl_route.py"
NORMALIZER_SOURCE = ROOT / "scripts/build_retrieval_gloss_plans.py"
NATIVE_METRICS = ROOT / "external/baselines/MSKA/metrics.py"
DEV_MANIFEST = ROOT / "data/csl-daily/csl_daily_dev.json"
CANONICAL_DEV = ROOT / "external/baselines/MSKA/data/CSL-Daily/CSL-Daily.dev"
QUERIES_DEV = REVISION / "csl_full_method_prereg_v2/queries_dev.json"
JOBS = REVISION / "csl_full_method_dev_score_v5/jobs.json"
CANDIDATE_REVIEW = (
    REVISION / "csl_full_method_recovery_v8/candidate/independent_review.json"
)
CANDIDATE_MERGED = (
    REVISION / "csl_full_method_recovery_v8/candidate/merged_result.json"
)
CANDIDATE_STAGE = CANDIDATE_DIR / "stage.json"
CANDIDATE_SEAL = CANDIDATE_DIR / "seal.json"
CANDIDATE_CERTIFICATES = CANDIDATE_DIR / "certificates.json"
CANDIDATE_CONSTRUCTION = CANDIDATE_DIR / "construction.json"
CANDIDATE_ROUTES = CANDIDATE_DIR / "routes.json"
V9_WRAPPER = ROOT / "scripts/run_csl_paired_local_recovery_v9.py"
CANARY_REVIEW = LOCAL_DIR / "canary_review.json"
SHARED_LOCK = (
    REVISION / "csl_full_method_resource_gate_v3/shared_heavy.lock"
)

PINS = {
    PREREG: "7f88ec967938cc0cb4a06c730b187251382238a9755b35c1bb52fbef9359c246",
    PREREG_SEAL: "ab9ecc2ba64578ee6a07dfb428d4461d49ba25d7946756284519a0b0d54b6224",
    METHOD_SOURCE: "818d8893a5bf300cb123e6c61298e0b6e4740cdf1fbd9361b180f95a24b5427e",
    METRIC_HELPER: "ad47d700905bd5bc8dfb9196a9bccebed07c3ccf487e3579fe4ce49fd9085065",
    NORMALIZER_SOURCE: "cc06ea2e30605723ce4b4ff2acc09bafbb8293d964569fda529d134080426dcc",
    NATIVE_METRICS: "cc18a21a19d9fe817a8177a6da2eb277fea9863b1de1c3f11fdd08acfe859445",
    DEV_MANIFEST: "adb4e784b1704a668ab8c14eda17b35ae467f2fafcbce079aeacd29d3deaad78",
    CANONICAL_DEV: "917f05bc0f69eab0ea6607f0e6de3049557dbabc700fbb26957b46d3f481abad",
    QUERIES_DEV: "9ef7de5714ad8fb90ea960b7b95edc9210ed128669b50733b00a29ad42d5eeab",
    JOBS: "a1acb06002c16742545ac6d781e2fc4601043437eee2bf8d96ee30453b5b5115",
    CANDIDATE_REVIEW: "076734f49286d4c64cddf10aadaf5ee475aca533fb1dc2031cab2db8cc47c360",
    CANDIDATE_MERGED: "b79658ff27768b07f3bec562fb75a8445299681e23b5ecdc68d5f765f829b836",
    CANDIDATE_STAGE: "555d4ac05bf0dd61b713c09d9a118790dfca0251ad20edf1797bd985bc4647dc",
    CANDIDATE_SEAL: "7a0d70791b95fe72e394c8618e9eb7ea1e082bec793555d4d1bcb449c34fbbf2",
    CANDIDATE_CERTIFICATES: "32196349163d4d3d2bcbd4542c15d6dc5adf728edb2a7ba5cdfeff8c280f3ecd",
    CANDIDATE_CONSTRUCTION: "a3840da4580b09867ef7672ed372e923ad5ca8dc17a44193fa8066108b6d9545",
    CANDIDATE_ROUTES: "1d33b725029b41ed8a90940f24f4c4fb58d73ac282413a444ec2c38b2fea41ac",
    V9_WRAPPER: "4a7847d78f5fcc230e79c6395f0cbad5bca7917a7b01dc4d6ea9f2817de1821a",
    CANARY_REVIEW: "927e45ac2502dd510fa81cbadc31713142731ec5eb529f6479c9c29218e9031a",
}

HEADS = (
    "ensemble_last_hyp",
    "body_hyp",
    "fuse_hyp",
    "left_hyp",
    "right_hyp",
)
REQUIRED_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MALLOC_ARENA_MAX": "2",
    "CUDA_VISIBLE_DEVICES": "",
}
EXPECTED_NUMERICAL_RUNTIME = {
    "python": "3.12.3",
    "executable": "/home/kumwilai/research/coopns-slr/.venv/bin/python",
    "real_executable": "/usr/bin/python3.12",
    "numpy": "2.4.2",
}
PREANALYSIS_CHECKS = {
    "complete_local_merge_and_receipts",
    "candidate_independent_review_valid",
    "preregistered_method_source_recovered_exact",
    "paired_denominator_and_sign_convention",
    "caption_and_signer_bootstrap_exact",
    "analysis_targets_fresh",
    "development_only_claim_boundary",
}


class ProtocolError(RuntimeError):
    pass


def require(condition, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicate_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def checked_json(path: Path, expected_sha: str) -> dict:
    path = path.resolve()
    require(path.is_file() and sha256(path) == expected_sha,
            f"JSON hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=no_duplicate_object)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def write_new(path: Path, value: dict) -> None:
    path = path.resolve()
    require(path.parent == ANALYSIS_DIR.resolve() and path.suffix == ".json",
            "analysis output escapes assigned directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def available_memory_kib() -> int:
    values = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    require(values.get("MemAvailable", 0) > 0, "MemAvailable unavailable")
    return values["MemAvailable"]


def process_identity() -> dict:
    """Return a Linux process identity that survives PID-reuse checks."""
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    raw = Path("/proc/self/stat").read_text().strip()
    pid_prefix, separator, _ = raw.partition("(")
    _, tail_separator, fields_text = raw.rpartition(") ")
    fields = fields_text.split()
    require(separator == "(" and tail_separator == ") " and len(fields) > 19,
            "Linux process identity is unavailable")
    pid = os.getpid()
    require(pid_prefix.strip().isdigit() and int(pid_prefix) == pid
            and boot_id and fields[19].isdigit(),
            "Linux process identity is malformed")
    return {
        "boot_id": boot_id,
        "pid": pid,
        "start_time_clock_ticks_since_boot": int(fields[19]),
    }


def validate_distinct_process(first_identity: dict,
                              current_identity: dict) -> None:
    keys = {"boot_id", "pid", "start_time_clock_ticks_since_boot"}
    for label, identity in (("first", first_identity),
                            ("current", current_identity)):
        require(isinstance(identity, dict) and set(identity) == keys
                and isinstance(identity["boot_id"], str)
                and bool(identity["boot_id"])
                and type(identity["pid"]) is int and identity["pid"] > 0
                and type(identity["start_time_clock_ticks_since_boot"]) is int
                and identity["start_time_clock_ticks_since_boot"] > 0,
                f"{label} process identity differs")
    require(first_identity != current_identity,
            "repeat must execute in a distinct process")


def numerical_runtime(np) -> dict:
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "executable": str(Path(sys.executable).absolute()),
        "real_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
    }
    require(observed == EXPECTED_NUMERICAL_RUNTIME,
            "numerical runtime differs from preregistration")
    return observed


def validate_repeat_runtime(first: dict, observed: dict) -> None:
    require(first.get("runtime", {}).get("numerical") == observed,
            "separate-process numerical runtime differs")


def validate_environment() -> None:
    require(all(os.environ.get(key) == value for key, value in REQUIRED_ENV.items()),
            "analysis environment differs")
    require(sha256(Path(__file__).resolve()) == ARGS.source_sha256,
            "analysis source hash differs")


def validate_static_pins() -> dict:
    for path, expected in PINS.items():
        require(path.resolve().is_relative_to(ROOT.resolve()),
                f"static input escapes project root: {path}")
        require(path.is_file() and sha256(path) == expected,
                f"static input hash differs: {path}")
    prereg = checked_json(PREREG, PINS[PREREG])
    seal = checked_json(PREREG_SEAL, PINS[PREREG_SEAL])
    require(
        seal.get("schema") == "csl_full_structure_wer_adapted_v1_prereg_seal"
        and seal.get("preregistration_sha256") == PINS[PREREG]
        and prereg.get("counts", {}).get("dev") == 1077
        and prereg.get("analysis", {}).get("primary")
        == "fixed40 minus local ensemble_last corpus gloss-WER"
        and prereg.get("analysis", {}).get("bootstrap") == {
            "interval": "percentile2.5/97.5; paired cluster resampling; recompute reference and hypothesis denominators",
            "replicates": 10000,
            "scope": "conditional on frozen policies, observed caption/signer groups and available signer metadata; no refitting uncertainty, session inference, or population coverage claim",
            "seed": 30373,
            "units": ["normalized caption", "explicit signer"],
        }
        and prereg.get("inputs", {}).get("implementation", {}).get("sha256")
        == PINS[METHOD_SOURCE]
        and prereg.get("inputs", {}).get("metric_helper", {}).get("sha256")
        == PINS[METRIC_HELPER]
        and prereg.get("inputs", {}).get("normalizer", {}).get("sha256")
        == PINS[NORMALIZER_SOURCE]
        and prereg.get("inputs", {}).get("dev_manifest", {}).get("sha256")
        == PINS[DEV_MANIFEST]
        and {
            "python": prereg.get("runtime", {}).get("python"),
            "executable": prereg.get("runtime", {}).get("executable"),
            "real_executable": prereg.get("runtime", {}).get("real_executable"),
            "numpy": prereg.get("runtime", {}).get("numerical", {}).get("numpy"),
        } == EXPECTED_NUMERICAL_RUNTIME,
        "preregistered analysis contract differs",
    )
    candidate_review = checked_json(CANDIDATE_REVIEW, PINS[CANDIDATE_REVIEW])
    require(
        candidate_review.get("schema")
        == "csl_sharded_recovery_v8_independent_candidate_review"
        and candidate_review.get("verdict")
        == "PASS_VALID_FIXED20_DEVELOPMENT_MEASUREMENT"
        and candidate_review.get("bindings", {}).get("merged_result")
        == record(CANDIDATE_MERGED)
        and candidate_review.get("bindings", {}).get("candidate_stage")
        == record(CANDIDATE_STAGE)
        and candidate_review.get("bindings", {}).get("candidate_seal")
        == record(CANDIDATE_SEAL),
        "candidate independent review differs",
    )
    stage = checked_json(CANDIDATE_STAGE, PINS[CANDIDATE_STAGE])
    stage_seal = checked_json(CANDIDATE_SEAL, PINS[CANDIDATE_SEAL])
    require(
        stage_seal.get("stage_sha256") == PINS[CANDIDATE_STAGE]
        and stage.get("split") == "dev"
        and stage.get("n") == 1077
        and stage.get("outputs", {}).get("certificates.json")
        == PINS[CANDIDATE_CERTIFICATES]
        and stage.get("outputs", {}).get("construction.json")
        == PINS[CANDIDATE_CONSTRUCTION]
        and stage.get("outputs", {}).get("routes.json")
        == PINS[CANDIDATE_ROUTES],
        "candidate stage bindings differ",
    )
    canary = checked_json(CANARY_REVIEW, PINS[CANARY_REVIEW])
    require(canary.get("schema")
            == "csl_paired_local_recovery_v9_independent_canary_review"
            and canary.get("verdict") == "GO",
            "independent canary review differs")
    return {"preregistration": prereg, "candidate_review": candidate_review,
            "candidate_stage": stage}


def validate_merged(path: Path, expected_sha: str, inventory: Path,
                    inventory_sha: str) -> dict:
    require(path.resolve() == (LOCAL_DIR / "merged_result.json").resolve(),
            "local merged path differs")
    require(inventory.resolve() == (LOCAL_DIR / "receipt_inventory.json").resolve(),
            "local receipt inventory path differs")
    checked_json(inventory, inventory_sha)
    merged = checked_json(path, expected_sha)
    require(
        merged.get("schema") == "csl_paired_local_recovery_v9_merged"
        and merged.get("status") == "PROVISIONAL_AWAITING_INDEPENDENT_REVIEW"
        and merged.get("lane") == "local_comparator"
        and merged.get("n") == 20
        and merged.get("head_names") == sorted(HEADS)
        and merged.get("receipt_inventory")
        == {"path": str(inventory.resolve()), "sha256": inventory_sha}
        and merged.get("exact_cross_lane_replay", {}).get(
            "references_and_all_five_hypotheses_exact") is True,
        "local merged result differs",
    )
    return merged


def validate_preanalysis_review(path: Path, expected_sha: str,
                                local_merged: Path, local_sha: str,
                                inventory: Path, inventory_sha: str) -> dict:
    require(path.resolve() == (ANALYSIS_DIR / "preanalysis_review.json").resolve(),
            "pre-analysis review path differs")
    review = checked_json(path, expected_sha)
    expected_bindings = {
        "analysis_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": ARGS.source_sha256,
        },
        "local_merged_result": {
            "path": str(local_merged.resolve()), "sha256": local_sha,
        },
        "local_receipt_inventory": {
            "path": str(inventory.resolve()), "sha256": inventory_sha,
        },
        "candidate_merged_result": record(CANDIDATE_MERGED),
        "candidate_independent_review": record(CANDIDATE_REVIEW),
        "preregistration": record(PREREG),
        "preregistered_method_source": record(METHOD_SOURCE),
    }
    require(
        review.get("schema") == "csl_paired_fixed20_v10_preanalysis_review"
        and review.get("verdict") == "PASS_FROZEN_ANALYSIS_FOR_EXECUTION"
        and review.get("bindings") == expected_bindings
        and set(review.get("checks", {})) == PREANALYSIS_CHECKS
        and all(review["checks"].values()),
        "independent pre-analysis review differs",
    )
    return review


def extract_definitions(source: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    require(len(nodes) == len(names) and {node.name for node in nodes} == names,
            "method source definitions missing or duplicated")
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   str(source), "exec")
    exec(code, namespace)
    return namespace


def load_preregistered_methods():
    # Numerical imports happen only after every external review/hash gate.
    import numpy as np
    runtime = numerical_runtime(np)
    sys.path.insert(0, str(ROOT))
    try:
        from scripts import reduced_learned_csl_route as core
    finally:
        sys.path.pop(0)
    require(Path(core.__file__).resolve() == METRIC_HELPER.resolve()
            and tuple(core.HEADS) == HEADS,
            "metric helper import differs")
    namespace = {
        "Counter": Counter,
        "OrderedDict": OrderedDict,
        "Path": Path,
        "core": core,
        "digest": sha256,
        "io": io,
        "np": np,
        "pickle": pickle,
        "require": require,
        "sys": sys,
        "HEADS": HEADS,
    }
    extract_definitions(METHOD_SOURCE, {
        "FloatStorageHeader", "numpy_float_storage", "numpy_tensor",
        "NativeNumpyUnpickler", "native_archive", "native_pose",
        "clustered_uncertainty", "pose_geometry",
    }, namespace)
    normalizer = extract_definitions(
        NORMALIZER_SOURCE, {"normalize_caption"},
        {"unicodedata": unicodedata},
    )["normalize_caption"]
    return np, core, namespace, normalizer, runtime


def recomputed_summary(core, rows: dict) -> dict:
    require(list(rows) and all(set(row) == {"ref", *HEADS}
                               for row in rows.values()),
            "result row schema differs")
    return core.summarize(rows)


def load_pose(np, native_pose, path: Path, expected_file_sha: str,
              expected_pose: dict):
    require(path.is_file() and sha256(path) == expected_file_sha,
            f"candidate pose file differs: {path}")
    with path.open("rb") as handle:
        value = np.load(handle, allow_pickle=False)
        require(not handle.read(1), "trailing pose artifact bytes")
    value = native_pose(value)
    commitment = {
        "dtype": str(value.dtype),
        "shape": list(map(int, value.shape)),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(value).tobytes(order="C")
        ).hexdigest(),
    }
    require(commitment == expected_pose, "pose semantic commitment differs")
    return value


def source_and_geometry_summary(np, methods: dict, ids: list[str],
                                jobs: dict, stage: dict, certificates: dict,
                                construction: dict, canonical: dict) -> dict:
    native_pose = methods["native_pose"]
    pose_geometry = methods["pose_geometry"]
    scale = construction.get("admission", {}).get(
        "boundary_shoulder_scale_native_pixels")
    require(type(scale) in (int, float) and scale > 0,
            "train-only shoulder scale differs")
    gt = {}
    for sample_id in ids:
        row = canonical.get(sample_id)
        require(isinstance(row, dict) and row.get("name") == sample_id,
                "canonical development row differs")
        gt[sample_id] = pose_geometry(row, scale)
    summaries = {}
    pose_hashes = {}
    geometry_per_id = {"ground_truth": gt}
    for label, cert_route, job_name, folder in (
        ("local", "local", "local", "local"),
        ("fixed40", "fixed40", "fixed40_first20", "fixed40"),
    ):
        per_id = {}
        corpus_mass = Counter()
        hashes = []
        for sample_id in ids:
            rel = f"{folder}/{hashlib.sha256(sample_id.encode()).hexdigest()}.npy"
            expected_pose = jobs[job_name]["pose_commitments"][sample_id]
            pose = load_pose(np, native_pose, CANDIDATE_DIR / rel,
                             stage["outputs"][rel], expected_pose)
            cert = certificates[sample_id][cert_route]["payload"]
            require(
                cert.get("id") == sample_id
                and cert.get("route") == cert_route
                and cert.get("pose") == expected_pose
                and cert.get("context", {}).get("implementation_sha256")
                == PINS[METHOD_SOURCE]
                and cert.get("context", {}).get("prereg_sha256")
                == PINS[PREREG],
                "candidate certificate identity differs",
            )
            masses = cert.get("source_quarter_frames", {})
            total = cert.get("quarter_frames_total")
            require(masses and all(type(v) is int and v > 0 for v in masses.values())
                    and sum(masses.values()) == total == 4 * len(pose),
                    "certificate source mass differs")
            concentration = sum((value / total) ** 2
                                for value in masses.values())
            row = pose_geometry(pose, scale)
            row.update({
                "train_source_count": len(masses),
                "max_source_fraction": max(masses.values()) / total,
                "source_concentration": concentration,
                "effective_source_count": 1 / concentration,
                "output_gt_length_ratio": len(pose) / gt[sample_id]["frames"],
            })
            per_id[sample_id] = row
            corpus_mass.update(masses)
            hashes.append(expected_pose["sha256"])
        keys = tuple(next(iter(per_id.values())))
        corpus_total = sum(corpus_mass.values())
        mean = {key: float(np.mean([per_id[sid][key] for sid in ids]))
                for key in keys}
        gt_mean = {
            key: float(np.mean([gt[sid][key] for sid in ids]))
            for key in next(iter(gt.values()))
        }
        summaries[label] = {
            "n": len(ids),
            "unique_pose_hashes": len(set(hashes)),
            "unique_train_sources": len(corpus_mass),
            "corpus_max_source_frame_fraction": max(corpus_mass.values()) / corpus_total,
            "corpus_effective_source_count": 1 / sum(
                (value / corpus_total) ** 2 for value in corpus_mass.values()
            ),
            "mean_per_clip": mean,
            "total_emitted_frames": sum(per_id[sid]["frames"] for sid in ids),
            "total_mska_retained_frames": sum(
                per_id[sid]["mska_retained_frames"] for sid in ids
            ),
            "total_gt_frames": sum(gt[sid]["frames"] for sid in ids),
            "corpus_output_gt_length_ratio": (
                sum(per_id[sid]["frames"] for sid in ids)
                / sum(gt[sid]["frames"] for sid in ids)
            ),
            "motion_ratios_to_gt_mean": {
                "hand_displacement": (
                    mean["hand_displacement_px_per_frame"]
                    / gt_mean["hand_displacement_px_per_frame"]
                ),
                "hand_temporal_variance": (
                    mean["hand_temporal_variance_px2"]
                    / gt_mean["hand_temporal_variance_px2"]
                ),
            },
            "train_source_quarter_frames": dict(sorted(corpus_mass.items())),
        }
        groups = defaultdict(list)
        for sample_id, pose_hash in zip(ids, hashes):
            groups[pose_hash].append(sample_id)
        summaries[label]["duplicate_pose_groups"] = sorted(
            sorted(group) for group in groups.values() if len(group) > 1
        )
        pose_hashes[label] = dict(zip(ids, hashes))
        geometry_per_id[label] = per_id
    gt_mean = {
        key: float(np.mean([gt[sid][key] for sid in ids]))
        for key in next(iter(gt.values()))
    }
    return {
        "units": "native512px; train-only shoulder normalization",
        "not_joint_aligned_fidelity": True,
        "ground_truth_mean_per_clip": gt_mean,
        "route_summaries": summaries,
        "pose_hashes": pose_hashes,
        "geometry_per_id": geometry_per_id,
    }


def hypothesis_coverage(ids: list[str], routes: dict[str, dict]) -> dict:
    result = {}
    for label, rows in routes.items():
        require(list(rows) == ids, f"{label} result order differs")
        require(all(set(rows[sid]) == {"ref", *HEADS} for sid in ids),
                f"{label} result schema differs")
        result[label] = {
            "requested_n": len(ids),
            "scored_n": len(rows),
            "missing_ids": [],
            "missing_head_predictions": sum(
                head not in rows[sid] for sid in ids for head in HEADS
            ),
            "empty_reference_count": sum(not rows[sid]["ref"].split()
                                         for sid in ids),
            "empty_hypothesis_counts": {
                head: sum(not rows[sid][head].split() for sid in ids)
                for head in HEADS
            },
            "unique_reference_count": len({rows[sid]["ref"] for sid in ids}),
            "unique_hypothesis_counts": {
                head: len({rows[sid][head] for sid in ids}) for head in HEADS
            },
        }
    require(all(row["missing_head_predictions"] == 0
                and row["empty_reference_count"] == 0
                and not any(row["empty_hypothesis_counts"].values())
                for row in result.values()),
            "missing or empty paired prediction")
    return result


def build_scientific(static: dict, local: dict, local_sha: str,
                     inventory: Path, inventory_sha: str,
                     review: Path, review_sha: str) -> tuple[dict, dict]:
    np, core, methods, normalize_caption, runtime = load_preregistered_methods()
    candidate = checked_json(CANDIDATE_MERGED, PINS[CANDIDATE_MERGED])
    ids = candidate.get("ordered_ids")
    require(isinstance(ids, list) and len(ids) == 20
            and ids == local.get("ordered_ids")
            and list(candidate.get("per_id", {})) == ids
            and list(local.get("per_id", {})) == ids,
            "paired fixed20 ID denominator/order differs")
    fixed_rows, local_rows = candidate["per_id"], local["per_id"]
    require(all(fixed_rows[sid]["ref"] == local_rows[sid]["ref"]
                for sid in ids), "paired references differ")
    fixed_summary = recomputed_summary(core, fixed_rows)
    local_summary = recomputed_summary(core, local_rows)
    require(fixed_summary == candidate["summary"]
            and local_summary == local["summary"],
            "stored corpus summaries differ from native recomputation")

    manifest_rows = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"),
                               object_pairs_hook=no_duplicate_object)
    query_rows = json.loads(QUERIES_DEV.read_text(encoding="utf-8"),
                            object_pairs_hook=no_duplicate_object)
    manifest = {row["id"]: row for row in manifest_rows}
    queries = {row["id"]: row for row in query_rows}
    require(len(manifest) == len(manifest_rows) == 1077
            and len(queries) == len(query_rows) == 1077,
            "development annotation/query denominator differs")
    require(all(manifest[sid]["gloss"] == fixed_rows[sid]["ref"]
                and type(manifest[sid].get("signer")) in (str, int)
                and str(manifest[sid]["signer"]).strip()
                and queries[sid]["text"] == manifest[sid]["text"]
                for sid in ids),
            "fixed20 reference/caption/signer lineage differs")
    captions = {sid: normalize_caption(queries[sid]["text"]) for sid in ids}
    signers = {sid: str(manifest[sid]["signer"]) for sid in ids}
    require(all(captions.values()) and len(set(captions.values())) >= 2
            and len(set(signers.values())) >= 2,
            "insufficient explicit bootstrap clusters")
    uncertainty = {
        "normalized_caption": methods["clustered_uncertainty"](
            local_rows, fixed_rows, captions, replicates=10000, seed=30373
        ),
        "explicit_signer": methods["clustered_uncertainty"](
            local_rows, fixed_rows, signers, replicates=10000, seed=30373
        ),
    }
    deltas = {}
    for head in HEADS:
        point = fixed_summary[head]["wer"] - local_summary[head]["wer"]
        require(all(abs(block["heads"][head]["delta_corpus_wer"] - point)
                    < 1e-12 for block in uncertainty.values()),
                "bootstrap point delta differs from corpus summaries")
        deltas[head] = {
            "local": local_summary[head],
            "fixed40": fixed_summary[head],
            "fixed40_minus_local_corpus_wer_points": point,
            "negative_delta_means_fixed40_lower_wer": True,
        }

    jobs = checked_json(JOBS, PINS[JOBS])
    certificates = checked_json(CANDIDATE_CERTIFICATES,
                                PINS[CANDIDATE_CERTIFICATES])
    construction = checked_json(CANDIDATE_CONSTRUCTION,
                                PINS[CANDIDATE_CONSTRUCTION])
    route_plan = checked_json(CANDIDATE_ROUTES, PINS[CANDIDATE_ROUTES])
    require(jobs["fixed40_first20"]["ids"] == ids
            and jobs["fixed40_first20"]["n"] == 20,
            "frozen job membership differs")
    canonical = methods["native_archive"](CANONICAL_DEV, PINS[CANONICAL_DEV])
    geometry = source_and_geometry_summary(
        np, methods, ids, jobs, static["candidate_stage"], certificates,
        construction, canonical,
    )
    del canonical
    selected_whole = [sid for sid in ids if route_plan["fixed40"]["mask"][sid]]
    selected_local = [sid for sid in ids if not route_plan["fixed40"]["mask"][sid]]
    require(len(selected_whole) == 13 and len(selected_local) == 7,
            "fixed20 frozen route mixture differs")
    same_ids = [sid for sid in ids
                if geometry["pose_hashes"]["local"][sid]
                == geometry["pose_hashes"]["fixed40"][sid]]
    replay = local["exact_cross_lane_replay"]
    require(same_ids == replay["ids"]
            and replay["indices"] == [ids.index(sid) for sid in same_ids]
            and all(local_rows[sid] == fixed_rows[sid] for sid in same_ids),
            "independent exact cross-lane replay differs")
    coverage = hypothesis_coverage(
        ids, {"local": local_rows, "fixed40": fixed_rows}
    )

    bindings = {
        "analysis_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": ARGS.source_sha256,
        },
        "preregistration": record(PREREG),
        "preregistration_seal": record(PREREG_SEAL),
        "preregistered_method_source": record(METHOD_SOURCE),
        "metric_helper": record(METRIC_HELPER),
        "native_metrics": record(NATIVE_METRICS),
        "candidate_merged_result": record(CANDIDATE_MERGED),
        "candidate_independent_review": record(CANDIDATE_REVIEW),
        "local_merged_result": {
            "path": str(ARGS.local_merged.resolve()), "sha256": local_sha,
        },
        "local_receipt_inventory": {
            "path": str(inventory.resolve()), "sha256": inventory_sha,
        },
        "independent_preanalysis_review": {
            "path": str(review.resolve()), "sha256": review_sha,
        },
    }
    scientific = {
        "schema": "csl_paired_fixed20_v10_scientific",
        "scope": {
            "split": "dev",
            "subset": "preserved lexical-first20 IDs of frozen 1077-item development job",
            "n": 20,
            "primary_head": "ensemble_last_hyp",
            "primary_comparison": "fixed40 minus local corpus gloss-WER",
            "route_counts": {"local": len(selected_local),
                             "whole": len(selected_whole)},
            "selection_or_tuning_after_outcomes": False,
        },
        "protocol_adaptation": (
            "Separately versioned fixed20-development subset adapter. The exact "
            "preregistered statistic is executed from the preserved source snapshot; "
            "the original full-test analyze entry point is not invoked."
        ),
        "bindings": bindings,
        "ordered_ids": ids,
        "all_head_metrics_and_deltas": deltas,
        "conditional_paired_cluster_bootstrap": uncertainty,
        "grouping": {
            "normalized_caption": {
                "clusters": len(set(captions.values())),
                "cluster_sizes": dict(sorted(Counter(captions.values()).items())),
            },
            "explicit_signer": {
                "clusters": len(set(signers.values())),
                "cluster_sizes": dict(sorted(Counter(signers.values()).items())),
                "authority": "explicit CSL JSON manifest signer field; never parsed from ID",
            },
        },
        "coverage_and_hypothesis_diversity": coverage,
        "pose_geometry_and_source_diversity": geometry,
        "exact_cross_lane_replay": {
            "ids": same_ids,
            "indices": [ids.index(sid) for sid in same_ids],
            "count": len(same_ids),
            "references_and_all_five_hypotheses_exact": True,
        },
        "primary_gate_by_observed_grouping": {
            name: block["heads"]["ensemble_last_hyp"]["improvement_gate"]
            for name, block in uncertainty.items()
        },
        "negative_results_retained": True,
        "statistical_scope": (
            "Conditional on frozen policies and the observed caption/signer groups; "
            "no refitting, independent-sample, session, or population uncertainty claim."
        ),
        "claim_boundary": (
            "Paired descriptive native-MSKA contrast for exactly the lexical-first20 "
            "CSL-Daily development IDs. It is not full-development/test transfer, "
            "generalization, superiority, release, visual-quality, human-intelligibility, "
            "or human-evaluation evidence."
        ),
    }
    return scientific, runtime


def main(args) -> None:
    global ARGS
    ARGS = args
    validate_environment()
    require(args.out.resolve().parent == ANALYSIS_DIR.resolve(),
            "analysis output directory differs")
    is_repeat = args.repeat_of is not None
    require(args.out.name == ("analysis_repeat.json" if is_repeat
                              else "analysis_first.json"),
            "analysis output name differs")
    require(not args.out.exists(), "analysis output already exists")
    if is_repeat:
        require(args.repeat_of.resolve()
                == (ANALYSIS_DIR / "analysis_first.json").resolve()
                and args.repeat_of_sha256,
                "repeat authority differs")
    else:
        require(args.repeat_of_sha256 is None,
                "repeat hash supplied without repeat artifact")
    current_process = process_identity()
    with SHARED_LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProtocolError("shared heavy-process lock is held") from error
        before = available_memory_kib()
        require(before >= 16 * 1024 * 1024,
                "analysis requires at least 16 GiB available")
        limit = 4 * 1024**3
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        require(all(value == resource.RLIM_INFINITY or value >= limit
                    for value in (soft, hard)),
                "inherited address-space limit is too small")
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        static = validate_static_pins()
        local = validate_merged(
            args.local_merged, args.local_merged_sha256,
            args.receipt_inventory, args.receipt_inventory_sha256,
        )
        validate_preanalysis_review(
            args.preanalysis_review, args.preanalysis_review_sha256,
            args.local_merged, args.local_merged_sha256,
            args.receipt_inventory, args.receipt_inventory_sha256,
        )
        first = None
        if is_repeat:
            first = checked_json(args.repeat_of, args.repeat_of_sha256)
            require(
                first.get("schema") == "csl_paired_fixed20_v10_analysis"
                and first.get("status")
                == "PROVISIONAL_AWAITING_REPEAT_AND_INDEPENDENT_REVIEW"
                and first.get("repeat") == {
                    "required": True,
                    "status": "PENDING_SEPARATE_PROCESS_REPEAT",
                },
                "first analysis repeat authority differs",
            )
            validate_distinct_process(
                first.get("runtime", {}).get("process_identity"),
                current_process,
            )
        scientific, runtime = build_scientific(
            static, local, args.local_merged_sha256,
            args.receipt_inventory, args.receipt_inventory_sha256,
            args.preanalysis_review, args.preanalysis_review_sha256,
        )
        repeat = {"required": True, "status": "PENDING_SEPARATE_PROCESS_REPEAT"}
        if is_repeat:
            validate_repeat_runtime(first, runtime)
            require(first.get("scientific") == scientific,
                    "separate-process scientific repeat differs")
            repeat = {
                "required": True,
                "status": "PASS_EXACT_SCIENTIFIC_FIELDS",
                "first_analysis": {
                    "path": str(args.repeat_of.resolve()),
                    "sha256": args.repeat_of_sha256,
                },
            }
        after = available_memory_kib()
        value = {
            "schema": "csl_paired_fixed20_v10_analysis",
            "status": (
                "REPEAT_EXACT_AWAITING_INDEPENDENT_REVIEW" if is_repeat
                else "PROVISIONAL_AWAITING_REPEAT_AND_INDEPENDENT_REVIEW"
            ),
            "scientific": scientific,
            "repeat": repeat,
            "runtime": {
                "created_unix_ns": time.time_ns(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "mem_available_before_kib": before,
                "mem_available_after_kib": after,
                "address_space_limit_bytes": limit,
                "numerical": runtime,
                "process_identity": current_process,
                "model_or_recognizer_launched": False,
                "shared_heavy_lock": str(SHARED_LOCK.resolve()),
            },
        }
        write_new(args.out, value)
        print(json.dumps({
            "artifact": record(args.out),
            "status": value["status"],
            "primary": scientific["all_head_metrics_and_deltas"][
                "ensemble_last_hyp"
            ],
            "gates": scientific["primary_gate_by_observed_grouping"],
            "peak_rss_kib": value["runtime"]["peak_rss_kib"],
        }, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-merged", type=Path, required=True)
    ap.add_argument("--local-merged-sha256", required=True)
    ap.add_argument("--receipt-inventory", type=Path, required=True)
    ap.add_argument("--receipt-inventory-sha256", required=True)
    ap.add_argument("--preanalysis-review", type=Path, required=True)
    ap.add_argument("--preanalysis-review-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--repeat-of", type=Path)
    ap.add_argument("--repeat-of-sha256")
    return ap


ARGS = None
if __name__ == "__main__":
    main(parser().parse_args())
