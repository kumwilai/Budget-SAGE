"""Confined RGB -> current CorrNet decision-equivalence audit, not posterior reproduction.

Stages: synthetic-gate, preregister, execute (pilot/full/repeat), compare.
Inference requires an explicit --authorize-inference flag. The external cache
CLI is never invoked. Reviewed definitions and the exact posterior-producing
statements of process_split are compiled from the bound source. Its optional
MSKA pose/NMM feature branch is outside this audit's scope.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import gc
import glob
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import pickle
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import sysconfig
import time
import types
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

# These are part of the sealed execution policy, including CPU-only tests.
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/revision/nonhuman_closure_20260907"
COOP = Path("/home/kumwilai/research/coopns-slr")
CORR = COOP / "external/CorrNet"
DATASET = COOP / "data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
SCHEMA = "raw_rgb_corrnet_decision_equivalence_v7"
SELECTION = "v5_input_only_pilot_minus_permanently_disclosed_calibration_v6"
CALIBRATION_ID = "26August_2009_Wednesday_heute-6551"
FROZEN_PILOT_IDS = (
    CALIBRATION_ID, "09December_2009_Wednesday_heute-4741", "17April_2010_Saturday_tagesschau-405",
    "03July_2009_Friday_tagesschau-2012", "01May_2010_Saturday_tagesschau-7186",
    "28September_2012_Friday_tagesschau-926", "21June_2011_Tuesday_tagesschau-1262",
    "07May_2011_Saturday_tagesschau-114", "19February_2010_Friday_tagesschau-4096",
    "02October_2012_Tuesday_tagesschau-2846",
)
SEED = 20260907
POLICY = {
    "split": "train", "original_pilot_size": 10, "pilot_size": 9, "blank_id": 0,
    "calibration": "first v5 clip permanently CALIBRATION; never validation; prior continuous failure retained",
    "validation": "exact remaining nine v5 IDs; no replacements, exclusions or threshold revision after any violation",
    "resize": "OpenCV INTER_LANCZOS4 210x260 -> 256x256; BGR decoded pixels",
    "preprocess": "bound CorrNet CenterCrop(224), RGB, /127.5-1, K5/P2/K5/P2 padding",
    "posterior_fields": ["log_probs", "gloss_indices", "T_out"],
    "model_return_contract": {
        "feat_len": "torch.float32; shape (1,); finite, exactly integral, exactly ceil(input_frames/4)",
        "sequence_logits_and_conv_logits": "torch.float32; shape (ceil(input_frames/4), 1, vocab_size); every value finite",
        "length_conversion": "validation only; preserve production tensor and unchanged inference-core int(item()) conversion",
    },
    "historical_comparison": "strict_margin_certified_decision_equivalence_only",
    "margin_gate": "for every frame t: 2*max_j(abs(current_log_probs[t,j]-historical_log_probs[t,j])) < historical_top1_log_prob[t]-historical_top2_log_prob[t]",
    "continuous_errors": "complete arrays retained; signed/absolute distributions and per-frame errors reported; no fitted numeric acceptance tolerance; not historical posterior reproduction",
    "deterministic_repeat": "two nonoverlapping CPU runs; disjoint artifacts; tensor/NPZ-byte-exact outputs and byte-exact reconstructed scientific reports",
    "full_scope": "separate preregistration after independently approved pilot+repeat; census every 7096 item with calibration separated and the same all-item semantic gate",
    "pixels_exact": True, "tokens_exact": True, "T_out_exact": True,
    "ctc_spans_exact": True, "framewise_argmax_exact": True, "skips_allowed": 0, "fallbacks_allowed": 0,
    "torch_deterministic": True, "cudnn_benchmark": False,
    "tf32": False, "seed": SEED, "threads": 1,
    "peak_rss_limit_bytes": 7 * 1024**3, "minimum_available_ram_bytes": 11 * 1024**3,
    "maximum_frames_per_clip": 512, "minimum_free_gpu_bytes": 11 * 1024**3,
    "one_memory_heavy_process": True,
}
REVISION = {
    "version": 7,
    "v2_review_status": "BLOCKED; superseded for execution, retained as historical evidence",
    "v2_block_reasons": ["external source closure was not required to match the tested gate",
                         "repeat allowed self-comparison and tolerance-level drift with declared denominators",
                         "same-version numerical binary swaps and actual backend flags were not bound"],
    "v3_review_status": "BLOCKED; superseded for execution, retained as historical evidence",
    "v3_block_reasons": ["runtime closure omitted loaded non-numerical modules and executable mappings",
                         "Python/stdlib extension dependencies and late executable imports were not closed"],
    "v4_review_status": "BLOCKED after first real pilot; all evidence retained; no completed clip or posterior artifact",
    "v4_block_reasons": ["guard incorrectly required integer feat_len; pinned TemporalConv.update_lgt produces integral torch.float32 via torch.div without rounding_mode"],
    "v5_semantic_change": "admit only production float32 integral lengths, reject integer/other dtypes; additionally require exact float32 shape/finiteness for both logit tensors",
    "v5_review_status": "BLOCKED: first real clip failed the frozen continuous posterior tolerance; permanently disclosed calibration, not validation",
    "v6_semantic_change": "new decision-equivalence estimand with a strict mathematical per-frame margin certificate; nine untouched frozen validation IDs and two exact CPU runs required; no CPU-versus-GPU causal assertion",
    "v6_review_status": "BLOCKED before validation inference; a coherent nine-clip pilot report could falsely declare full_training_census_complete=True",
    "v7_contract_change": "require a typed census claim derived from sealed scope, complete item/role coverage and reconstructed denominators; forbid comparison-only claims in individual reports; no estimand, partition or acceptance change",
    "v7_calibration_reuse": "exact pinned v6 calibration and reference commitments; rederive retained small-array metrics without another production-cache deserialization",
    "superseded_evidence": {
        "v4_pilot_failure": {"path": str(OUT / "raw_repro_pilot_v4/failure.json"), "bytes": 293,
                             "sha256": "38dd00bccdc7c89cc126a419aa79e5b4b40e2e60bb023b0db6cfb7e2051f07cf"},
        "v4_pilot_timing": {"path": str(OUT / "raw_repro_pilot_v4/time_stderr.log"), "bytes": 2746,
                            "sha256": "6789f6e21e5a0681b1403d0b292987a94ca89cd31a3ea90edace6f9e2b3ab897"},
        "v4_prereg_seal": {"path": str(OUT / "raw_repro_prereg_v4/seal.json"), "bytes": 290,
                           "sha256": "f011e35e25c9fd02adecdebc5d5bb7940c69ad362686b4d813faccee6863fcb6"},
        "v5_pilot_failure": {"path": str(OUT / "raw_repro_pilot_v5/failure.json"), "bytes": 567,
                             "sha256": "163b8e5e394c5ce72463dc2777315272899233911f076104b1ed1fe49ae5c17c"},
        "v5_pilot_output": {"path": str(OUT / "raw_repro_pilot_v5/posteriors/26August_2009_Wednesday_heute-6551.npz"), "bytes": 113082,
                            "sha256": "88cd91ec71a8d4521d9ee3581c7ae7059706bbca4d902174ccd7689e622dc333"},
        "v5_execution_receipt": {"path": str(OUT / "raw_repro_pilot_v5/execution_record.json"), "bytes": 2398,
                                 "sha256": "8b83c5a59b991aef9d3dea9e4949f22636b6dc909e97bcea2699c7d39766594d"},
        "v5_prereg_seal": {"path": str(OUT / "raw_repro_prereg_v5/seal.json"), "bytes": 290,
                           "sha256": "1045fe1fa38bd6d53d0794506a2d6383ddebdb7b032a169914db261b6103c604"},
        "v6_calibration": {"path": str(OUT / "raw_repro_synthetic_gate_v6/calibration.json"), "bytes": 11501,
                           "sha256": "1797d06c9d34a0d20ceec93b5af94737ec1191e03cc910a550e8f42d1e5a4304"},
        "v6_gate": {"path": str(OUT / "raw_repro_synthetic_gate_v6/gate.json"), "bytes": 4568235,
                    "sha256": "dec24665e3159c6e2f82f49a52aa567e74638c5032e02b891299a51985c22fff"},
        "v6_prereg_seal": {"path": str(OUT / "raw_repro_prereg_v6/seal.json"), "bytes": 294,
                           "sha256": "f8f4c67a4d45086bd5201c45db4a4ce7ade1618a9563b4ae927225c997c65286"},
    },
}
GPU_CONTEXT_OBSERVATION = {
    "observed_date": "2026-09-07",
    "direct_exec_command_query": "nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader,nounits -i 0",
    "direct_exec_command_result": "0, GPU-5b3b2b89-1c37-60a2-3a1d-01f55f1425f7, NVIDIA GeForce RTX 5070 Ti, 595.79, 16303",
    "python_before_mask": {"torch": "2.10.0+cu128", "cuda_build": "12.8", "cuda_available": False, "device_count": 0},
    "python_subprocess_result": "Failed to initialize NVML: GPU access blocked by the operating system",
    "interpretation": "Direct tool GPU visibility and Python execution-context access differ. CPU is an explicitly authorized protocol; GPU diagnostics do not gate CPU execution.",
}
CLAIM_LIMIT = {
    "scope": "Original PHOENIX training RGB -> decoded cached256 pixel identity -> current frozen CorrNet DECISION-equivalent outputs and strict CTC spans only.",
    "historical_continuous_posteriors_reproduced": False,
    "historical_backend_reproduced": False,
    "calibration_is_validation": False,
    "custodian_supplied": ["historical holistic .pose files", "organizer train.pt poses_3d", "native-178 pose archive"],
    "not_reproduced": ["historical continuous posteriors", "historical backend", "raw-video holistic pose producer", "organizer poses_3d producer", "recognizer training", "human intelligibility", "entire raw-to-release chain"],
    "pose_boundary": "build_slrtp178_exemplars.py replaces exemplar pose values with organizer poses_3d; raw RGB replay does not reproduce those values.",
}
DEFAULTS = {
    "manifest": ROOT / "data/phoenix/phoenix_train.json",
    "info": CORR / "preprocess/phoenix2014-T/train_info.npy",
    "vocab": CORR / "preprocess/phoenix2014-T/gloss_dict.npy",
    "checkpoint": COOP / "models/corrnet/corrnet_phoenix2014t.pt",
    "resnet_init": Path("/home/kumwilai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
    "posterior_cache": COOP / "outputs/cached_posteriors_corrnet/train.pkl",
    "raw_root": DATASET / "features/fullFrame-210x260px/train",
    "cache_root": DATASET / "features/fullFrame-256x256px/train",
}
SOURCE_PATHS = {
    "wrapper": Path(__file__).resolve(),
    "tests": ROOT / "tests/test_audit_raw_reproduction_closure.py",
    "corrnet_cache": COOP / "scripts/cache_corrnet_posteriors.py",
    "resnet": CORR / "modules/resnet.py", "tconv": CORR / "modules/tconv.py",
    "bilstm": CORR / "modules/BiLSTM.py",
    "resize": CORR / "preprocess/dataset_preprocess-T.py",
    "ctc": ROOT / "src/data/build_dict_aligned.py",
    "exemplar_builder": ROOT / "scripts/build_pergloss_exemplars.py",
    "native178_builder": ROOT / "scripts/build_slrtp178_exemplars.py",
}
DEFINITIONS = ("Identity", "NormLinear", "corrnet_resnet18_csl_daily", "CorrNetModel",
               "build_corrnet_model", "load_and_preprocess_video", "pad_video_for_model")
MODEL_ASSET_KEYS = ("checkpoint", "resnet_init", "vocab")
RUNTIME_DISTRIBUTIONS = ("numpy", "torch", "opencv-python", "opencv-python-headless", "opencv-contrib-python", "threadpoolctl")
ENVIRONMENT_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                    "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH",
                    "MKL_CBWR", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_MAX_ACTIVE_LEVELS")
_THREADPOOL_LIMITER = None
RUNTIME_BOUNDARY = {
    "phase": "after explicit inference/preprocess/CTC, hash, resource, serialization imports and runtime configuration",
    "inventory": "union of sys.modules code files, file-backed executable maps, Python executable, enumerated stdlib/numerical distribution code, audit sources, and recursively resolved ELF dependencies",
    "lazy_import_rule": "observations may vary only within the exact content-hashed frozen file/alias inventory; no directory wildcard admission at validation",
    "hashing": "fresh SHA256 of every regular file per capture, 1 MiB buffers; no cross-capture/stat cache",
    "mapping_normalization": "canonical paths and disk/mapping identity; omit ASLR addresses, offsets and duplicate segments",
    "mapping_device_exception": {"path": "/usr/lib/wsl/lib/libcuda.so.1", "mapped_device": [0, 36],
                                 "filesystem_device": [0, 58], "same_inode_required": True,
                                 "reason": "Observed WSL map/filesystem device-number disagreement; map_files target agrees but map_files bytes are permission-denied. Both identities and on-disk bytes are bound; no claim of hashing in-memory relocated code."},
    "exclusions": ["anonymous mappings, including anonymous JIT code/data", "kernel-provided bracket mappings such as [vdso], [vsyscall], [vvar]",
                   "kernel, firmware, device state, container image, environment outside explicit pins, non-code data outside separately pinned assets"],
    "claim": "finite on-disk Python/native execution closure, not fully hermetic OS/container closure",
}


class AuditError(ValueError):
    """The frozen acceptance contract was not satisfied."""


def require(ok, message):
    if not ok:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def owned(path):
    path = Path(os.path.abspath(path))
    base = Path(os.path.abspath(OUT))
    require(path.is_relative_to(base), "output outside assigned revision root")
    parts = path.relative_to(base).parts
    require(parts and re.fullmatch(r"raw_repro_(synthetic_gate|prereg|pilot|full|repeat|compare)_v[1-9][0-9]*", parts[0]),
            "output outside raw_repro ownership")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(base)),
            "symlink in output path")
    require(path.resolve().is_relative_to(base.resolve()), "resolved output escapes root")
    return path


def stage_dir(path, stage, create=True):
    path = owned(path)
    require(path.parent == OUT.absolute() and re.fullmatch(r"raw_repro_" + stage + r"_v[1-9][0-9]*", path.name),
            "wrong output path for stage")
    if create:
        require(not path.exists(), "output exists; use a fresh version")
        path.mkdir(parents=True, exist_ok=False)
    return path


def write_bytes_new(path, data):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_new(path, value):
    write_bytes_new(path, canonical(value) + b"\n")


def file_record(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"missing/non-file input: {path}")
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f"input mutation while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def verify_record(record):
    require(file_record(record["path"]) == record, f"input/hash drift: {record['path']}")


def validate_revision_lineage():
    for record in REVISION["superseded_evidence"].values():
        verify_record(record)


def read_bound(record):
    verify_record(record)
    value = json.loads(Path(record["path"]).read_text())
    verify_record(record)
    return value


def checked_id(sid):
    require(isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_+.-]+", sid)
            and sid not in (".", ".."), "invalid/path-traversing clip ID")
    return sid


def validate_metadata(manifest, info, vocab):
    require(isinstance(manifest, list) and manifest, "empty/malformed training manifest")
    require(isinstance(info, dict), "malformed CorrNet info")
    integer_keys = [k for k in info if type(k) is int]
    require(sorted(integer_keys) == list(range(len(integer_keys))), "missing/duplicate CorrNet metadata index")
    require(len(vocab) > 0, "empty vocabulary")
    vocab_ids = [int(v[0]) for v in vocab.values()]
    require(sorted(vocab_ids) == list(range(1, len(vocab) + 1)), "duplicate/invalid vocabulary IDs or blank ID")
    corr_by_id = {}
    for key in integer_keys:
        item = info[key]
        sid = checked_id(item["fileid"])
        require(sid not in corr_by_id, "duplicate CorrNet metadata ID")
        corr_by_id[sid] = item
    rows, seen = [], set()
    for item in manifest:
        sid = checked_id(item["id"])
        require(sid not in seen, "duplicate training ID")
        seen.add(sid)
        require(sid in corr_by_id, f"missing CorrNet metadata ID: {sid}")
        corr = corr_by_id[sid]
        require(item.get("signer") == corr.get("signer") and item.get("gloss") == corr.get("label"),
                f"signer/gloss metadata mismatch: {sid}")
        require(isinstance(item["signer"], str) and item["signer"], "missing signer")
        require(type(item.get("length")) is int and item["length"] > 0, "invalid metadata length")
        tokens = item["gloss"].split()
        require(tokens and all(t in vocab for t in tokens), f"token drop/OOV/empty gloss: {sid}")
        rows.append({"id": sid, "signer": item["signer"], "length": item["length"],
                     "gloss": item["gloss"], "gloss_indices": [int(vocab[t][0]) for t in tokens]})
    require(seen == set(corr_by_id), "training/CorrNet ID universe mismatch")
    return sorted(rows, key=lambda row: row["id"])


def select_pilot(rows, count=10):
    require(count == 10 and len(rows) >= count, "pilot must contain exactly ten training clips")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate selection input ID")
    ordered = sorted(rows, key=lambda r: (r["length"], r["id"]))
    strata = {r["id"]: min(2, 3 * i // len(ordered)) for i, r in enumerate(ordered)}
    signers = sorted({r["signer"] for r in rows})
    require(len(signers) <= count, "pilot cannot cover every signer")
    chosen, covered = [], Counter()
    tie = lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()
    for signer in signers:
        candidates = [r for r in rows if r["signer"] == signer]
        row = min(candidates, key=lambda r: (covered[strata[r["id"]]], tie(r)))
        chosen.append(row)
        covered[strata[row["id"]]] += 1
    while len(chosen) < count:
        seen = {r["id"] for r in chosen}
        row = min((r for r in rows if r["id"] not in seen),
                  key=lambda r: (covered[strata[r["id"]]], tie(r)))
        chosen.append(row)
        covered[strata[row["id"]]] += 1
    require(set(covered) == {0, 1, 2}, "pilot fails length-tertile coverage")
    return [dict(row, length_tertile=strata[row["id"]]) for row in chosen]


def selection_partition(rows, scope):
    """Freeze the v5 input-only selection; the disclosed first item never validates."""
    require(scope in ("pilot", "full"), "unsupported selection scope")
    original = select_pilot(rows, 10)
    require(tuple(r["id"] for r in original) == FROZEN_PILOT_IDS and original[0]["id"] == CALIBRATION_ID,
            "frozen v5 selection drift; replacement validation clips forbidden")
    validation = original[1:]
    require(len(validation) == 9 and CALIBRATION_ID not in {r["id"] for r in validation},
            "calibration contamination of validation")
    require(len({r["signer"] for r in validation}) == 8 and
            {r["length_tertile"] for r in validation} == {0, 1, 2}, "validation stratum drift")
    selected = validation if scope == "pilot" else rows
    roles = {r["id"]: ("CALIBRATION" if r["id"] == CALIBRATION_ID else
                       "VALIDATION" if scope == "pilot" else "NONCALIBRATION_CENSUS") for r in selected}
    return selected, {"calibration_id": CALIBRATION_ID, "calibration_metadata": original[0],
                      "original_input_only_pilot_ids": list(FROZEN_PILOT_IDS),
                      "frozen_validation_ids": [r["id"] for r in validation],
                      "frozen_validation_metadata": validation, "execution_roles": roles,
                      "calibration_is_validation": False, "replacement_clips_allowed": False}


def configure_runtime(device):
    """Configure once before capture; fingerprint/validation never repair drift."""
    global _THREADPOOL_LIMITER
    require(device == "cpu" or re.fullmatch(r"cuda:[0-9]+", device), "invalid execution device")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.deterministic = True
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def backend_settings():
    import torch
    from threadpoolctl import threadpool_info
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "opencv_threads": cv2.getNumThreads(), "opencv_opencl": cv2.ocl.useOpenCL(),
            "opencv_optimized": cv2.useOptimized(), "torch_initial_seed": torch.initial_seed(),
            "threadpools": sorted(threadpool_info(), key=lambda row: row["filepath"])}


def enforce_backend_settings(settings, device):
    expected = {"deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_deterministic": True, "cudnn_benchmark": False,
                "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
                "float32_matmul_precision": "highest", "mkldnn_deterministic": True,
                "intraop_threads": 1, "interop_threads": 1, "opencv_threads": 1,
                "opencv_opencl": False, "torch_initial_seed": SEED}
    require(all(settings.get(key) == value for key, value in expected.items()), "actual backend/thread settings violate policy")
    require(settings["threadpools"] and all(row["num_threads"] == 1 for row in settings["threadpools"]),
            "actual numerical threadpool is not single-threaded")
    for key in ENVIRONMENT_KEYS[:5]:
        require(os.environ.get(key) == "1", f"thread environment drift: {key}")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS environment drift")
    if device == "cpu":
        import torch
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and torch.cuda.is_available() is False,
                "CPU contract requires empty CUDA visibility and unavailable CUDA")


def runtime_file_record(path):
    """Fresh content hash: file metadata alone cannot certify an unchanged binary."""
    path = Path(path).resolve(strict=True)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode), f"non-regular runtime file: {path}")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    disk = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after) == identity(disk), f"runtime input mutation while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest(),
            "identity": {"device": after.st_dev, "inode": after.st_ino, "mode": stat.S_IMODE(after.st_mode)}}


def runtime_alias(path):
    """Bind every lexical symlink component as well as the canonical target."""
    path = Path(os.path.abspath(path))
    pending, visited, links = {path}, set(), {}
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        require(len(visited) <= 128, "runtime symlink chain exceeds finite bound")
        for component in [item, *item.parents]:
            if component.is_symlink():
                target = os.readlink(component)
                links[str(component)] = target
                pending.add(Path(os.path.abspath(component.parent / target)))
    return {"path": str(path), "canonical_path": str(path.resolve(strict=True)),
            "symlinks": [{"path": p, "target": target} for p, target in sorted(links.items())]}


def prepare_runtime_capture():
    # Import/first-touch the exact support APIs before taking either module or
    # mapping snapshots. No weights, historical posterior cache or real model.
    import _hashlib
    import _json
    import _pickle
    import ctypes.util
    import torch.nn.functional
    import torch.serialization
    import torch._weights_only_unpickler
    import zipfile
    import bz2
    import lzma
    import zlib
    import numpy.lib.format
    import threadpoolctl
    require(all(sys.modules[name].__file__ for name in ("_hashlib", "_json", "resource")),
            "required hash/JSON/resource extension has no file")
    buffer = io.BytesIO()
    np.savez(buffer, fixture=np.zeros((1,), dtype=np.float32))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        require(archive["fixture"].shape == (1,), "serialization preparation failed")
    pickle.loads(pickle.dumps(np.zeros((1,), dtype=np.float32)))
    json.loads(canonical({"preparation": True}))
    ET.fromstring("<preparation/>")
    resource.getrusage(resource.RUSAGE_SELF)
    platform.platform()


def module_file_inventory():
    paths = set()
    required = {}
    for name, module in list(sys.modules.items()):
        # Read module dictionaries, not module-level lazy __getattr__ hooks.
        namespace = vars(module) if module is not None else {}
        filename = namespace.get("__file__")
        if not filename or str(filename).startswith("<"):
            continue  # builtins/frozen/namespace modules have no regular file
        path = Path(os.path.abspath(filename))
        require(path.is_file(), f"loaded module file is missing: {name}: {path}")
        paths.add(path)
        if path.suffix == ".pyc":
            try:
                source = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                source = path.with_suffix(".py")
            if source.is_file():
                paths.add(source)  # retain/hash actual bytecode AND meaningful source
        cached = namespace.get("__cached__")
        if cached and Path(cached).is_file():
            paths.add(Path(os.path.abspath(cached)))
        if name in ("_hashlib", "_json", "resource"):
            required[name] = str(path.resolve(strict=True))
    require(set(required) == {"_hashlib", "_json", "resource"}, "required runtime extension missing")
    return paths, required


def executable_mapping_records():
    records = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        require(len(parts) >= 5, "malformed executable mapping row")
        if "x" not in parts[1] or len(parts) == 5 or parts[5].startswith("["):
            continue  # anonymous/kernel mappings explicitly excluded by RUNTIME_BOUNDARY
        require(parts[5].startswith("/") and not parts[5].endswith(" (deleted)"),
                "unregistered/deleted executable mapping")
        path = Path(parts[5])
        require(path.is_file(), f"mapped executable file missing: {path}")
        disk = path.stat()
        major, minor = (int(x, 16) for x in parts[3].split(":"))
        require(disk.st_ino == int(parts[4]), "mapped executable differs from disk inode")
        if disk.st_dev != os.makedev(major, minor):
            exception = RUNTIME_BOUNDARY["mapping_device_exception"]
            require(str(path) == exception["path"] and [major, minor] == exception["mapped_device"] and
                    [os.major(disk.st_dev), os.minor(disk.st_dev)] == exception["filesystem_device"],
                    "unregistered mapped/filesystem device disagreement")
            require(str((Path("/proc/self/map_files") / parts[0]).readlink()) == str(path),
                    "mapped executable symlink target differs from disk path")
        record = {"path": str(path.resolve(strict=True)),
                  "mapped": {"device": os.makedev(major, minor), "inode": int(parts[4])},
                  "filesystem": {"device": disk.st_dev, "inode": disk.st_ino},
                  "virtualization_discrepancy": disk.st_dev != os.makedev(major, minor)}
        if record["path"] in records:
            require(records[record["path"]] == record, "inconsistent executable map segments")
        records[record["path"]] = record
    require(records, "executable mapping inventory is empty")
    return dict(sorted(records.items()))


def executable_mapping_inventory():
    return {Path(path) for path in executable_mapping_records()}


def standard_library_inventory():
    roots = sorted({Path(sysconfig.get_path(key)).resolve(strict=True) for key in ("stdlib", "platstdlib")})
    paths = set()
    for root in roots:
        for directory, folders, files in os.walk(root):
            folders[:] = sorted(n for n in folders if n not in ("site-packages", "dist-packages"))
            for name in sorted(files):
                if name.endswith((".py", ".pyc")) or ".so" in name:
                    paths.add(Path(directory) / name)
    require(paths, "standard-library immutable inventory is empty")
    return roots, paths


def native_dependencies(roots):
    """Resolve every captured ELF root transitively; unresolved dependencies fail."""
    resolver = shutil.which("ldd")
    require(resolver is not None, "ELF dependency resolver is unavailable")
    pending = {Path(p).resolve(strict=True) for p in roots}
    with Path(resolver).open("rb") as handle:
        header = handle.readline(4096)
    if header.startswith(b"#!"):
        interpreter = header[2:].decode().strip().split()[0]
        require(interpreter.startswith("/") and interpreter != "/usr/bin/env", "unbound dependency-resolver interpreter")
        pending.add(Path(interpreter).resolve(strict=True))
    known = set(pending)
    paths, resolutions = set(), {}
    while pending:
        root = min(pending)
        pending.remove(root)
        if root in paths:
            continue
        paths.add(root)
        with root.open("rb") as handle:
            if handle.read(4) != b"\x7fELF":
                continue
        result = subprocess.run([resolver, str(root)], capture_output=True, text=True, check=False, timeout=20)
        if result.returncode:
            require("statically linked" in result.stdout + result.stderr or
                    "not a dynamic executable" in result.stdout + result.stderr,
                    f"native dependency resolution failed: {root}")
        dependencies = sorted({Path(p) for p in re.findall(r"(?:=>\s+)?(/[^\s]+)\s+\(0x[0-9a-fA-F]+\)", result.stdout)})
        known.update(p.resolve(strict=True) for p in dependencies)
        inherited = {}
        for name in re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", result.stdout, re.M):
            # A standalone ldd invocation loses the importing extension's RPATH
            # context. Prefer an explicitly inventoried same-bundle sibling;
            # otherwise require a globally unique inventoried native file.
            matches = {p for p in known if p.name == name}
            siblings = {p for p in matches if p.parent == root.parent}
            if siblings:
                matches = siblings
            require(len(matches) == 1, f"unresolved/ambiguous native dependency: {root}: {name}")
            inherited[name] = str(next(iter(matches)))
        dependencies = sorted(set(dependencies) | {Path(p) for p in inherited.values()})
        resolutions[str(root)] = {"dependencies": [str(p) for p in dependencies],
                                  "standalone_ldd_inherited_rpath_resolutions": inherited,
                                  "kind": "dynamic" if dependencies else "static_or_no_dynamic_dependencies"}
        pending.update(p.resolve(strict=True) for p in dependencies if p.resolve(strict=True) not in paths)
    return paths, resolutions, Path(resolver)


def numerical_extension_roots():
    import torch
    from numpy._core import _multiarray_umath
    cv_extensions = sorted(Path(cv2.__file__).parent.glob("cv2*.so"))
    require(len(cv_extensions) == 1, "ambiguous loaded OpenCV extension")
    return [Path(_multiarray_umath.__file__).resolve(), Path(torch._C.__file__).resolve(), cv_extensions[0].resolve()]


def numerical_binary_closure():
    """Historical name retained; v4 captures the complete finite runtime boundary."""
    prepare_runtime_capture()
    resources()
    distributions, all_paths = {}, set()
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
            continue
        require(distribution.files is not None, f"runtime distribution has no file inventory: {name}")
        files = sorted({Path(os.path.abspath(distribution.locate_file(p))) for p in distribution.files
                        if str(p).endswith((".py", ".pyi", ".pyc", "RECORD", "METADATA", "WHEEL")) or ".so" in Path(str(p)).name})
        require(files, f"runtime distribution closure is empty: {name}")
        distributions[name] = {"version": distribution.version, "files": [str(p) for p in files]}
        all_paths.update(files)
    roots = numerical_extension_roots()
    all_paths.update(roots)
    stdlib_roots, stdlib_files = standard_library_inventory()
    all_paths.update(stdlib_files)
    all_paths.update(SOURCE_PATHS.values())  # AST-extracted local inference code is separately source-bound too.
    all_paths.add(Path(sys.executable))
    modules, required = module_file_inventory()
    mapping_records = executable_mapping_records()
    mappings = {Path(path) for path in mapping_records}
    all_paths.update(modules | mappings)
    native_roots = {p for p in all_paths if ".so" in p.name} | mappings | {Path(sys.executable)}
    dependencies, resolutions, resolver = native_dependencies(native_roots)
    all_paths.update(dependencies)
    all_paths.add(resolver)
    # Resolver calls must not introduce an unrecorded import or executable map.
    after_modules, after_required = module_file_inventory()
    after_mappings = executable_mapping_records()
    require(modules == after_modules and required == after_required and mapping_records == after_mappings,
            "runtime capture phase changed during dependency resolution")
    aliases = [runtime_alias(path) for path in sorted(all_paths)]
    canonical_paths = sorted({Path(alias["canonical_path"]) for alias in aliases})
    records = [runtime_file_record(path) for path in canonical_paths]
    require(aliases == [runtime_alias(path) for path in sorted(all_paths)], "runtime symlink target changed during capture")
    require(module_file_inventory()[0] == modules and executable_mapping_records() == mapping_records,
            "runtime module/mapping membership changed during hashing")
    resources()
    return {"policy": RUNTIME_BOUNDARY,
            "distributions": distributions, "loaded_extension_roots": [str(p) for p in roots],
            "standard_library_roots": [str(p) for p in stdlib_roots],
            "standard_library_files": sorted(str(p) for p in stdlib_files),
            "native_dependencies": sorted(str(p) for p in dependencies), "native_resolutions": resolutions,
            "required_extensions": required, "aliases": aliases, "files": records,
            "observations": {"loaded_module_files": sorted(str(p.resolve()) for p in modules),
                             "executable_mappings": sorted(str(p.resolve()) for p in mappings),
                             "mapped_file_identities": mapping_records},
            "sha256": object_hash(records), "inventory_sha256": object_hash(aliases)}


def runtime_fingerprint(device="cpu"):
    import torch
    settings = backend_settings()
    enforce_backend_settings(settings, device)
    versions = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    gpu = "cpu-only"
    if device.startswith("cuda:"):
        require(re.fullmatch(r"cuda:[0-9]+", device), "explicit CUDA device index required")
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                 "--format=csv,noheader,nounits", "-i", device.split(":")[1]],
                                capture_output=True, text=True, check=True, timeout=15)
        gpu = result.stdout.strip()
        require(gpu and len(gpu.splitlines()) == 1, "missing/ambiguous GPU identity")
    return {"python": sys.version, "executable": file_record(Path(sys.executable).resolve()),
            "platform": platform.platform(), "machine": platform.machine(), "packages": versions,
            "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            "device": device, "gpu": gpu, "policy": POLICY,
            "actual_backend_settings": settings, "numerical_binary_closure": numerical_binary_closure(),
            "torch_runtime": {"version": torch.__version__, "cuda_build": torch.version.cuda,
                              "cuda_available": torch.cuda.is_available(),
                              "intraop_threads": torch.get_num_threads(),
                              "interop_threads": torch.get_num_interop_threads()},
            "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}}


def validate_runtime(expected, observed):
    # Exact inventory+content equality; only current-use observations may vary
    # within that enumerated immutable inventory (e.g. a known lazy stdlib import).
    def bound(value):
        value = copy.deepcopy(value)
        if "numerical_binary_closure" in value:
            value["numerical_binary_closure"].pop("observations", None)
        return value
    require(bound(expected) == bound(observed), "runtime drift since preregistration")
    require("actual_backend_settings" in observed and "numerical_binary_closure" in observed,
            "runtime closure is incomplete")
    enforce_backend_settings(observed["actual_backend_settings"], observed["device"])
    for runtime in (expected, observed):
        closure = runtime["numerical_binary_closure"]
        require(closure["policy"] == RUNTIME_BOUNDARY, "runtime boundary drift")
        require(closure["sha256"] == object_hash(closure["files"]) and
                closure["inventory_sha256"] == object_hash(closure["aliases"]), "runtime binary manifest digest mismatch")
        files = {r["path"] for r in closure["files"]}
        require(len(files) == len(closure["files"]) and
                files == {r["canonical_path"] for r in closure["aliases"]}, "runtime inventory membership mismatch")
        require(set(closure["required_extensions"]) == {"_hashlib", "_json", "resource"} and
                set(closure["required_extensions"].values()) <= files, "required runtime extension omitted")
        observations = closure["observations"]
        require(set(observations) == {"loaded_module_files", "executable_mappings", "mapped_file_identities"},
                "runtime observations missing")
        require(all(set(paths) <= files for paths in observations.values()),
                "unregistered loaded module/executable mapping")
        require(set(observations["mapped_file_identities"]) == set(observations["executable_mappings"]),
                "mapped identity inventory mismatch")
        by_path = {r["path"]: r for r in closure["files"]}
        for path, mapping in observations["mapped_file_identities"].items():
            disk = {key: by_path[path]["identity"][key] for key in ("device", "inode")}
            require(mapping["path"] == path and mapping["filesystem"] == disk and
                    mapping["mapped"]["inode"] == disk["inode"], "mapped/disk runtime identity drift")
            discrepancy = mapping["mapped"]["device"] != disk["device"]
            require(mapping["virtualization_discrepancy"] is discrepancy, "mapping discrepancy label drift")
            if discrepancy:
                exception = RUNTIME_BOUNDARY["mapping_device_exception"]
                require(path == exception["path"] and
                        mapping["mapped"]["device"] == os.makedev(*exception["mapped_device"]) and
                        disk["device"] == os.makedev(*exception["filesystem_device"]),
                        "unregistered mapped/filesystem device disagreement")
    old_maps = expected["numerical_binary_closure"]["observations"]["mapped_file_identities"]
    new_maps = observed["numerical_binary_closure"]["observations"]["mapped_file_identities"]
    require(all(new_maps.get(path) == record for path, record in old_maps.items()),
            "mapped runtime identity disappeared/changed since preregistration")


def gpu_diagnostic():
    observation = dict(GPU_CONTEXT_OBSERVATION)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                 "--format=csv,noheader,nounits", "-i", "0"],
                                capture_output=True, text=True, check=False, timeout=15)
        observation.update(current_subprocess_returncode=result.returncode,
                           current_subprocess_stdout=result.stdout, current_subprocess_stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        observation["current_subprocess_error"] = str(error)
    return observation


def resources(device=None, preflight=False):
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    require(rss < POLICY["peak_rss_limit_bytes"], f"peak RSS exceeds <7 GiB gate: {rss}")
    memory = dict(re.findall(r"^(\w+):\s+(\d+) kB$", Path("/proc/meminfo").read_text(), re.M))
    available = int(memory["MemAvailable"]) * 1024
    if preflight:
        require(available >= POLICY["minimum_available_ram_bytes"], "available RAM below 11 GiB preflight gate")
    telemetry = {"peak_rss_bytes": rss, "available_ram_bytes": available,
                 "resource_checks": "preflight and stage/clip boundaries; no claim of a hard RSS sandbox"}
    if device and device.startswith("cuda:"):
        import torch
        free, total = torch.cuda.mem_get_info(torch.device(device))
        if preflight:
            require(free >= POLICY["minimum_free_gpu_bytes"], "free GPU memory below 11 GiB gate")
        telemetry.update(gpu_free_bytes=free, gpu_total_bytes=total,
                         gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(device)))
    return telemetry


@contextmanager
def heavy_lock(path):
    # This lock serializes this sealed raw-reproduction family. The primary
    # separately serializes unrelated project jobs before authorizing inference.
    lock = owned(path)
    require(lock.name == "heavy.lock" and lock.is_file(), "owned raw-reproduction lock is absent")
    with lock.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AuditError("another memory-heavy process owns the shared lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def frame_paths(root, sid, expected_count):
    root = Path(root).resolve(strict=True)
    folder = root / checked_id(sid)
    require(folder.is_dir() and not folder.is_symlink(), f"missing/symlink frame folder: {sid}")
    files = sorted(folder.iterdir())
    require(len(files) == expected_count, f"missing/extra frames: {sid}")
    require(files and all(p.is_file() and not p.is_symlink() and
                         re.fullmatch(r"images[0-9]{4}\.png", p.name) for p in files),
            f"unexpected/corrupt frame name or type: {sid}")
    require([int(p.stem[6:]) for p in files] == list(range(1, expected_count + 1)),
            f"duplicate/missing frame sequence: {sid}")
    require(len({(p.stat().st_dev, p.stat().st_ino) for p in files}) == len(files),
            f"duplicate/hardlinked input frames: {sid}")
    return files


def frame_inventory(raw_root, cache_root, row):
    raw = frame_paths(raw_root, row["id"], row["length"])
    cached = frame_paths(cache_root, row["id"], row["length"])
    require([p.name for p in raw] == [p.name for p in cached], "raw/cache frame names differ")
    return [{"name": a.name, "raw": file_record(a), "cached": file_record(b)} for a, b in zip(raw, cached)]


def verify_pixels(inventory, raw_root, cache_root, row):
    current = frame_inventory(raw_root, cache_root, row)
    require(current == inventory, f"frame input mutation: {row['id']}")
    total = 0
    for item in inventory:
        raw = cv2.imread(item["raw"]["path"], cv2.IMREAD_COLOR)
        cached = cv2.imread(item["cached"]["path"], cv2.IMREAD_COLOR)
        require(raw is not None and cached is not None, f"corrupt/undecodable frame: {item['name']}")
        require(raw.shape == (260, 210, 3) and cached.shape == (256, 256, 3), "unexpected RGB frame shape")
        resized = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        require(np.array_equal(resized, cached), f"256px pixel mismatch: {row['id']}/{item['name']}")
        total += int(cached.size)
    return {"frames": len(inventory), "pixel_channels": total, "exact": True}


def _definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    require({n.name for n in selected} == set(names) and len(selected) == len(names), "external source definition drift")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return tree


def inference_core_body(path):
    """One extraction definition is used for both the tested digest and execution."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_split")
    loop = next(n for n in process.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "idx")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and
                 ast.unparse(n.targets[0]) == "(padded_video, video_length)")
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For) and
               isinstance(n.target, ast.Name) and n.target.id == "g")
    body = copy.deepcopy(loop.body[start:end + 1])
    require(len(body) == 10, "external process inference core changed; review required")
    return body


def inference_core_hash(path):
    return object_hash(ast.dump(ast.Module(body=inference_core_body(path), type_ignores=[]), include_attributes=False))


def source_closure():
    sources = {key: file_record(path) for key, path in SOURCE_PATHS.items()}
    return {"sources": sources, "core_ast_sha256": inference_core_hash(sources["corrnet_cache"]["path"]),
            "import_policy": "AST allowlist; no external CLI/top-level code; local-only pinned ResNet initialization",
            "model_assets": {key: file_record(DEFAULTS[key]) for key in MODEL_ASSET_KEYS}}


def validate_source_closure(expected, check_model_assets=True):
    require(set(expected["sources"]) == set(SOURCE_PATHS), "tested source closure membership drift")
    require(expected["sources"] == {key: file_record(path) for key, path in SOURCE_PATHS.items()},
            "tested source closure drift")
    require(expected["core_ast_sha256"] == inference_core_hash(SOURCE_PATHS["corrnet_cache"]),
            "tested inference-core digest drift")
    require(set(expected["model_assets"]) == set(MODEL_ASSET_KEYS), "model asset closure membership drift")
    if check_model_assets:
        for key, record in expected["model_assets"].items():
            require(Path(record["path"]) == DEFAULTS[key].resolve(), "tested model asset path drift")
            verify_record(record)


def load_corrnet_source(sources, resnet_record, device):
    """Load only reviewed definitions, with local-only ResNet initialization."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    enforce_backend_settings(backend_settings(), device)
    for record in sources.values():
        verify_record(record)

    def local_resnet(url, *args, **kwargs):
        require(url == "https://download.pytorch.org/models/resnet18-f37072fd.pth" and not args and not kwargs,
                "unexpected model URL or download arguments")
        verify_record(resnet_record)
        return torch.load(resnet_record["path"], map_location="cpu", weights_only=True)

    common = {"torch": torch, "nn": nn, "F": F, "np": np, "copy": copy,
              "collections": __import__("collections")}
    modules = {}
    for key in ("resnet", "tconv", "bilstm"):
        path = sources[key]["path"]
        tree = ast.parse(Path(path).read_text(), filename=path)
        namespace = dict(common, model_zoo=types.SimpleNamespace(load_url=local_resnet))
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                # The reviewed ResNet file has only literal __all__/model_urls assignments.
                require(all(isinstance(t, ast.Name) and t.id in ("__all__", "model_urls") for t in node.targets),
                        f"unreviewed external module assignment: {key}")
                ast.literal_eval(node.value)
                nodes.append(node)
            elif not isinstance(node, ast.Import) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                raise AuditError(f"unreviewed external module top-level statement: {key}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
        modules[key] = namespace
    namespace = dict(common, cv2=cv2, glob=glob, OrderedDict=OrderedDict, types=types,
                     DEVICE=torch.device(device), corrnet_resnet18=modules["resnet"]["resnet18"],
                     TemporalConv=modules["tconv"]["TemporalConv"], BiLSTMLayer=modules["bilstm"]["BiLSTMLayer"])
    source = sources["corrnet_cache"]["path"]
    _definitions(source, DEFINITIONS, namespace)
    # Preserve, without rewriting, pad -> forward -> log_softmax -> token-index statements.
    body = inference_core_body(source)
    stub = ast.parse("def infer_core(model, gloss_dict, gloss_label, video):\n    pass\n").body[0]
    stub.body = body + ast.parse("return {'log_probs': log_probs, 'T_out': T_out, 'gloss_indices': gloss_indices}").body
    ast.fix_missing_locations(stub)
    exec(compile(ast.Module(body=[stub], type_ignores=[]), source, "exec"), namespace)
    ctc_namespace = {"np": np, "List": list, "Tuple": tuple}
    _definitions(sources["ctc"]["path"], ["ctc_forced_align"], ctc_namespace)
    namespace["ctc_forced_align"] = ctc_namespace["ctc_forced_align"]
    namespace["core_ast_sha256"] = object_hash(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False))
    return types.SimpleNamespace(**namespace)


def validate_posterior(entry, row, vocab_size):
    require(isinstance(entry, dict), "missing/corrupt posterior entry")
    array = np.asarray(entry.get("log_probs"))
    require(array.dtype == np.float32 and array.ndim == 2 and array.shape[1] == vocab_size and array.shape[0] > 0,
            "posterior shape/dtype mismatch")
    require(np.isfinite(array).all(), "nonfinite/NaN posterior")
    require(type(entry.get("T_out")) is int and entry["T_out"] == array.shape[0], "T_out mismatch")
    require(entry.get("gloss_indices") == row["gloss_indices"], "token drop or gloss index mismatch")
    require(all(type(i) is int and 0 < i < vocab_size for i in entry["gloss_indices"]), "invalid/blank target token")
    if "name" in entry:
        require(entry["name"] == f"train/{row['id']}", "posterior name/ID mismatch")
    if "gloss_ref" in entry:
        require(entry["gloss_ref"] == row["gloss"], "posterior gloss reference mismatch")
    require(np.allclose(np.exp(array.astype(np.float64)).sum(axis=1), 1.0, atol=2e-6, rtol=2e-6),
            "posterior is not normalized log-probability")
    return array


def checked_inference(source, model, vocab, row, video):
    """Validate model returns before the unchanged source core trims logits."""
    torch = source.torch
    expected_frames = (row["length"] + 3) // 4
    class CheckedModel:
        def __call__(self, *args):
            ret = model(*args)
            require(isinstance(ret, dict) and set(ret) == {"sequence_logits", "conv_logits", "feat_len"},
                    "unexpected CorrNet model return schema")
            lengths = ret["feat_len"]
            expected_shape = (expected_frames, 1, len(vocab) + 1)
            for name in ("sequence_logits", "conv_logits"):
                logits = ret[name]
                require(torch.is_tensor(logits) and tuple(logits.shape) == expected_shape,
                        f"invalid {name} shape; silent trimming forbidden")
                require(logits.dtype == torch.float32, f"invalid {name} dtype")
                require(bool(torch.isfinite(logits).all()), f"nonfinite {name} before trimming")
            require(torch.is_tensor(lengths) and lengths.shape == (1,) and lengths.dtype == torch.float32,
                    "invalid feat_len shape/dtype")
            require(bool(torch.isfinite(lengths).all()), "nonfinite feat_len before trimming")
            value = lengths[0].item()
            require(value.is_integer(), "nonintegral feat_len; silent trimming forbidden")
            require(value == expected_frames,
                    "model output length differs from padding contract; silent trimming forbidden")
            return ret
    return source.infer_core(CheckedModel(), vocab, row["gloss"], video)


def strict_ctc_spans(log_probs, targets):
    """Independent feasibility/backtrace check; no proportional recovery."""
    require(np.asarray(log_probs).ndim == 2 and np.isfinite(log_probs).all(), "nonfinite/malformed CTC input")
    T, V = log_probs.shape
    require(targets and all(type(x) is int and 0 < x < V for x in targets), "empty/invalid CTC target")
    require(T >= len(targets) + sum(a == b for a, b in zip(targets, targets[1:])),
            "infeasible CTC target; fallback forbidden")
    extended = [0]
    for target in targets:
        extended.extend([target, 0])
    L = len(extended)
    dp = np.full((T, L), -np.inf, dtype=np.float64)
    bp = np.zeros((T, L), dtype=np.int8)
    dp[0, 0], dp[0, 1] = log_probs[0, 0], log_probs[0, targets[0]]
    for t in range(1, T):
        for s in range(L):
            best, back = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, back = dp[t - 1, s - 1], 1
            if s >= 2 and extended[s] != 0 and extended[s] != extended[s - 2] and dp[t - 1, s - 2] > best:
                best, back = dp[t - 1, s - 2], 2
            dp[t, s], bp[t, s] = best + log_probs[t, extended[s]], back
    s = L - 1 if dp[-1, L - 1] >= dp[-1, L - 2] else L - 2
    require(np.isfinite(dp[-1, s]), "unreachable CTC terminal; fallback forbidden")
    frames = [[] for _ in targets]
    for t in range(T - 1, -1, -1):
        if extended[s] != 0:
            frames[(s - 1) // 2].append(t)
        if t:
            s -= int(bp[t, s])
    require(all(frames), "unvisited CTC token; fallback forbidden")
    return [[min(f), max(f) + 1] for f in frames]


def checked_alignment(source, log_probs, targets):
    strict = strict_ctc_spans(log_probs, targets)
    original = [list(pair) for pair in source.ctc_forced_align(log_probs, targets, blank=0)]
    require(original == strict, "legacy CTC result differs from strict no-fallback alignment")
    return strict


def decoded_tokens(log_probs):
    path = np.argmax(log_probs, axis=1).tolist()
    return [value for i, value in enumerate(path) if value != 0 and (i == 0 or value != path[i - 1])]


def entry_commitment(entry):
    """Content identity of cache tensors, independent of pickle/NPZ serialization."""
    array = np.asarray(entry["log_probs"])
    require(array.dtype == np.float32 and array.ndim == 2 and np.isfinite(array).all(),
            "invalid committed historical array")
    require(type(entry.get("T_out")) is int and entry["T_out"] == array.shape[0] and
            isinstance(entry.get("gloss_indices"), list) and entry["gloss_indices"] and
            all(type(token) is int and 0 < token < array.shape[1] for token in entry["gloss_indices"]),
            "historical commitment cannot coerce discrete fields")
    payload = {"shape": list(array.shape), "dtype": str(array.dtype),
               "C_order_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
               "gloss_indices": entry["gloss_indices"], "T_out": entry["T_out"]}
    return object_hash(payload)


def error_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1)
    require(values.size > 0 and np.isfinite(values).all(), "nonfinite/empty error distribution")
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "quantiles_linear": {str(q): float(np.quantile(values, q, method="linear")) for q in quantiles}}


def decision_diagnostics(actual, expected, row, vocab_size, actual_spans, expected_spans):
    """All errors are diagnostic. The acceptance certificate is per-frame and strict."""
    a, b = validate_posterior(actual, row, vocab_size), validate_posterior(expected, row, vocab_size)
    require(a.shape == b.shape, "posterior shape/T_out comparison mismatch")
    delta = a.astype(np.float64) - b.astype(np.float64)
    absolute = np.abs(delta)
    maxima = absolute.max(axis=1)
    historical_top_two = np.sort(np.partition(b.astype(np.float64), -2, axis=1)[:, -2:], axis=1)
    margins = historical_top_two[:, 1] - historical_top_two[:, 0]
    certified = 2.0 * maxima < margins
    current_path, historical_path = np.argmax(a, axis=1), np.argmax(b, axis=1)
    current_decoded, historical_decoded = decoded_tokens(a), decoded_tokens(b)
    return {"mode": "decision_equivalence", "deterministic_claim": False,
            "continuous_posterior_reproduction_claim": False,
            "log_probs_exact": bool(np.array_equal(a, b)), "max_abs_error": float(absolute.max()),
            "exact_continuous_elements": int(np.count_nonzero(a == b)),
            "signed_error_distribution": error_distribution(delta),
            "absolute_error_distribution": error_distribution(absolute),
            "per_frame_max_abs_error": maxima.tolist(), "historical_log_margins": margins.tolist(),
            "strict_margin_certified": certified.tolist(), "strict_margin_certified_frames": int(certified.sum()),
            "strict_margin_formula": POLICY["margin_gate"],
            "framewise_argmax_exact": bool(np.array_equal(current_path, historical_path)),
            "framewise_argmax_equal_count": int(np.count_nonzero(current_path == historical_path)),
            "current_argmax": current_path.tolist(), "historical_argmax": historical_path.tolist(),
            "decoded_tokens_exact": current_decoded == historical_decoded,
            "current_decoded_tokens": current_decoded, "historical_decoded_tokens": historical_decoded,
            "ctc_spans_exact": actual_spans == expected_spans,
            "current_ctc_spans": actual_spans, "historical_ctc_spans": expected_spans,
            "gloss_indices_exact": True, "T_out_exact": True,
            "decoded_token_count": len(current_decoded), "tokens": len(row["gloss_indices"]),
            "T_out": int(a.shape[0]), "elements": int(a.size)}


def require_decision_equivalence(diagnostic):
    require(diagnostic["framewise_argmax_exact"] is True, "framewise argmax mismatch")
    require(diagnostic["decoded_tokens_exact"] is True, "decoded token mismatch")
    require(diagnostic["ctc_spans_exact"] is True, "CTC span mismatch")
    require(all(diagnostic["strict_margin_certified"]) and
            diagnostic["strict_margin_certified_frames"] == diagnostic["T_out"],
            "strict per-frame margin certificate failed (ties/equality are rejected)")


def compare_entries(actual, expected, row, vocab_size, actual_spans, expected_spans, mode="decision_equivalence"):
    require(mode in ("deterministic", "decision_equivalence"), "unknown comparison mode; numeric tolerance cannot accept")
    diagnostic = decision_diagnostics(actual, expected, row, vocab_size, actual_spans, expected_spans)
    if mode == "deterministic":
        require(diagnostic["log_probs_exact"] and
                actual["log_probs"].tobytes() == expected["log_probs"].tobytes(), "deterministic posterior array drift")
        require(diagnostic["decoded_tokens_exact"] and diagnostic["ctc_spans_exact"], "deterministic decision/span drift")
    else:
        require_decision_equivalence(diagnostic)
    return dict(diagnostic, mode=mode, deterministic_claim=mode == "deterministic")


def v5_preregistration():
    validate_revision_lineage()
    seal = read_bound(REVISION["superseded_evidence"]["v5_prereg_seal"])
    prior = read_bound(seal["preregistration"])
    require(prior["schema"] == "raw_rgb_corrnet_ctc_reproduction_v5" and
            prior["selected_ids"] == list(FROZEN_PILOT_IDS), "bound v5 selection/schema drift")
    return prior


def calibration_measurement(actual, historical, row, spans, pixels, old_policy):
    old_spans = strict_ctc_spans(historical["log_probs"], row["gloss_indices"])
    require(spans == strict_ctc_spans(actual["log_probs"], row["gloss_indices"]), "calibration stored span drift")
    diagnostic = decision_diagnostics(actual, historical, row, actual["log_probs"].shape[1], spans, old_spans)
    continuous_pass = bool(np.allclose(actual["log_probs"], historical["log_probs"],
                                      atol=old_policy["log_probs_atol"], rtol=old_policy["log_probs_rtol"]))
    return {"role": "CALIBRATION", "is_validation": False, "id": row["id"], "metadata": row,
            "pixels": pixels, "historical_v5_continuous_gate": {
                "atol": old_policy["log_probs_atol"], "rtol": old_policy["log_probs_rtol"],
                "passed": continuous_pass, "max_abs_error": diagnostic["max_abs_error"]},
            "decision_diagnostics": diagnostic, "backend_causation": "not established; no CPU-versus-GPU assertion",
            "claim": "disclosed calibration observation only; never validation evidence"}


def derive_calibration(output, authorize_cache_read=False):
    """One explicitly scheduled historical-cache read; no model or new inference."""
    require(authorize_cache_read, "historical cache read requires explicit heavy-slot authorization")
    output = stage_dir(output, "synthetic_gate", create=False)
    require(not (output / "calibration.json").exists(), "calibration exists; never overwrite")
    prior = v5_preregistration()
    require(prior["clips"][0]["metadata"]["id"] == CALIBRATION_ID, "wrong calibration identity")
    failure = read_bound(REVISION["superseded_evidence"]["v5_pilot_failure"])
    require(failure["status"] == "FAIL" and failure["completed_clips"] == 0 and failure["active_clip"] == CALIBRATION_ID
            and failure["active_output"] == REVISION["superseded_evidence"]["v5_pilot_output"], "v5 failure lineage drift")
    write_bytes_new(output / "heavy.lock", b"")
    started = time.monotonic()
    with heavy_lock(output / "heavy.lock"):
        resources(preflight=True)
        cache_record = prior["assets"]["posterior_cache"]
        verify_record(cache_record)
        with Path(cache_record["path"]).open("rb") as handle:
            cache = pickle.load(handle)
        universe = {"train/" + row["id"] for row in prior["training_universe"]}
        require(isinstance(cache, dict) and set(cache) == universe and len(cache) == 7096,
                "calibration cache universe drift")
        resources()
        # Commitments do not inspect/evaluate new validation outcomes, select IDs,
        # or fit a threshold. They permit subsequent array-derived report checks.
        commitments = {sid.removeprefix("train/"): entry_commitment(cache[sid]) for sid in sorted(cache)}
        historical = cache["train/" + CALIBRATION_ID]
        del cache
        gc.collect()
        verify_record(cache_record)
        row = prior["clips"][0]["metadata"]
        actual, spans = load_posterior(REVISION["superseded_evidence"]["v5_pilot_output"])
        frames = read_bound(prior["clips"][0]["frames"])
        pixels = verify_pixels(frames, prior["roots"]["raw_root"], prior["roots"]["cache_root"], row)
        measurement = calibration_measurement(actual, historical, row, spans, pixels, prior["policy"])
        require(measurement["historical_v5_continuous_gate"]["passed"] is False and
                failure["error"] == "posterior tolerance failure; max_abs=" + str(measurement["decision_diagnostics"]["max_abs_error"]),
                "calibration does not reproduce the preserved v5 failure")
        historical_record = save_posterior(output / "calibration_historical.npz", historical,
                                           measurement["decision_diagnostics"]["historical_ctc_spans"])
        write_new(output / "historical_commitments.json", {"schema": SCHEMA, "historical_cache": cache_record,
                  "purpose": "content commitments only; no validation CPU outcomes or threshold fitting",
                  "entries": commitments, "count": len(commitments)})
        record = {"schema": SCHEMA, "status": "DISCLOSED_CALIBRATION_ONLY", "new_inference": False,
                  "revision": REVISION, "sources": {key: file_record(path) for key, path in SOURCE_PATHS.items()},
                  "v5_preregistration_seal": REVISION["superseded_evidence"]["v5_prereg_seal"],
                  "v5_assets": prior["assets"], "actual": REVISION["superseded_evidence"]["v5_pilot_output"],
                  "historical": historical_record, "frames": prior["clips"][0]["frames"],
                  "historical_commitments": file_record(output / "historical_commitments.json"),
                  "measurement": measurement, "elapsed_seconds": time.monotonic() - started, "resources": resources()}
        write_new(output / "calibration.json", record)
    return {"status": "DISCLOSED_CALIBRATION_ONLY", "calibration": file_record(output / "calibration.json"),
            "new_inference": False}


def validate_calibration(record):
    calibration = read_bound(record)
    schema, revision = SCHEMA, REVISION
    sources = {key: file_record(path) for key, path in SOURCE_PATHS.items()}
    if record == REVISION["superseded_evidence"].get("v6_calibration"):
        # Only this exact historical record can cross the version boundary.
        # The new gate binds current code; calibration bytes/metrics stay v6.
        old_gate = read_bound(REVISION["superseded_evidence"]["v6_gate"])
        old_seal = read_bound(REVISION["superseded_evidence"]["v6_prereg_seal"])
        old_prereg = read_bound(old_seal["preregistration"])
        require(old_gate["schema"] == "raw_rgb_corrnet_decision_equivalence_v6" and
                old_gate["revision"]["version"] == 6 and old_gate["calibration"] == record and
                old_prereg["calibration"] == record and old_prereg["gate"] == REVISION["superseded_evidence"]["v6_gate"] and
                old_prereg["policy"] == POLICY and old_prereg["claim_limit"] == CLAIM_LIMIT and
                old_prereg["selection_algorithm"] == SELECTION,
                "pinned v6 calibration protocol drift")
        require(old_gate["sources"] == old_gate["tested_closure"]["sources"] and
                all(old_gate["sources"][key] == value for key, value in sources.items()
                    if key not in ("wrapper", "tests")), "pinned calibration external source drift")
        schema, revision, sources = old_gate["schema"], old_gate["revision"], old_gate["sources"]
    require(calibration["schema"] == schema and calibration["status"] == "DISCLOSED_CALIBRATION_ONLY"
            and calibration["new_inference"] is False and calibration["revision"] == revision,
            "calibration schema/status/lineage drift")
    require(calibration["sources"] == sources, "calibration derivation source drift")
    prior = v5_preregistration()
    require(calibration["v5_preregistration_seal"] == REVISION["superseded_evidence"]["v5_prereg_seal"] and
            calibration["v5_assets"] == prior["assets"] and
            calibration["actual"] == REVISION["superseded_evidence"]["v5_pilot_output"] and
            calibration["frames"] == prior["clips"][0]["frames"], "calibration input lineage drift")
    commitments = read_bound(calibration["historical_commitments"])
    require(commitments["schema"] == schema and commitments["historical_cache"] == prior["assets"]["posterior_cache"]
            and commitments["count"] == len(commitments["entries"]) == 7096 and
            set(commitments["entries"]) == {r["id"] for r in prior["training_universe"]}
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in commitments["entries"].values()),
            "historical reference commitment universe drift")
    actual, spans = load_posterior(calibration["actual"])
    historical, historical_spans = load_posterior(calibration["historical"])
    require(entry_commitment(historical) == commitments["entries"][CALIBRATION_ID], "calibration historical tensor drift")
    row = prior["clips"][0]["metadata"]
    pixels = {"frames": row["length"], "pixel_channels": row["length"] * 256 * 256 * 3, "exact": True}
    measurement = calibration_measurement(actual, historical, row, spans, pixels, prior["policy"])
    require(historical_spans == measurement["decision_diagnostics"]["historical_ctc_spans"] and
            calibration["measurement"] == measurement, "calibration metric or role drift")
    return calibration


def validate_claims(report):
    require(report.get("claim_limit") == CLAIM_LIMIT, "claim ceiling changed")
    require(report.get("raw_pose_producers_reproduced") is False, "unsupported raw-pose producer claim")
    require(report.get("historical_continuous_posteriors_reproduced") is False and
            report.get("historical_backend_reproduced") is False, "unsupported historical posterior/backend reproduction claim")
    require(report.get("scope") in ("pilot", "full", "synthetic"), "unknown claim scope")
    require(report.get("generalizes_to_full_training") is False, "pilot promoted to a full-training/generalization claim")
    require(type(report.get("full_training_census_complete")) is bool, "missing or ill-typed full-training census flag")
    require(report["scope"] == "full" or report["full_training_census_complete"] is False,
            "pilot/synthetic report promoted to a full-training census claim")


def validate_gate(gate_path, expected_hash):
    record = file_record(owned(gate_path))
    require(record["sha256"] == expected_hash, "synthetic gate hash drift")
    gate = read_bound(record)
    require(gate.get("status") == "PASS" and gate.get("real_inference") is False, "synthetic gate incomplete")
    require(gate.get("schema") == SCHEMA, "synthetic gate schema drift")
    require(gate.get("revision") == REVISION, "synthetic gate revision lineage drift")
    validate_revision_lineage()
    validate_source_closure(gate["tested_closure"])
    validate_calibration(gate["calibration"])
    require(gate["sources"] == gate["tested_closure"]["sources"] and
            gate["core_ast_sha256"] == gate["tested_closure"]["core_ast_sha256"], "synthetic gate closure inconsistency")
    validate_claims(gate)
    verify_record(gate["unit_tests"])
    tree = ET.parse(gate["unit_tests"]["path"])
    cases = list(tree.iter("testcase"))
    require(cases and not list(tree.iter("failure")) and not list(tree.iter("error"))
            and not list(tree.iter("skipped")), "synthetic unit suite did not fully pass")
    return record


def validate_pilot_review(evidence):
    require(isinstance(evidence, dict) and set(evidence) == {"pilot", "repeat", "review"},
            "full preregistration requires a pinned pilot, exact repeat, and independent review")
    comparison = compare_runs(evidence["pilot"], evidence["repeat"])
    require(comparison["validation_closure_verified"] is True and comparison["clips"] == 9,
            "full preregistration lacks a successful nine-clip pilot/repeat")
    review = read_bound(evidence["review"])
    require(review.get("status") == "PASS" and review.get("scope") == "v7_independent_pilot_repeat_review"
            and review.get("inputs") == [evidence["pilot"], evidence["repeat"]]
            and review.get("approve_full_preregistration") is True and isinstance(review.get("reviewer"), str)
            and review["reviewer"].strip(), "independent pilot/repeat review does not approve full preregistration")
    return pinned_run_document(evidence["pilot"])


def preregister(output, gate_path, gate_hash, scope="pilot", device="cpu", pilot_report=None, pilot_hash=None,
                pilot_repeat=None, pilot_repeat_hash=None, pilot_review=None, pilot_review_hash=None):
    require(scope in ("pilot", "full"), "unsupported preregistration scope")
    require(device == "cpu", "v7 is an explicit CPU-only protocol, not a backend fallback")
    gate = validate_gate(gate_path, gate_hash)
    tested_gate = read_bound(gate)
    calibration = validate_calibration(tested_gate["calibration"])
    configure_runtime(device)
    load_corrnet_source(tested_gate["tested_closure"]["sources"],
                        tested_gate["tested_closure"]["model_assets"]["resnet_init"], device)
    runtime = runtime_fingerprint(device)
    validate_runtime(tested_gate["tested_runtime"], runtime)
    runtime_observations = runtime["numerical_binary_closure"]["observations"]
    runtime = tested_gate["tested_runtime"]  # retain the exact tested inventory and gate observations
    # Original input-only selection is frozen; calibration is removed, never replaced.
    metadata_assets = {key: file_record(DEFAULTS[key]) for key in ("manifest", "info", "vocab")}
    require(metadata_assets["vocab"]["sha256"].startswith("161a2aa7"), "wrong CorrNet vocabulary")
    with DEFAULTS["manifest"].open() as handle:
        manifest = json.load(handle)
    info = np.load(DEFAULTS["info"], allow_pickle=True).item()
    vocab = np.load(DEFAULTS["vocab"], allow_pickle=True).item()
    for record in metadata_assets.values():
        verify_record(record)
    rows = validate_metadata(manifest, info, vocab)
    require(len(rows) == 7096 and len({r["signer"] for r in rows}) == 9, "PHOENIX training denominator drift")
    selected, partition = selection_partition(rows, scope)
    output = stage_dir(output, "prereg")
    start = time.monotonic()
    write_new(output / "start.json", {"stage": "preregister", "scope": scope, "pid": os.getpid()})
    try:
        assets = dict(tested_gate["tested_closure"]["model_assets"])
        assets.update({k: file_record(v) for k, v in DEFAULTS.items() if not k.endswith("_root") and k not in assets})
        require(all(assets[key] == record for key, record in metadata_assets.items()), "metadata input mutation before seal")
        require(assets["checkpoint"]["sha256"].startswith("e0e7e567"), "wrong CorrNet checkpoint")
        require(assets["vocab"]["sha256"].startswith("161a2aa7"), "wrong CorrNet vocabulary")
        require(assets == calibration["v5_assets"], "assets changed since disclosed v5 calibration")
        sources = tested_gate["tested_closure"]["sources"]
        validate_source_closure(tested_gate["tested_closure"], check_model_assets=False)
        gpu_diagnostics = None
        if device == "cpu":
            write_new(output / "gpu_context_diagnostic.json", gpu_diagnostic())
            gpu_diagnostics = file_record(output / "gpu_context_diagnostic.json")
        pilot_evidence = None
        if scope == "full":
            require(all((pilot_report, pilot_hash, pilot_repeat, pilot_repeat_hash, pilot_review, pilot_review_hash)),
                    "full preregistration requires pilot, exact repeat, and independent review hashes")
            pilot_evidence = {key: file_record(owned(path)) for key, path in
                              (("pilot", pilot_report), ("repeat", pilot_repeat), ("review", pilot_review))}
            require([pilot_evidence[k]["sha256"] for k in ("pilot", "repeat", "review")] ==
                    [pilot_hash, pilot_repeat_hash, pilot_review_hash], "pilot/repeat/review evidence hash drift")
            pilot = validate_pilot_review(pilot_evidence)
            require(pilot["scope"] == "pilot" and pilot["assets"] == assets and pilot["sources"] == sources,
                    "full-run assets/sources differ from passed pilot")
            validate_runtime(pilot["runtime"], runtime)
            lock = pilot["lock"]
            verify_record(lock)
        else:
            write_bytes_new(output / "heavy.lock", b"")
            lock = file_record(output / "heavy.lock")
        clips = []
        for row in selected:
            require(row["length"] <= POLICY["maximum_frames_per_clip"], "clip exceeds resource length ceiling")
            frames = frame_inventory(DEFAULTS["raw_root"], DEFAULTS["cache_root"], row)
            path = output / "frame_inventories" / f"{row['id']}.json"
            write_new(path, frames)
            clips.append({"metadata": row, "role": partition["execution_roles"][row["id"]], "frames": file_record(path)})
        document = {"schema": SCHEMA, "stage": "preregistration", "scope": scope, "policy": POLICY,
                    "revision": REVISION, "tested_closure": tested_gate["tested_closure"],
                    "core_ast_sha256": tested_gate["core_ast_sha256"], "vocab_size": len(vocab) + 1,
                    "selection_algorithm": SELECTION, "selection_uses_model_outputs": False,
                    "calibration": tested_gate["calibration"], "partition": partition,
                    "historical_commitments": calibration["historical_commitments"],
                    "training_universe": rows, "training_count": len(rows), "clips": clips,
                    "selected_ids": [r["metadata"]["id"] for r in clips], "gate": gate,
                    "sources": sources, "assets": assets, "runtime": runtime,
                    "runtime_capture_observations": runtime_observations,
                    "roots": {k: str(v.resolve()) for k, v in DEFAULTS.items() if k.endswith("_root")},
                    "pilot_evidence": pilot_evidence, "claim_limit": CLAIM_LIMIT,
                    "gpu_diagnostics": gpu_diagnostics, "lock": lock,
                    "elapsed_seconds": time.monotonic() - start, "resources": resources()}
        write_new(output / "preregistration.json", document)
        seal = {"schema": SCHEMA, "preregistration": file_record(output / "preregistration.json")}
        write_new(output / "seal.json", seal)
        return {"status": "PREREGISTERED", "output": str(output), "selected_ids": document["selected_ids"],
                "seal_sha256": sha256(output / "seal.json"), "inference_run": False}
    except BaseException as error:
        write_new(output / "failure.json", {"status": "FAIL", "stage": "preregister", "error": str(error)})
        raise


def load_prereg(output, expected_seal_hash, check_inputs=True):
    output = stage_dir(output, "prereg", create=False)
    seal_path = owned(output / "seal.json")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_seal_hash or ""), "explicit seal SHA256 required")
    require(sha256(seal_path) == expected_seal_hash, "immutable preregistration seal drift")
    seal = json.loads(seal_path.read_text())
    require(Path(seal["preregistration"]["path"]) == output / "preregistration.json", "seal points outside preregistration")
    doc = read_bound(seal["preregistration"])
    require(doc["schema"] == SCHEMA and doc["policy"] == POLICY and doc["claim_limit"] == CLAIM_LIMIT
            and doc.get("revision") == REVISION,
            "preregistration policy/claim drift")
    validate_revision_lineage()
    gate = read_bound(doc["gate"])
    require(doc["tested_closure"] == gate["tested_closure"] and doc["sources"] == gate["tested_closure"]["sources"]
            and doc["core_ast_sha256"] == gate["core_ast_sha256"], "preregistration differs from tested source closure")
    require(doc["runtime"] == gate["tested_runtime"], "preregistration differs from tested runtime closure")
    require(all(doc["assets"][key] == gate["tested_closure"]["model_assets"][key] for key in MODEL_ASSET_KEYS),
            "preregistered model assets differ from gate")
    validate_source_closure(doc["tested_closure"], check_model_assets=check_inputs)
    require(doc["calibration"] == gate["calibration"], "calibration differs from tested gate")
    calibration = validate_calibration(doc["calibration"])
    require(doc["assets"] == calibration["v5_assets"] and
            doc["historical_commitments"] == calibration["historical_commitments"], "calibration/reference asset drift")
    rows = doc["training_universe"]
    expected, partition = selection_partition(rows, doc["scope"])
    require([x["metadata"] for x in doc["clips"]] == expected and doc["selected_ids"] == [x["id"] for x in expected],
            "preregistered ID/stratum selection drift")
    require(doc["training_count"] == len(rows) == 7096, "training universe denominator drift")
    require(len(doc["clips"]) == (9 if doc["scope"] == "pilot" else len(rows)), "selected denominator drift")
    require(doc["partition"] == partition and
            [clip["role"] for clip in doc["clips"]] == [partition["execution_roles"][row["id"]] for row in expected],
            "calibration/validation partition drift")
    require(doc["runtime"]["device"] == "cpu", "v7 runtime must remain CPU-only")
    if doc["scope"] == "full":
        validate_pilot_review(doc["pilot_evidence"])
    else:
        require(doc.get("pilot_evidence") is None, "unexpected pilot evidence in nine-clip preregistration")
    require(doc["selection_uses_model_outputs"] is False and doc["selection_algorithm"] == SELECTION,
            "selection contract drift")
    if check_inputs:
        load_corrnet_source(doc["sources"], doc["assets"]["resnet_init"], doc["runtime"]["device"])
        validate_runtime(doc["runtime"], runtime_fingerprint(doc["runtime"]["device"]))
        for key, record in doc["assets"].items():
            if key in MODEL_ASSET_KEYS:
                continue  # Already checked against the tested closure above.
            verify_record(record)
        verify_record(doc["lock"])
        if doc.get("gpu_diagnostics"):
            verify_record(doc["gpu_diagnostics"])
        for clip in doc["clips"]:
            inventory = read_bound(clip["frames"])
            require(inventory == frame_inventory(doc["roots"]["raw_root"], doc["roots"]["cache_root"], clip["metadata"]),
                    "preregistered frame input drift")
    return doc


def save_posterior(path, entry, spans):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, log_probs=entry["log_probs"], gloss_indices=np.asarray(entry["gloss_indices"], dtype=np.int64),
                 T_out=np.asarray(entry["T_out"], dtype=np.int64), ctc_spans=np.asarray(spans, dtype=np.int64),
                 decoded_tokens=np.asarray(decoded_tokens(entry["log_probs"]), dtype=np.int64))
    path.chmod(0o444)
    return file_record(path)


def load_posterior(record):
    canonical_record_path(record)
    verify_record(record)
    with np.load(record["path"], allow_pickle=False) as data:
        require(set(data.files) == {"log_probs", "gloss_indices", "T_out", "ctc_spans", "decoded_tokens"}, "unexpected posterior output schema")
        require(data["gloss_indices"].dtype == np.int64 and data["gloss_indices"].ndim == 1 and
                data["T_out"].dtype == np.int64 and data["T_out"].shape == () and
                data["ctc_spans"].dtype == np.int64 and data["ctc_spans"].shape == (len(data["gloss_indices"]), 2) and
                data["decoded_tokens"].dtype == np.int64 and data["decoded_tokens"].ndim == 1,
                "posterior discrete-field dtype/shape drift")
        entry = {"log_probs": data["log_probs"], "gloss_indices": data["gloss_indices"].tolist(), "T_out": int(data["T_out"])}
        spans = data["ctc_spans"].tolist()
        require(data["decoded_tokens"].tolist() == decoded_tokens(entry["log_probs"]), "stored decoded tokens differ from arrays")
    return entry, spans


def canonical_record_path(record):
    raw = record.get("path")
    require(isinstance(raw, str) and Path(raw).is_absolute(), "record path must be canonical and absolute")
    path = owned(raw)
    require(raw == str(path) and str(path.resolve()) == raw, "record path alias is forbidden")
    return path


def record_inode(record):
    stat = canonical_record_path(record).stat()
    return (stat.st_dev, stat.st_ino)


def pinned_run_document(record):
    path = canonical_record_path(record)
    require(path.name == "report.json", "run record must identify report.json")
    report = read_bound(record)
    require(report.get("report_path") == str(path), "run record/report path mismatch")
    return report


def scientific_clip(clip):
    return {key: clip[key] for key in ("id", "role", "pixels", "comparison")} | {
        "current_npz_sha256": clip["output"]["sha256"],
        "historical_npz_sha256": clip["historical"]["sha256"],
        "diagnostics_sha256": clip["diagnostics"]["sha256"]}


def scientific_document(prereg, seal_hash, clips, counts):
    return {"schema": SCHEMA, "scope": prereg["scope"], "preregistration_seal_sha256": seal_hash,
            "policy": POLICY, "claim_limit": CLAIM_LIMIT, "partition": prereg["partition"],
            "calibration_sha256": prereg["calibration"]["sha256"],
            "selected_ids": prereg["selected_ids"], "counts": dict(counts),
            "role_counts": dict(Counter(c["role"] for c in clips)),
            "clips": [scientific_clip(clip) for clip in clips],
            "continuous_errors_are_diagnostic_only": True,
            "distribution_scope": "per-clip signed/absolute quantiles and per-frame errors; complete distributions reconstructible from retained current/historical arrays"}


def validate_interval(report, start):
    finish_path = canonical_record_path(report["finish"])
    require(finish_path == Path(report["run_dir"]) / "finish.json" and
            record_inode(report["finish"]) != record_inode(report["start"]), "finish/start path alias")
    finish = read_bound(report["finish"])
    require(finish["run_id"] == start["run_id"] and finish["run_dir"] == start["run_dir"] and
            finish["boot_id"] == start["boot_id"] and isinstance(start["boot_id"], str) and start["boot_id"],
            "run finish identity mismatch")
    for clock in ("unix", "monotonic"):
        first, last = start[f"started_{clock}_ns"], finish[f"finished_{clock}_ns"]
        require(type(first) is int and type(last) is int and 0 < first < last, "invalid/same run start/end time")
    return finish


def validate_census_claim(report, prereg, counts):
    """Called only after every item's arrays, comparisons and counts are checked."""
    scope = prereg["scope"]
    require(scope in ("pilot", "full") and report["scope"] == scope, "census scope differs from seal")
    require(report["kind"] in (scope, "repeat"), "run kind differs from sealed census scope")
    require(type(report.get("full_training_census_complete")) is bool and
            report["full_training_census_complete"] is (scope == "full"), "full-training census flag differs from sealed scope")
    if scope == "full":
        rows = prereg["training_universe"]
        ids = [row["id"] for row in rows]
        clips = report["clips"]
        require(type(prereg["training_count"]) is int and prereg["training_count"] == len(ids) == len(set(ids)) == 7096
                and prereg["selected_ids"] == report["selected_ids"] == ids
                and [clip["metadata"] for clip in prereg["clips"]] == rows
                and [clip["id"] for clip in clips] == ids
                and type(report["completed_clips"]) is int and type(report["expected_clips"]) is int
                and report["completed_clips"] == report["expected_clips"] == 7096,
                "full-training census item coverage incomplete")
        roles = {sid: "CALIBRATION" if sid == CALIBRATION_ID else "NONCALIBRATION_CENSUS" for sid in ids}
        require(CALIBRATION_ID in ids and prereg["partition"]["execution_roles"] == roles
                and [clip["role"] for clip in prereg["clips"]] == [roles[sid] for sid in ids]
                and [clip["role"] for clip in clips] == [roles[sid] for sid in ids]
                and all(type(value) is int for value in report["role_counts"].values())
                and report["role_counts"] == {"CALIBRATION": 1, "NONCALIBRATION_CENSUS": 7095},
                "full-training census calibration/role coverage incomplete")
        frames = sum(row["length"] for row in rows)
        times = sum((row["length"] + 3) // 4 for row in rows)
        expected = {"frames": frames, "pixel_channels": frames * 256 * 256 * 3,
                    "time_steps": times, "tokens": sum(len(row["gloss_indices"]) for row in rows),
                    "posterior_elements": times * prereg["vocab_size"]}
        require(all(type(counts.get(key)) is int and counts[key] == value for key, value in expected.items()),
                "full-training census reconstructed denominator mismatch")


def validate_run_report(report):
    """Reconstruct all counts from pinned metadata/frame lists and actual NPZ arrays."""
    require(report.get("schema") == SCHEMA and report.get("status") == "PASS", "run did not pass")
    validate_claims(report)
    require(report.get("result_kind") == "individual_decision_equivalence" and report.get("deterministic_repeat_claim") is False
            and report.get("validation_closure_verified") is False
            and report.get("historical_agreement_mode") == POLICY["historical_comparison"],
            "individual run cannot promote decision agreement to a deterministic-repeat claim")
    require(not {"deterministic_repeat_verified", "all_item_semantic_census_verified"}.intersection(report),
            "comparison-only verification claims are forbidden in individual run reports")
    require(report.get("kind") in ("pilot", "full", "repeat"), "invalid run kind")
    directory = stage_dir(report["run_dir"], report["kind"], create=False)
    require(str(directory) == report["run_dir"] and report["report_path"] == str(directory / "report.json"), "run directory/path alias")
    start_path = canonical_record_path(report["start"])
    require(start_path == directory / "start.json", "start record outside run directory")
    start = read_bound(report["start"])
    require(re.fullmatch(r"[0-9a-f]{32}", report["run_id"]) and start["run_id"] == report["run_id"] and
            start["run_dir"] == str(directory) and start["stage"] == report["kind"] and
            type(start["started_unix_ns"]) is int and start["started_unix_ns"] > 0, "run ID/start record mismatch")
    finish = validate_interval(report, start)
    seal = report["preregistration_seal"]
    seal_path = canonical_record_path(seal)
    require(seal_path.name == "seal.json" and seal["sha256"] == report["preregistration_seal_sha256"] == start["seal_sha256"],
            "run preregistration seal mismatch")
    verify_record(seal)
    prereg = load_prereg(seal_path.parent, seal["sha256"], check_inputs=False)
    for key in ("scope", "sources", "assets", "runtime", "core_ast_sha256", "partition"):
        require(report[key] == prereg[key], f"run {key} differs from pinned preregistration")
    ids = [clip["metadata"]["id"] for clip in prereg["clips"]]
    clips = report.get("clips", [])
    require(ids and report["selected_ids"] == ids and len(ids) == len(set(ids)) and [c["id"] for c in clips] == ids,
            "duplicate/missing output ID")
    require(len(ids) == (9 if report["scope"] == "pilot" else 7096), "wrong run denominator")
    require(report.get("skips") == 0 and report.get("fallbacks") == 0, "skip/fallback in run")
    require(report.get("completed_clips") == report.get("expected_clips") == len(ids), "incomplete run denominator")
    counts = Counter(frames=0, pixel_channels=0, posterior_elements=0, tokens=0, time_steps=0, decoded_tokens=0)
    commitments = read_bound(prereg["historical_commitments"])["entries"]
    artifacts, inodes = set(), set()
    for clip, registered in zip(clips, prereg["clips"]):
        row = registered["metadata"]
        require(clip.get("role") == registered["role"] and
                (clip["role"] != "VALIDATION" or row["id"] != CALIBRATION_ID), "calibration/validation role drift")
        frame_list = read_bound(registered["frames"])
        frame_count = len(frame_list)
        require(frame_count == row["length"] and len({f["name"] for f in frame_list}) == frame_count,
                "pinned frame inventory denominator mismatch")
        for key, folder, suffix in (("output", "posteriors", "npz"), ("historical", "historical", "npz"),
                                    ("diagnostics", "diagnostics", "json")):
            path = canonical_record_path(clip[key])
            require(path == directory / folder / f"{row['id']}.{suffix}", "posterior/diagnostic artifact outside its run/ID path")
            inode = record_inode(clip[key])
            require(path not in artifacts and inode not in inodes, "duplicate/aliased posterior artifact")
            artifacts.add(path)
            inodes.add(inode)
        entry, spans = load_posterior(clip["output"])
        array = validate_posterior(entry, row, prereg["vocab_size"])
        require(entry["T_out"] == (frame_count + 3) // 4, "actual array time/frame denominator mismatch")
        require(spans == strict_ctc_spans(array, row["gloss_indices"]), "stored CTC spans differ from actual arrays")
        historical, historical_spans = load_posterior(clip["historical"])
        require(entry_commitment(historical) == commitments[row["id"]], "historical array differs from pinned cache commitment")
        require(historical_spans == strict_ctc_spans(historical["log_probs"], row["gloss_indices"]), "historical CTC span drift")
        derived_comparison = compare_entries(entry, historical, row, prereg["vocab_size"], spans, historical_spans)
        derived = {"elements": int(array.size), "tokens": len(entry["gloss_indices"]), "T_out": int(array.shape[0]),
                   "decoded_token_count": len(decoded_tokens(array))}
        require(all(type(clip["comparison"].get(k)) is int and clip["comparison"][k] == v for k, v in derived.items()),
                "per-clip denominator differs from actual arrays")
        require(clip["pixels"]["frames"] == frame_count and clip["pixels"]["pixel_channels"] == frame_count * 256 * 256 * 3,
                "per-clip frame denominator differs from pinned inventory")
        require(clip["pixels"]["exact"] is True and all(clip["comparison"][key] is True for key in
                ("framewise_argmax_exact", "gloss_indices_exact", "T_out_exact", "ctc_spans_exact", "decoded_tokens_exact")),
                "failed equality promoted to PASS")
        require(clip["comparison"] == derived_comparison and read_bound(clip["diagnostics"]) == derived_comparison,
                "scientific comparison differs from actual/reference arrays")
        counts.update(frames=frame_count, pixel_channels=frame_count * 256 * 256 * 3,
                      posterior_elements=int(array.size), tokens=len(entry["gloss_indices"]),
                      time_steps=int(array.shape[0]), decoded_tokens=len(decoded_tokens(array)))
    require(all(type(report.get(key)) is int and report[key] == value for key, value in counts.items()),
            "run denominator differs from pinned inventory/actual arrays")
    require(type(report.get("exact_continuous_clips")) is int and report["exact_continuous_clips"] ==
            sum(c["comparison"]["log_probs_exact"] is True for c in clips), "historical exact-clip denominator mismatch")
    scientific = scientific_document(prereg, seal["sha256"], clips, counts)
    require(canonical_record_path(report["scientific_report"]) == directory / "scientific.json" and
            read_bound(report["scientific_report"]) == scientific, "scientific report is not array/partition-derived")
    require(report["role_counts"] == scientific["role_counts"], "calibration/validation role denominator mismatch")
    validate_census_claim(report, prereg, counts)
    return {"preregistration": prereg, "counts": dict(counts), "artifact_paths": artifacts, "artifact_inodes": inodes,
            "start": start, "finish": finish, "scientific": scientific}


def execute(prereg, seal_hash, output, kind, authorize_inference=False, reference_report=None, reference_hash=None):
    require(authorize_inference, "real inference has not been explicitly authorized")
    require(kind in ("pilot", "full", "repeat"), "unknown execution kind")
    output = stage_dir(output, kind)
    started = time.monotonic()
    run_id = uuid.uuid4().hex
    start_record = {"stage": kind, "pid": os.getpid(), "seal_sha256": seal_hash,
                    "run_id": run_id, "run_dir": str(output), "started_unix_ns": time.time_ns(),
                    "started_monotonic_ns": time.monotonic_ns(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    write_new(output / "start.json", start_record)
    completed = []
    active_clip, active_output = None, None
    active_historical, active_diagnostics = None, None
    try:
        sealed = load_prereg(prereg, seal_hash, check_inputs=False)
        configure_runtime(sealed["runtime"]["device"])
        verify_record(sealed["lock"])
        with heavy_lock(sealed["lock"]["path"]):
            resources(preflight=True)
            doc = load_prereg(prereg, seal_hash)
            require(kind == doc["scope"] or kind == "repeat", "wrong preregistration scope for run")
            bound_reference = None
            if kind == "repeat":
                require(reference_report and reference_hash, "repeat requires a pinned reference before inference")
                bound_reference = file_record(owned(reference_report))
                require(bound_reference["sha256"] == reference_hash, "repeat reference report hash drift")
                prior_report = pinned_run_document(bound_reference)
                prior_audit = validate_run_report(prior_report)
                require(prior_report["kind"] == doc["scope"] and prior_report["preregistration_seal_sha256"] == seal_hash,
                        "repeat requires the first run under the exact same preregistration")
                require_nonoverlap(prior_audit, {"start": start_record})
            device = doc["runtime"]["device"]
            resources(device, preflight=True)
            vocab = np.load(doc["assets"]["vocab"]["path"], allow_pickle=True).item()
            # Load the 1.08 GB historical cache once; keep only the selected entries.
            with Path(doc["assets"]["posterior_cache"]["path"]).open("rb") as handle:
                cache = pickle.load(handle)
            require(isinstance(cache, dict) and set(cache) == {"train/" + r["id"] for r in doc["training_universe"]},
                    "missing/extra posterior-cache ID")
            reference = {sid: cache["train/" + sid] for sid in doc["selected_ids"]}
            del cache
            gc.collect()
            for clip in doc["clips"]:
                validate_posterior(reference[clip["metadata"]["id"]], clip["metadata"], len(vocab) + 1)
            commitments = read_bound(doc["historical_commitments"])["entries"]
            require(all(entry_commitment(reference[sid]) == commitments[sid] for sid in doc["selected_ids"]),
                    "historical reference differs from pre-validation cache commitment")
            resources(device)
            source = load_corrnet_source(doc["sources"], doc["assets"]["resnet_init"], device)
            require(source.core_ast_sha256 == doc["core_ast_sha256"], "executed inference-core digest differs from tested gate")
            model = source.build_corrnet_model(vocab, doc["assets"]["checkpoint"]["path"])
            require(not model.training, "CorrNet unexpectedly in training mode")
            resources(device)
            for clip in doc["clips"]:
                row = clip["metadata"]
                active_clip, active_output = row["id"], None
                active_historical, active_diagnostics = None, None
                frames = read_bound(clip["frames"])
                pixels = verify_pixels(frames, doc["roots"]["raw_root"], doc["roots"]["cache_root"], row)
                pattern = str(Path(doc["roots"]["cache_root"]) / row["id"] / "*.png")
                video, count = source.load_and_preprocess_video(pattern)
                require(video is not None and count == row["length"] and tuple(video.shape) == (count, 3, 224, 224),
                        "preprocessor skipped/fell back on frames")
                require(bool(source.torch.isfinite(video).all()), "nonfinite preprocessed video")
                actual = checked_inference(source, model, vocab, row, video)
                expected = reference[row["id"]]
                validate_posterior(actual, row, len(vocab) + 1)
                validate_posterior(expected, row, len(vocab) + 1)
                spans = checked_alignment(source, actual["log_probs"], actual["gloss_indices"])
                require(frames == frame_inventory(doc["roots"]["raw_root"], doc["roots"]["cache_root"], row),
                        "input mutation during inference")
                record = save_posterior(output / "posteriors" / f"{row['id']}.npz", actual, spans)
                active_output = record
                old_spans = checked_alignment(source, expected["log_probs"], expected["gloss_indices"])
                active_historical = save_posterior(output / "historical" / f"{row['id']}.npz", expected, old_spans)
                comparison = decision_diagnostics(actual, expected, row, len(vocab) + 1, spans, old_spans)
                write_new(output / "diagnostics" / f"{row['id']}.json", comparison)
                active_diagnostics = file_record(output / "diagnostics" / f"{row['id']}.json")
                require_decision_equivalence(comparison)
                result = {"id": row["id"], "role": clip["role"], "pixels": pixels, "comparison": comparison, "output": record,
                          "historical": active_historical, "diagnostics": active_diagnostics,
                          "resources": resources(device)}
                write_new(output / "clips" / f"{row['id']}.json", result)
                completed.append(result)
                print(json.dumps({"stage": kind, "completed": len(completed), "expected": len(doc["clips"]),
                                  "id": row["id"], "decision_equivalent": True,
                                  "continuous_exact_diagnostic": comparison["log_probs_exact"]}), flush=True)
                del video, actual, expected
            del model, reference
            gc.collect()
            # Recheck every input and the externally pinned seal after all computation.
            load_prereg(prereg, seal_hash)
            write_new(output / "finish.json", {"run_id": run_id, "run_dir": str(output),
                      "boot_id": start_record["boot_id"], "finished_unix_ns": time.time_ns(),
                      "finished_monotonic_ns": time.monotonic_ns(),
                      "boundary": "all model/input/runtime checks complete; before final report validation/serialization"})
            report = {"schema": SCHEMA, "status": "PASS", "scope": doc["scope"], "kind": kind,
                      "result_kind": "individual_decision_equivalence", "deterministic_repeat_claim": False,
                      "validation_closure_verified": False,
                      "historical_agreement_mode": POLICY["historical_comparison"],
                      "run_id": run_id, "run_dir": str(output), "report_path": str(output / "report.json"),
                      "start": file_record(output / "start.json"),
                      "finish": file_record(output / "finish.json"), "partition": doc["partition"],
                      "preregistration_seal": file_record(Path(prereg) / "seal.json"),
                      "preregistration_seal_sha256": seal_hash, "selected_ids": doc["selected_ids"],
                      "completed_clips": len(completed), "expected_clips": len(doc["clips"]), "clips": completed,
                      "frames": sum(c["pixels"]["frames"] for c in completed),
                      "posterior_elements": sum(c["comparison"]["elements"] for c in completed),
                      "pixel_channels": sum(c["pixels"]["pixel_channels"] for c in completed),
                      "tokens": sum(c["comparison"]["tokens"] for c in completed),
                      "time_steps": sum(c["comparison"]["T_out"] for c in completed),
                      "decoded_tokens": sum(c["comparison"]["decoded_token_count"] for c in completed),
                      "exact_continuous_clips": sum(c["comparison"]["log_probs_exact"] for c in completed),
                      "role_counts": dict(Counter(c["role"] for c in completed)),
                      "skips": 0, "fallbacks": 0, "runtime": doc["runtime"], "assets": doc["assets"],
                      "sources": doc["sources"], "core_ast_sha256": source.core_ast_sha256,
                      "lock": doc["lock"], "gpu_diagnostics": doc["gpu_diagnostics"],
                      "claim_limit": CLAIM_LIMIT, "raw_pose_producers_reproduced": False,
                      "historical_continuous_posteriors_reproduced": False, "historical_backend_reproduced": False,
                      "generalizes_to_full_training": False, "full_training_census_complete": doc["scope"] == "full",
                      "elapsed_seconds": time.monotonic() - started, "resources": resources(device)}
            counts = {key: report[key] for key in ("frames", "pixel_channels", "posterior_elements", "tokens", "time_steps", "decoded_tokens")}
            write_new(output / "scientific.json", scientific_document(doc, seal_hash, completed, counts))
            report["scientific_report"] = file_record(output / "scientific.json")
            validate_run_report(report)
            write_new(output / "report.json", report)
            if kind == "repeat":
                comparison = compare_runs(bound_reference, file_record(output / "report.json"), mode="deterministic")
                write_new(output / "repeat_comparison.json", comparison)
            return {"status": "PASS", "output": str(output), "report_sha256": sha256(output / "report.json")}
    except BaseException as error:
        write_new(output / "failure.json", {"status": "FAIL", "stage": kind, "completed_clips": len(completed),
                  "active_clip": active_clip, "active_output": active_output,
                  "active_historical": active_historical, "active_diagnostics": active_diagnostics,
                  "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.monotonic() - started,
                  "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                  "raw_pose_producers_reproduced": False})
        raise


def require_nonoverlap(first, second):
    require(first["start"]["boot_id"] == second["start"]["boot_id"], "run clock/boot identity differs")
    for clock in ("unix", "monotonic"):
        require(first["finish"][f"finished_{clock}_ns"] < second["start"][f"started_{clock}_ns"],
                "same/overlapping run time; sequential independent executions required")


def compare_runs(first_record, second_record, mode="deterministic"):
    """Compare two pinned, distinct run records; deterministic is strictly exact."""
    require(mode == "deterministic", "only tensor/byte-exact deterministic repeat is admissible")
    first_path, second_path = canonical_record_path(first_record), canonical_record_path(second_record)
    require(first_path != second_path and record_inode(first_record) != record_inode(second_record),
            "same report/run cannot demonstrate a deterministic repeat")
    first, second = pinned_run_document(first_record), pinned_run_document(second_record)
    require(first["run_dir"] != second["run_dir"] and first["run_id"] != second["run_id"],
            "same canonical run directory or run ID")
    require(first["start"]["path"] != second["start"]["path"] and first["start"]["sha256"] != second["start"]["sha256"]
            and record_inode(first["start"]) != record_inode(second["start"]), "same start record/run alias")
    audited = [validate_run_report(report) for report in (first, second)]
    require(first["kind"] == first["scope"] and second["kind"] == "repeat", "need distinct initial and repeat executions")
    require_nonoverlap(*audited)
    require(audited[0]["artifact_paths"].isdisjoint(audited[1]["artifact_paths"]) and
            audited[0]["artifact_inodes"].isdisjoint(audited[1]["artifact_inodes"]), "overlapping/aliased posterior artifacts")
    for key in ("preregistration_seal_sha256", "scope", "selected_ids", "runtime", "assets", "sources", "core_ast_sha256"):
        require(first[key] == second[key], f"repeat {key} drift")
    require(first["preregistration_seal"] == second["preregistration_seal"], "repeat preregistration record drift")
    require(record_inode(first["scientific_report"]) != record_inode(second["scientific_report"]),
            "aliased/hardlinked scientific reports")
    require(first["scientific_report"]["sha256"] == second["scientific_report"]["sha256"] and
            audited[0]["scientific"] == audited[1]["scientific"], "deterministic scientific report mismatch")
    comparisons = []
    for left, right, registered in zip(first["clips"], second["clips"], audited[0]["preregistration"]["clips"]):
        a, a_spans = load_posterior(left["output"])
        b, b_spans = load_posterior(right["output"])
        require(all(left[key]["sha256"] == right[key]["sha256"] for key in ("output", "historical", "diagnostics")),
                "deterministic per-clip artifact hash drift")
        comparisons.append(compare_entries(b, a, registered["metadata"], audited[0]["preregistration"]["vocab_size"],
                                           b_spans, a_spans, mode=mode))
    return {"status": "PASS", "clips": len(comparisons), "elements": sum(c["elements"] for c in comparisons),
            "schema": SCHEMA, "mode": mode, "deterministic_repeat_verified": True,
            "validation_closure_verified": first["scope"] == "pilot",
            "all_item_semantic_census_verified": first["scope"] == "full",
            "claim": "two nonoverlapping byte-exact CPU executions with historical decision equivalence only",
            "claim_limit": CLAIM_LIMIT, "historical_continuous_posteriors_reproduced": False,
            "historical_backend_reproduced": False, "calibration_is_validation": False,
            "inputs": [first_record, second_record], "run_ids": [first["run_id"], second["run_id"]],
            "reconstructed_denominators": audited[0]["counts"],
            "exact_cpu_posterior_clips": sum(c["log_probs_exact"] for c in comparisons),
            "scientific_report_sha256": first["scientific_report"]["sha256"], "all_ctc_spans_exact": True,
            "max_abs_error": max(c["max_abs_error"] for c in comparisons)}


def synthetic_gate(output, junit, calibration_record=None):
    output = stage_dir(output, "synthetic_gate", create=False)
    require(owned(junit).is_relative_to(output), "unit-test evidence outside synthetic output")
    require(calibration_record is not None, "a derived and pinned calibration record is required")
    calibration = validate_calibration(calibration_record)
    tree = ET.parse(junit)
    require(list(tree.iter("testcase")) and not any(list(tree.iter(t)) for t in ("failure", "error", "skipped")),
            "unit-test evidence incomplete or failing")
    write_new(output / "start.json", {"stage": "synthetic-gate", "real_inference": False})
    configure_runtime("cpu")
    validate_revision_lineage()
    closure = source_closure()
    sources = closure["sources"]
    source = load_corrnet_source(sources, closure["model_assets"]["resnet_init"], "cpu")
    tested_runtime = runtime_fingerprint("cpu")
    # Tiny deterministic fake model exercises the exact imported inference core.
    torch = source.torch
    class FakeModel:
        def __call__(self, video, length):
            require(tuple(video.shape) == (1, 20, 3, 224, 224), "synthetic padding drift")
            logits = torch.tensor([[[0., 8., 0.]], [[0., 0., 8.]]], dtype=torch.float32)
            return {"sequence_logits": logits, "conv_logits": logits,
                    "feat_len": torch.tensor([2.0], dtype=torch.float32)}
    raw = output / "fixture" / "raw"
    cached = output / "fixture" / "cached"
    row = {"id": "synthetic", "length": 8, "gloss": "A B", "gloss_indices": [1, 2]}
    for i in range(1, 9):
        pixels = np.full((260, 210, 3), i * 7, dtype=np.uint8)
        for root, pixels_out in ((raw, pixels), (cached, cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_LANCZOS4))):
            ok, encoded = cv2.imencode(".png", pixels_out)
            require(ok, "synthetic PNG encoder failure")
            write_bytes_new(root / row["id"] / f"images{i:04d}.png", encoded.tobytes())
    inventory = frame_inventory(raw, cached, row)
    pixels = verify_pixels(inventory, raw, cached, row)
    video, count = source.load_and_preprocess_video(str(cached / row["id"] / "*.png"))
    require(count == 8 and tuple(video.shape) == (8, 3, 224, 224), "synthetic preprocessing count drift")
    entry = checked_inference(source, FakeModel(), {"A": [1], "B": [2]}, row, video)
    spans = checked_alignment(source, entry["log_probs"], entry["gloss_indices"])
    require(spans == [[0, 1], [1, 2]], "synthetic CTC mismatch")
    comparison = compare_entries(entry, entry, row, 3, spans, spans)
    report = {"schema": SCHEMA, "status": "PASS", "scope": "synthetic", "real_inference": False,
              "model_weight_loading_tested": False,
              "revision": REVISION, "tested_closure": closure, "tested_runtime": tested_runtime,
              "calibration": calibration_record, "contains_disclosed_prior_calibration": True,
              "sources": sources, "unit_tests": file_record(junit), "pixels": pixels,
              "comparison": comparison, "core_ast_sha256": source.core_ast_sha256,
              "claim_limit": CLAIM_LIMIT, "raw_pose_producers_reproduced": False,
              "historical_continuous_posteriors_reproduced": False, "historical_backend_reproduced": False,
              "generalizes_to_full_training": False, "full_training_census_complete": False, "resources": resources()}
    validate_claims(report)
    require(source.core_ast_sha256 == closure["core_ast_sha256"], "synthetic extracted inference-core digest mismatch")
    validate_source_closure(closure, check_model_assets=False)
    validate_runtime(tested_runtime, runtime_fingerprint("cpu"))
    write_new(output / "gate.json", report)
    return {"status": "PASS", "output": str(output), "gate_sha256": sha256(output / "gate.json"), "real_inference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--authorize-cache-read", action="store_true")
    gate = sub.add_parser("synthetic-gate")
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--junit", type=Path, required=True)
    gate.add_argument("--calibration", type=Path, required=True)
    gate.add_argument("--calibration-sha256", required=True)
    pre = sub.add_parser("preregister")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--gate", type=Path, required=True)
    pre.add_argument("--gate-sha256", required=True)
    pre.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    pre.add_argument("--device", default="cpu")
    pre.add_argument("--pilot-report", type=Path)
    pre.add_argument("--pilot-report-sha256")
    pre.add_argument("--pilot-repeat", type=Path)
    pre.add_argument("--pilot-repeat-sha256")
    pre.add_argument("--pilot-review", type=Path)
    pre.add_argument("--pilot-review-sha256")
    run = sub.add_parser("execute")
    run.add_argument("--prereg", type=Path, required=True)
    run.add_argument("--seal-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--kind", choices=("pilot", "full", "repeat"), required=True)
    run.add_argument("--authorize-inference", action="store_true")
    run.add_argument("--reference-report", type=Path)
    run.add_argument("--reference-report-sha256")
    comp = sub.add_parser("compare")
    comp.add_argument("--first", type=Path, required=True)
    comp.add_argument("--first-sha256", required=True)
    comp.add_argument("--second", type=Path, required=True)
    comp.add_argument("--second-sha256", required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--mode", choices=("deterministic",), default="deterministic")
    args = parser.parse_args()
    if args.command == "calibrate":
        result = derive_calibration(args.output, args.authorize_cache_read)
    elif args.command == "synthetic-gate":
        calibration = file_record(owned(args.calibration))
        require(calibration["sha256"] == args.calibration_sha256, "calibration artifact hash drift")
        result = synthetic_gate(args.output, args.junit, calibration)
    elif args.command == "preregister":
        result = preregister(args.output, args.gate, args.gate_sha256, args.scope, args.device,
                             args.pilot_report, args.pilot_report_sha256, args.pilot_repeat, args.pilot_repeat_sha256,
                             args.pilot_review, args.pilot_review_sha256)
    elif args.command == "execute":
        result = execute(args.prereg, args.seal_sha256, args.output, args.kind, args.authorize_inference,
                         args.reference_report, args.reference_report_sha256)
    else:
        records = [file_record(owned(args.first)), file_record(owned(args.second))]
        require([r["sha256"] for r in records] == [args.first_sha256, args.second_sha256], "comparison report hash drift")
        result = compare_runs(*records, mode=args.mode)
        directory = stage_dir(args.output, "compare")
        write_new(directory / "report.json", dict(result, inputs=records))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
