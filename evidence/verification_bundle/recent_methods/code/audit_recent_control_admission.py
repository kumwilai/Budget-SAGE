"""Sealed, read-only admission audit of historical CSL controls and one PT bank.

This does not train, regenerate a corpus, or run a recognizer. The two bounded
CPU probes execute definitions from the hash-bound historical implementations.
An ADMITTED PT row is a 641-item release-bank re-score evaluated with declared
--fps 12. It does not authenticate producer execution or native-rate provenance.

Outputs are exclusively created beneath recent_control_admission_v1. A caller
must supply the printed design hash to run or verify. Hashes provide consistency
under a trusted design digest, not authentication against an owner who replaces
both the design and its trusted digest. Pickle inputs are trusted local research
artifacts, checked against the design before deserialization.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import resource
import secrets
import stat
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

# Must precede numerical imports, including imports executed by a probe.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/revision/nonhuman_closure_20260907/recent_control_admission_v1"
SCHEMA = "recent_control_admission_v3"
EXECUTION_SCHEMA = "recent_control_execution_identity_v1"
RUN_FILES = ("start.json", "scientific.json", "telemetry.json", "output_seal.json")
COUNTS = {"train": 18400, "dev": 1077, "test": 1176}
TAGS = ("pt", "signidd", "soke", "g2p_ddm")
CK_NAMES = {"pt": "pt", "signidd": "diff", "soke": "soke", "g2p_ddm": "g2p_ddm"}
PROBE = {"seed": 0, "short_frames": 8, "long_frames": 16,
         "short_codes": 4, "long_codes": 8, "tolerance": 1e-5,
         "device": "cpu", "threads": 1,
         "gloss_policy": "lowest two positive checkpoint vocabulary IDs",
         "vq_code_policy": "arange(short_codes), right-pad with code 0"}
CB = "scripts/csl_baselines.py"
VQ = "scripts/csl_baselines_vq.py"
PT = "src/baselines/progressive_transformer.py"
MSKA = "external/baselines/MSKA/train.py"
EVAL = "external/SLRTP-Sign-Production-Evaluation"
V4 = "outputs/revision/astra_open_closure_20260906/csl_text_transfer_v4"
OFFICIAL = "data/csl-daily/CSL-Daily/sentence_label"
MAX_RSS_KIB = 7 * 1024 * 1024
MIN_AVAILABLE_KIB = 11 * 1024 * 1024


class AuditError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise AuditError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def confined(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    require(path != base and path.is_relative_to(base), f"path outside confined root: {path}")
    return path


def source(path):
    path = Path(path)
    return confined(path if path.is_absolute() else ROOT / path, ROOT)


def owned(path):
    return confined(path, OUT)


def no_alias_path(path):
    """Check the supplied spelling before resolve can conceal a symlink alias.

    Output paths deliberately reject '..', symlink components and proc magic
    links. Canonical-path and (st_dev, st_ino) checks additionally catch aliases
    such as bind mounts where the filesystem exposes the same underlying inode.
    """
    raw = Path(path)
    require(".." not in raw.parts, "parent traversal is not a canonical output path")
    absolute = raw if raw.is_absolute() else Path.cwd()/raw
    require(not (len(absolute.parts) > 1 and absolute.parts[1] == "proc"),
            "proc file-descriptor or magic-link output alias is forbidden")
    for component in (absolute, *absolute.parents):
        require(not component.is_symlink(), "symlink output path alias is forbidden")
    canonical_path = absolute.resolve()
    require(canonical_path == absolute, "noncanonical output path alias is forbidden")
    return canonical_path


def inode_identity(path, *, directory=False):
    path = no_alias_path(path)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            "execution artifact has the wrong filesystem type")
    if not directory:
        require(info.st_nlink == 1, "hardlinked execution artifact is forbidden")
    return {"device": int(info.st_dev), "inode": int(info.st_ino)}


def run_directory(path):
    path = owned(no_alias_path(path))
    return path, {"canonical_path": str(path), **inode_identity(path, directory=True)}


def run_artifact_identities(directory):
    result = {name: inode_identity(directory/name) for name in RUN_FILES}
    require(len({(x["device"], x["inode"]) for x in result.values()}) == len(result),
            "execution artifacts alias the same inode")
    return result


def compare_preflight(output, repeat):
    """Reject aliases using path/stat metadata only, before reading any file."""
    require(os.fspath(output) != os.fspath(repeat), "output and repeat arguments are identical")
    a, ai = run_directory(output)
    b, bi = run_directory(repeat)
    require(a != b, "output and repeat resolve to the same canonical directory")
    require((ai["device"], ai["inode"]) != (bi["device"], bi["inode"]),
            "output and repeat alias the same directory inode")
    artifacts_a, artifacts_b = run_artifact_identities(a), run_artifact_identities(b)
    inodes_a = {(x["device"], x["inode"]) for x in artifacts_a.values()}
    inodes_b = {(x["device"], x["inode"]) for x in artifacts_b.values()}
    require(inodes_a.isdisjoint(inodes_b), "output and repeat share execution-artifact inodes")
    return a, b


def new_dir(path):
    path = owned(no_alias_path(path))
    require(not path.exists(), "immutable output exists; use a fresh attempt directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_new(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))


def write_output_seal(path, value):
    """Reserve the seal inode exclusively, then bind that inode in its payload."""
    path = owned(no_alias_path(path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "invalid new seal inode")
        value = {**value, "seal_inode": {"device": int(info.st_dev), "inode": int(info.st_ino)}}
        handle.write(canonical(value))


def validate_execution_identity(execution, directory):
    require(isinstance(execution, dict) and set(execution) ==
            {"schema", "run_id", "start_unix_ns", "output_directory"}, "malformed execution identity")
    require(execution["schema"] == EXECUTION_SCHEMA, "execution identity schema differs")
    run_id = execution["run_id"]
    require(isinstance(run_id, str) and len(run_id) == 64 and all(c in "0123456789abcdef" for c in run_id),
            "invalid execution run ID")
    require(type(execution["start_unix_ns"]) is int and execution["start_unix_ns"] > 0,
            "invalid execution start time")
    _, current_directory = run_directory(directory)
    require(execution["output_directory"] == current_directory,
            "execution output path/directory inode binding differs")


def record(path):
    path = source(path)
    require(path.is_file(), f"required input missing: {path}")
    return {"path": str(path.relative_to(ROOT.resolve())), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def verify_record(rec):
    require(isinstance(rec, dict) and set(rec) == {"path", "bytes", "sha256"},
            "malformed file record")
    current = record(rec["path"])
    require(current == rec, f"input hash/size changed: {rec['path']}")
    return source(rec["path"])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_version(package):
    if package == "python":
        return platform.python_version()
    if package == "torch":
        import torch
        return str(torch.__version__)
    return importlib.metadata.version(package)


def verify_runtime_version(package, expected):
    current = runtime_version(package)
    require(current == expected, f"external metric runtime differs from stored run: {package}: {current} != {expected}")
    return {"recorded_runtime": expected, "current_runtime": current,
            "distribution": None if package == "python" else importlib.metadata.version(package)}


def environment():
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "jiwer", "sacrebleu", "fastdtw")},
            "torch_runtime_build": runtime_version("torch"),
            "device": "cpu", "numerical_threads": 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not_set")}


def git(args, directory):
    result = subprocess.run(["git", "-C", str(source(directory)), *args],
                            check=True, text=True, capture_output=True)
    return result.stdout.rstrip("\n")


def evidence(path, start, end):
    path = source(path)
    lines = path.read_bytes().splitlines(keepends=True)
    require(1 <= start <= end <= len(lines), "invalid source evidence range")
    return {**record(path), "start_line": start, "end_line": end,
            "lines_sha256": hashlib.sha256(b"".join(lines[start-1:end])).hexdigest()}


def tree(path):
    return ast.parse(source(path).read_text(encoding="utf-8"))


def node(path, name):
    parts = name.split(".")
    current = tree(path)
    for part in parts:
        found = [n for n in current.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                         ast.ClassDef)) and n.name == part]
        require(len(found) == 1, f"required source symbol missing/ambiguous: {path}:{name}")
        current = found[0]
    return current


def symbol(path, name):
    n = node(path, name)
    return {**evidence(path, n.lineno, n.end_lineno), "symbol": name,
            "ast_sha256": hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()}


def matching_lines(path, needle):
    lines = source(path).read_text(encoding="utf-8").splitlines()
    found = [evidence(path, i, i) for i, line in enumerate(lines, 1) if needle in line]
    require(found, f"source marker missing: {path}:{needle}")
    return found


def call_name(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return call_name(n.value) + "." + n.attr
    return "<dynamic>"


def calls(n):
    return sorted({call_name(x.func) for x in ast.walk(n) if isinstance(x, ast.Call)})


def static_audit():
    infer = node(CB, "generate_diff")
    ddim = node(CB, "Diffusion.ddim")
    main = node(CB, "main")
    seed_calls = [x for x in calls(infer) + calls(ddim) + calls(main)
                  if x.endswith(("manual_seed", ".seed", "seed_all"))]
    parser_seeds = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == "--seed"]
    generation_parser_seed = [n for n in parser_seeds if call_name(n.func).startswith("gn.")]
    random_draws = [evidence(CB, n.lineno, n.end_lineno) for n in ast.walk(ddim)
                    if isinstance(n, ast.Call) and call_name(n.func) == "torch.randn"
                    and not any(k.arg == "generator" for k in n.keywords)]
    evaluation = node(MSKA, "evaluate")
    selections = []
    for loop in ast.walk(evaluation):
        if not isinstance(loop, ast.For) or ast.unparse(loop.target) != "hyp_name":
            continue
        for n in ast.walk(loop):
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and call_name(n.value.func) == "min"
                    and any(ast.unparse(t) == "evaluation_results['wer']" for t in n.targets)
                    and "wer_results['wer']" in ast.unparse(n.value)):
                selections.append(evidence(MSKA, n.lineno, n.end_lineno))
    require(random_draws and not seed_calls and not generation_parser_seed,
            "historical unseeded-diffusion static finding no longer reproduces")
    require(selections, "historical per-result minimum-head selection no longer reproduces")
    return {"unseeded_continuous_diffusion": {
                "proved_in_bound_local_path": True, "inference_seed_calls": seed_calls,
                "generation_parser_has_seed": bool(generation_parser_seed),
                "random_draw_without_generator": random_draws,
                "scope": [symbol(CB, s) for s in ("generate_diff", "Diffusion.ddim", "main")],
                "limit": "Static call path, not reconstruction of the historical process RNG state."},
            "per_result_best_head": {"proved_in_bound_source": True,
                "selection": selections, "scope": symbol(MSKA, "evaluate"),
                "caller": symbol("scripts/score_csl_mska_fast.py", "score_one"),
                "meaning": "Minimum corpus WER across recognizer heads recomputed for each evaluated bank/split; not per-item oracle selection."}}


def method_fidelity():
    specs = {
        "pt": {
            "missing": ["official joint-plus-continuous-counter regression/input; local endpoint BCE is different"],
            "local": [(PT, "stop_targets_from_lengths"), (PT, "progressive_transformer_loss"),
                      (PT, "PoseDecoder"), (CB, "generate_pt")],
            "official": [("external/baselines/ProgressiveTransformersSLP/loss.py", "RegLoss")],
            "markers": [("external/baselines/ProgressiveTransformersSLP/model.py", "plus one for counter")],
            "absence_tokens": []},
        "signidd": {
            "missing": ["iconicity decomposition ID", "attribute-controlled ACD denoiser and attribute attention"],
            "local": [(CB, "PoseDenoiser"), (CB, "Diffusion"), (CB, "generate_diff")],
            "official": [("external/baselines/Sign-IDD/ID.py", "ID"),
                         ("external/baselines/Sign-IDD/ACD.py", "ACD.model_predictions"),
                         ("external/baselines/Sign-IDD/ACD_Denoiser.py", "ACD_Denoiser")],
            "markers": [], "absence_tokens": ["ID", "ACD", "ACD_Denoiser", "layers_mha_ac"]},
        "soke": {
            "missing": ["pretrained multilingual language model", "separate body/hand tokenizers and multi-head decoding", "retrieval-enhanced conditioning"],
            "local": [(VQ, "PoseVQVAE"), (VQ, "CodeGenerator"), (VQ, "train_gen"), (VQ, "generate")],
            "official": [("external/baselines/SOKE/mGPT/archs/mgpt_mbart.py", "correct_lang_token")],
            "markers": [("external/baselines/SOKE/README.md", "retrieval-enhanced"),
                        ("external/baselines/SOKE/configs/lm/mbart_h2s_csl_phoenix.yaml", "model_path")],
            "absence_tokens": ["Mbart", "MBart", "mbart", "retrieval", "lhand", "rhand"]},
        "g2p_ddm": {
            "missing": ["categorical diffusion posterior/sampling", "CodeUnet denoiser; local confidence-ordered argmax unmasking is different"],
            "local": [(VQ, "CodeGenerator"), (VQ, "train_gen"), (VQ, "generate")],
            "official": [("external/baselines/G2P-DDM/modules/vq_codeunet.py", "CodeUnet"),
                         ("external/baselines/G2P-DDM/stage2_models/vq_diffusion_codeunet.py", "Point2textModelStage2.q_posterior"),
                         ("external/baselines/G2P-DDM/stage2_models/vq_diffusion_codeunet.py", "Point2textModelStage2.log_sample_categorical")],
            "markers": [], "absence_tokens": ["CodeUnet", "q_posterior", "log_sample_categorical", "multinomial_kl"]}}
    result = {}
    for tag, spec in specs.items():
        # The negative search is bounded to the actual local production symbols.
        # Paired positive official symbols explain the architectural mismatch.
        names = set()
        for path, name in spec["local"]:
            for n in ast.walk(node(path, name)):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
        absent = {token: not any(token in n for n in names) for token in spec["absence_tokens"]}
        require(all(absent.values()), f"method contrast needs re-review: {tag}")
        result[tag] = {"named_method_fidelity": False, "defining_mechanisms_missing_or_changed": spec["missing"],
                       "local_evidence": [symbol(*s) for s in spec["local"]],
                       "official_evidence": [symbol(*s) for s in spec["official"]],
                       "official_markers": [v for p, n in spec["markers"] for v in matching_lines(p, n)],
                       "identifier_absence_in_local_scope": absent,
                       "ceiling": "Historical study-specific architectural control; not a reproduction of the named method."}
    return result


def manifest_rows(rows, expected=None):
    require(isinstance(rows, list) and rows, "manifest is empty or not a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "manifest row is not a mapping")
        sid = row.get("id")
        require(isinstance(sid, str) and sid and sid not in result, "invalid/duplicate manifest ID")
        require(isinstance(row.get("gloss"), str) and row["gloss"].strip(), "invalid manifest gloss")
        result[sid] = row
    if expected is not None:
        require(len(result) == expected, f"manifest count {len(result)} != {expected}")
    return result


def discard_tensor_storage(serialized_bytes):
    require(isinstance(serialized_bytes, bytes), "unexpected tensor storage encoding")
    return None


def tensor_metadata(storage, storage_offset, size, stride, *unused):
    require(storage is None and isinstance(size, tuple) and isinstance(stride, tuple),
            "unexpected tensor rebuild metadata")
    return {"tensor_shape": list(size), "tensor_stride": list(stride), "storage_materialized": False}


class MetadataUnpickler(pickle.Unpickler):
    """Read IDs/order/labels without allocating the 3.3 GB archive's tensors.

    Pickled storage bytes may reside in the unpickler memo until load completes;
    reconstructed tensors do not. Unknown globals fail closed. This deliberately
    does not assert training-pose numeric validity; output banks are scanned in
    full by bank_summary. The whole input is already hash-bound before loading.
    """
    def find_class(self, module, name):
        from collections import OrderedDict
        permitted = {("torch.storage", "_load_from_bytes"): discard_tensor_storage,
                     ("torch._utils", "_rebuild_tensor_v2"): tensor_metadata,
                     ("torch._utils", "_rebuild_tensor"): tensor_metadata,
                     ("collections", "OrderedDict"): OrderedDict}
        require((module, name) in permitted, f"unapproved metadata-pickle global: {module}.{name}")
        return permitted[module, name]


def load_archive_metadata(path):
    with source(path).open("rb") as handle:
        return MetadataUnpickler(handle).load()


def official_manifests():
    with source(f"{OFFICIAL}/split_1.txt").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    ids = [x["name"] for x in rows]
    require(len(ids) == len(set(ids)), "duplicate official split ID")
    require(all(x["split"] in COUNTS for x in rows), "unknown official split")
    with source(f"{OFFICIAL}/csl2020ct_v2.pkl").open("rb") as handle:
        annotations = pickle.load(handle)["info"]
    annotation = {x["name"]: x for x in annotations}
    require(len(annotation) == len(annotations), "duplicate official annotation ID")
    manifests, summary = {}, {}
    for split, count in COUNTS.items():
        declared = {x["name"] for x in rows if x["split"] == split}
        admitted = declared & set(annotation)
        manifests[split] = manifest_rows(read_json(source(f"data/csl-daily/csl_daily_{split}.json")), count)
        require(set(manifests[split]) == admitted, f"{split} manifest differs from official split/annotation intersection")
        for sid, row in manifests[split].items():
            require(row["gloss"].split() == annotation[sid]["label_gloss"], f"official gloss mismatch: {sid}")
        summary[split] = {"declared_split_count": len(declared), "admitted_count": len(admitted),
                          "unannotated_split_ids": sorted(declared - set(annotation)),
                          "admitted_ids_sha256": value_hash(sorted(admitted))}
    require(not any(set(manifests[a]) & set(manifests[b]) for a, b in
                    (("train", "dev"), ("train", "test"), ("dev", "test"))), "split overlap")
    return manifests, summary


def bank_summary(bank, expected, lengths, joints=133):
    import numpy as np
    import torch
    require(isinstance(bank, dict), "bank is not a mapping")
    missing, extra = sorted(set(expected) - set(bank)), sorted(set(bank) - set(expected))
    invalid = {k: [] for k in ("shape", "nonfinite", "gloss", "name", "num_frames", "reference_length")}
    conf_n = below = above = conf_nonfinite = coord_nonfinite = 0
    low, high = math.inf, -math.inf
    frame_lengths = []
    for sid in sorted(set(bank) & set(expected)):
        row = bank[sid]
        if not isinstance(row, dict):
            invalid["shape"].append(sid)
            continue
        p = row.get("keypoint")
        if not isinstance(p, (np.ndarray, torch.Tensor)):
            invalid["shape"].append(sid)
            continue
        a = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        if a.ndim != 3 or a.shape[0] <= 0 or a.shape[1:] != (joints, 3) or a.dtype.kind not in "fiu":
            invalid["shape"].append(sid)
            continue
        finite = np.isfinite(a)
        if not finite.all():
            invalid["nonfinite"].append(sid)
        coord_nonfinite += int((~finite).sum())
        c = a[:, :, 2]
        cf = np.isfinite(c)
        conf_n += c.size
        conf_nonfinite += int((~cf).sum())
        below += int((c[cf] < 0).sum())
        above += int((c[cf] > 1).sum())
        if cf.any():
            low, high = min(low, float(c[cf].min())), max(high, float(c[cf].max()))
        t = int(a.shape[0]); frame_lengths.append(t)
        if row.get("name") != sid:
            invalid["name"].append(sid)
        if not isinstance(row.get("gloss"), str) or row["gloss"].split() != expected[sid]["gloss"].split():
            invalid["gloss"].append(sid)
        if type(row.get("num_frames")) is not int or row["num_frames"] != t:
            invalid["num_frames"].append(sid)
        if sid not in lengths or lengths[sid] != t:
            invalid["reference_length"].append(sid)
    clean_structure = not missing and not extra and not any(invalid.values())
    return {"n": len(bank), "expected_n": len(expected), "missing_ids": missing, "extra_ids": extra,
            "id_set_sha256": value_hash(sorted(bank)), "invalid_ids": invalid,
            "structure_gloss_and_length_valid": clean_structure, "nonfinite_elements": coord_nonfinite,
            "frames": {"total": sum(frame_lengths), "min": min(frame_lengths, default=None),
                       "max": max(frame_lengths, default=None)},
            "confidence": {"elements": conf_n, "below_zero": below, "above_one": above,
                           "outside_unit_interval": below + above, "nonfinite": conf_nonfinite,
                           "outside_fraction": (below + above)/conf_n if conf_n else None,
                           "finite_range": [low, high] if low != math.inf else None,
                           "valid_probability_channel": conf_n > 0 and below + above + conf_nonfinite == 0}}


def checkpoint_summary(ck, log, metric):
    import torch
    require(isinstance(ck, dict) and isinstance(ck.get("model"), dict), "checkpoint missing model state")
    require(isinstance(log, list) and log, "empty training log")
    require([x.get("epoch") for x in log] == list(range(1, len(log)+1)), "training epoch sequence invalid")
    require(all(all(isinstance(v, (int, float)) and math.isfinite(v) for v in x.values()) for x in log),
            "nonfinite/non-numeric training record")
    state = ck["model"]
    tensor_rows = []
    for name, t in sorted(state.items()):
        require(isinstance(t, torch.Tensor), f"non-tensor checkpoint state: {name}")
        tensor_rows.append({"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
                            "finite": bool(torch.isfinite(t).all()), "elements": t.numel()})
    meta = {k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in ck.items() if k not in ("model", "vocab")}
    require(all(r["finite"] for r in tensor_rows), "checkpoint has nonfinite weights")
    if "vocab" in ck:
        vocab = ck["vocab"]
        require(isinstance(vocab, dict) and vocab.get("<pad>") == 0
                and set(vocab.values()) == set(range(len(vocab))), "checkpoint vocabulary malformed")
    for key in ("mean", "std"):
        if key in ck:
            value = torch.as_tensor(ck[key])
            require(value.shape == (3,) and bool(torch.isfinite(value).all()), "invalid normalization metadata")
            if key == "std":
                require(bool((value > 0).all()), "nonpositive normalization scale")
    required_meta = ("seed", "epoch", "training_ids_sha256", "code_sha256", "command", "environment")
    best = min(log, key=lambda r: r[metric])
    return {"stored_keys": sorted(ck), "stored_metadata": meta,
            "missing_frozen_training_metadata": [k for k in required_meta if k not in ck],
            "frozen_historical_training_provenance_complete": all(k in ck for k in required_meta),
            "tensor_schema_sha256": value_hash(tensor_rows), "state_tensors": len(state),
            "state_elements": sum(r["elements"] for r in tensor_rows), "all_weights_finite": True,
            "vocabulary_size": len(ck.get("vocab", {})), "vocabulary_sha256": value_hash(ck.get("vocab", {})),
            "training_log": {"epochs": len(log), "recorded_epoch_wall_seconds": math.fsum(x["wall"] for x in log),
                             "selection_metric": metric, "best_recorded_epoch": best["epoch"],
                             "best_recorded_value": best[metric],
                             "checkpoint_epoch_binding": "absent unless epoch is stored; best epoch above is inferred from log only"}}


def load_definitions(path, names, extra=None):
    """Execute exact AST definition nodes, avoiding the legacy import-time chdir.

    Function/class bodies and decorators are unchanged. Only pure dependency
    bindings are supplied explicitly. The sealed file hash and AST hashes bind
    the definition bytes used by these tiny probes.
    """
    import dataclasses
    import typing
    import numpy as np
    import torch
    module = types.ModuleType("_recent_audit_" + value_hash([str(path), names])[:16])
    sys.modules[module.__name__] = module
    module.__dict__.update({"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                           "np": np, "dataclass": dataclasses.dataclass, "MANUAL_DIM": 201,
                           "POSE_DIM": 399, "DOWN": 4,
                           **{k: getattr(typing, k) for k in ("Dict", "List", "Tuple", "Iterable", "Optional")},
                           **(extra or {})})
    selected = [node(path, name) for name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source(path)), "exec"), module.__dict__)
    return module


def difference(a, b, tolerance):
    import torch
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    require(a.shape == b.shape and a.numel() > 0, "invariance comparison shape mismatch")
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonfinite invariance probe")
    delta = (a.double() - b.double()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "elements_above_tolerance": int((delta > tolerance).sum()),
            "tolerance": tolerance, "invariant": bool(delta.max() <= tolerance)}


def pt_probe(model, tokens, mask, resample, policy=PROBE):
    import torch
    short, long, tol = policy["short_frames"], policy["long_frames"], policy["tolerance"]
    require(0 < short < long, "probe horizons must be positive and increasing")
    with torch.inference_mode():
        single, _ = model.generate(tokens[:1], mask[:1], max_steps=short, stop_threshold=2.0)
        horizon, _ = model.generate(tokens[:1], mask[:1], max_steps=long, stop_threshold=2.0)
        mixed, _ = model.generate(tokens, mask, max_steps=long, stop_threshold=2.0)
        reverse, _ = model.generate(tokens.flip(0), mask.flip(0), max_steps=long, stop_threshold=2.0)
    export = lambda x: torch.as_tensor(resample(x.cpu().numpy(), short))
    return {"units": "checkpoint normalized (x,y,confidence)",
            "tokens": tokens.tolist(), "mask": mask.tolist(),
            "short_frames": short, "long_frames": long,
            "prefix_at_longer_horizon": difference(single[0], horizon[0, :short], tol),
            "same_horizon_batch_composition": difference(horizon[0], mixed[0], tol),
            "same_horizon_reversed_order": difference(mixed[0], reverse[-1], tol),
            "historical_export_horizon": difference(export(single[0]), export(horizon[0]), tol),
            "historical_export_mixed_batch": difference(export(single[0]), export(mixed[0]), tol),
            "mechanism": "Local export resamples the complete batch-max trajectory, not the per-item prefix."}


def vq_probe(vq, policy=PROBE):
    import torch
    short, long, tol = policy["short_codes"], policy["long_codes"], policy["tolerance"]
    require(0 < short < long, "probe code horizons must be positive and increasing")
    codes = torch.arange(short, dtype=torch.long) % vq.vq.K
    padded = torch.cat([codes, torch.zeros(long-short, dtype=torch.long)])
    other = (torch.arange(long, dtype=torch.long) + short) % vq.vq.K
    batch = torch.stack([padded, other])
    with torch.inference_mode():
        decode = lambda c: vq.decode(vq.vq.codes_to_vec(c))
        single = decode(codes[None])[0]
        horizon = decode(padded[None])[0, :single.shape[0]]
        mixed = decode(batch)[:, :single.shape[0]]
        reverse = decode(batch.flip(0))[:, :single.shape[0]]
    return {"units": "checkpoint normalized (x,y,confidence)", "short_codes": codes.tolist(),
            "extended_codes": padded.tolist(), "other_codes": other.tolist(),
            "prefix_with_code_zero_padding": difference(single, horizon, tol),
            "mixed_batch_prefix": difference(single, mixed[0], tol),
            "same_padded_horizon_batch_composition": difference(horizon, mixed[0], tol),
            "same_padded_horizon_reversed_order": difference(mixed[0], reverse[-1], tol),
            "mechanism": "Convolutional VQ decoder sees future padded codes before per-item truncation; boundary outputs change."}


def real_probes():
    import torch
    torch.manual_seed(PROBE["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pt = load_definitions(PT, ["_causal_mask", "_resample_numpy", "ProgressiveTransformerConfig",
                              "GlossEncoder", "PoseDecoder", "Counter", "ProgressiveTransformer"])
    ck = torch.load(source("outputs/csl_baselines/pt/best.pt"), map_location="cpu", weights_only=True)
    cfg = pt.ProgressiveTransformerConfig(**ck["cfg"])
    model = pt.ProgressiveTransformer(cfg).eval()
    model.load_state_dict(ck["model"], strict=True)
    ids = sorted(x for x in ck["vocab"].values() if x > 0)[:2]
    require(len(ids) == 2, "probe requires two non-padding gloss IDs")
    tokens = torch.tensor([ids, ids[::-1]])
    result = {"policy": PROBE, "pt": pt_probe(model, tokens, torch.ones_like(tokens, dtype=torch.bool), pt._resample_numpy)}
    del model, ck; gc.collect()
    vm = load_definitions(VQ, ["VectorQuantizer", "PoseVQVAE"])
    ck = torch.load(source("outputs/csl_baselines/vqvae/best.pt"), map_location="cpu", weights_only=True)
    vq = vm.PoseVQVAE(ck["K"], ck["d_model"]).eval()
    vq.load_state_dict(ck["model"], strict=True)
    result["vq"] = vq_probe(vq)
    require(not result["pt"]["historical_export_horizon"]["invariant"], "PT horizon counterexample did not reproduce")
    require(not result["vq"]["prefix_with_code_zero_padding"]["invariant"], "VQ padding counterexample did not reproduce")
    return result


def verify_legacy_file(rec):
    # Historical records already use the same fields but may be absolute paths.
    current = record(rec["path"])
    require(current["sha256"] == rec["sha256"] and current["bytes"] == rec["bytes"],
            f"legacy provenance file changed: {rec['path']}")
    return current


def external_release_audit(provenance):
    import copy
    import jiwer
    import numpy as np
    import torch
    required = {"gt_identity_test_fps25", "gt_identity_dev_fps25", "pt_organizer_release_test_fps12",
                "pt_organizer_release_dev_fps12", "pt_organizer_release_test_fps25_WRONG_negative_control",
                "pt_demo30_dev_fps12"}
    runs = {x["tag"]: x for x in provenance["runs"]}
    require(len(runs) == len(provenance["runs"]) and required <= set(runs), "external provenance missing/duplicate required run")
    ev = provenance["evaluator"]
    require(git(["rev-parse", "HEAD"], EVAL) == ev["git_head"], "external evaluator HEAD differs")
    require(git(["diff", "--no-ext-diff", "HEAD", "--"], EVAL).rstrip() == ev["local_diff"].rstrip(),
            "external evaluator source patch differs from scored provenance")
    require(git(["remote", "get-url", "origin"], EVAL) == "https://github.com/walsharry/SLRTP-Sign-Production-Evaluation.git",
            "external evaluator origin differs")
    require(set(ev["bt_model_files"]) == {"best.ckpt", "config.yaml", "gls.vocab", "txt.vocab", "validations.txt"},
            "incomplete frozen external evaluator bundle")
    for rec in ev["bt_model_files"].values():
        verify_legacy_file(rec)
    runtime_checks = {package: verify_runtime_version(package, version) for package, version in ev["libs"].items()}
    metric = load_definitions(f"{EVAL}/metrics.py", ["wer"], {"copy": copy, "jiwer": jiwer})
    summary = {}
    for tag in sorted(required):
        run = runs[tag]
        for field in ("pred_file", "gt_file"):
            verify_legacy_file(run[field])
        actual = read_json(source(run["results_json"]))
        require(actual == run["metrics"], f"external result/provenance metric mismatch: {tag}")
        require(source(run["log"]).is_file(), f"missing external execution log: {tag}")
        require(run["fps_flag"] == (12 if "fps12" in tag else 25), f"external FPS mismatch: {tag}")
        import shlex
        command = shlex.split(run["command"])
        require(command[command.index("--fps")+1] == str(run["fps_flag"]), "command FPS differs")
        require(command[command.index("--tag")+1] == tag, "command tag differs")
        require(source(command[1]) == source(ev["path"]) and source(command[2]) == source(run["pred_file"]["path"])
                and source(command[3]) == source(run["gt_file"]["path"]), "external command input differs")
        require(source(command[4]) == source(ev["bt_model_files"]["best.ckpt"]["path"]).parent,
                "external command checkpoint directory differs")
        pred_path = source(run["results_json"]).with_name(tag+"_text_preds.pt")
        hyps = torch.load(pred_path, map_location="cpu", weights_only=True)
        gt = torch.load(source(run["gt_file"]["path"]), map_location="cpu", weights_only=True)
        require(isinstance(hyps, list) and len(hyps) == len(gt) == run["n_clips"]
                and all(isinstance(h, str) for h in hyps), "external saved text prediction mapping invalid")
        measured_wer = metric.wer(hyps, [r["text"] for r in gt.values()])
        require(abs(measured_wer-actual["wer"]) < 1e-10, "stored text predictions do not reproduce external WER")
        del gt; gc.collect()
        summary[tag] = {"n": run["n_clips"], "fps": run["fps_flag"], "metrics": actual,
                        "results": record(run["results_json"]), "log": record(run["log"]),
                        "stored_text_predictions": record(pred_path), "recomputed_text_wer": measured_wer}
    coverage = {}
    for split, expected_n in (("dev", 515), ("test", 641)):
        run = runs[f"pt_organizer_release_{split}_fps12"]
        bank = torch.load(source(run["pred_file"]["path"]), map_location="cpu", weights_only=False)
        gt = torch.load(source(run["gt_file"]["path"]), map_location="cpu", weights_only=False)
        require(isinstance(bank, dict) and isinstance(gt, dict) and set(bank) == set(gt) and len(bank) == expected_n,
                f"external release exact ID coverage fails: {split}")
        req, pred_t, gt_t = [], [], []
        for sid in sorted(bank):
            p, g = torch.as_tensor(bank[sid]), torch.as_tensor(gt[sid]["poses_3d"])
            require(p.ndim == g.ndim == 3 and p.shape[1:] == g.shape[1:] == (178, 3)
                    and p.shape[0] > 0 and g.shape[0] > 0, "external native pose shape invalid")
            require(bool(torch.isfinite(p).all() and torch.isfinite(g).all()), "external nonfinite pose")
            require(all(isinstance(gt[sid].get(k), str) and gt[sid][k].strip() for k in ("text", "gloss")),
                    "external request mapping incomplete")
            req.append({"id": sid, "text": gt[sid]["text"], "gloss": gt[sid]["gloss"]})
            pred_t.append(int(p.shape[0])); gt_t.append(int(g.shape[0]))
        avg_duration = float(np.mean(np.asarray(pred_t) / np.ceil(np.asarray(gt_t)/2)))
        require(abs(avg_duration-run["metrics"]["avg_duration"]) < 1e-12, "external duration fingerprint differs")
        require(run["n_clips"] == expected_n, "external recorded sample count differs")
        coverage[split] = {"n": expected_n, "id_set_sha256": value_hash(sorted(bank)),
                           "requests_sha256": value_hash(req), "shape_tail": [178, 3],
                           "all_finite": True, "pred_frames": sum(pred_t), "gt_frames": sum(gt_t),
                           "measured_avg_duration_under_declared_fps12": avg_duration,
                           "conversion": "identity; native 178-joint xyz; third coordinate is depth, not confidence"}
        del bank, gt; gc.collect()
        anchor = runs[f"gt_identity_{split}_fps25"]
        expected = provenance["organizer_reference_values"][f"gt_{split}"]
        for name in ("wer", "dtw_mje"):
            require(anchor["metrics"][name] == expected[name], "external ground-truth anchor differs")
        for name in ("bleu1", "bleu4"):
            require(anchor["metrics"]["bleu"][name] == expected[name], "external BLEU anchor differs")
        require(anchor["pred_file"] == anchor["gt_file"], "identity anchor uses different inputs")
    bad = runs["pt_organizer_release_test_fps25_WRONG_negative_control"]
    good = runs["pt_organizer_release_test_fps12"]
    require(bad["pred_file"] == good["pred_file"] and bad["gt_file"] == good["gt_file"], "FPS negative-control inputs differ")
    require(bad["metrics"]["avg_duration"] < .6 and good["metrics"]["avg_duration"] > .9,
            "external FPS negative control does not distinguish duration")
    return {"complete": True, "coverage": coverage, "stored_runs": summary,
            "evaluator_git_head": ev["git_head"], "historical_environment": ev["libs"],
            "runtime_version_checks": runtime_checks,
            "official_release_listing": matching_lines(f"{EVAL}/README.md", "PT_baseline_test.pt"),
            "official_fps_example": matching_lines(f"{EVAL}/README.md", "--tag demo --fps 12"),
            "official_url": "https://github.com/walsharry/SLRTP-Sign-Production-Evaluation",
            "limits": ["Stored scores and execution logs verified; no recognizer rerun in this audit.",
                       "Indexed organizer output-bank evaluation, not a newly trained named-method reproduction.",
                       "Historical producer checkpoint/seed/input execution traces were not identified or accessible in the inspected official sources; the indexed release-bank admission route applies.",
                       "This audit admits the 641-item test release-bank re-score evaluated with declared --fps 12; producer execution and native-rate provenance are not authenticated.",
                       "The declared --fps 12 evaluation setting follows the organizer demo and stored records. The duration fingerprint is consistent with that setting; it does not prove the producing frame rate. No target-pose conversion or length modification was performed."]}


def admit_local(tag, bank_checks, ck, train, fidelity, static, probes):
    reasons = ["historical_architectural_control_not_named_reproduction"]
    if not fidelity["named_method_fidelity"]:
        reasons.append("defining_named_method_mechanisms_missing_or_changed")
    if not train["raw_equals_admitted_train"]:
        reasons.append("raw_training_archive_contains_nonadmitted_sources")
    if not ck["frozen_historical_training_provenance_complete"]:
        reasons.append("checkpoint_lacks_frozen_training_provenance")
    if static["per_result_best_head"]["proved_in_bound_source"]:
        reasons.append("historical_per_result_best_head_selection")
    if tag == "signidd" and static["unseeded_continuous_diffusion"]["proved_in_bound_local_path"]:
        reasons.append("generation_rng_seed_not_bound")
    if tag == "pt" and not probes["pt"]["historical_export_horizon"]["invariant"]:
        reasons.append("historical_export_changes_with_batch_horizon")
    if tag in ("soke", "g2p_ddm") and not probes["vq"]["prefix_with_code_zero_padding"]["invariant"]:
        reasons.append("shared_vq_decoder_changes_with_padded_horizon")
    for split, result in bank_checks.items():
        if not result["structure_gloss_and_length_valid"]:
            reasons.append(f"{split}_bank_structure_gloss_or_length_invalid")
        if not result["confidence"]["valid_probability_channel"]:
            reasons.append(f"{split}_confidence_outside_unit_interval_or_nonfinite")
    return {"id": tag, "status": "REJECTED", "reasons": sorted(set(reasons)),
            "claim_ceiling": fidelity["ceiling"], "split_checks": bank_checks,
            "checkpoint": ck, "method_fidelity": fidelity}


def dependencies():
    paths = {"implementation": "scripts/audit_recent_control_admission.py",
             "tests": "tests/test_audit_recent_control_admission.py", "local_base": CB,
             "local_vq": VQ, "local_pt": PT, "historical_scorer": "scripts/score_csl_mska_fast.py",
             "mska_train": MSKA, "chain": "scripts/run_csl_baselines_chain.sh",
             "protocol": "outputs/csl_baselines/PROTOCOL.md",
             "official_split": f"{OFFICIAL}/split_1.txt", "official_annotations": f"{OFFICIAL}/csl2020ct_v2.pkl",
             "train_archive": "external/baselines/MSKA/data/CSL-Daily/CSL-Daily.train",
             "admission": f"{V4}/archive_admission.json", "admitted_protocol": f"{V4}/protocol.json",
             "external_provenance": "outputs/baseline_protocol/provenance.json",
             "official_status": "outputs/revision/nonhuman_closure_20260907/recent_slp_official_artifact_update.md"}
    for split in COUNTS:
        paths[f"manifest_{split}"] = f"data/csl-daily/csl_daily_{split}.json"
    for split in ("dev", "test"):
        paths[f"length_reference_{split}"] = f"outputs/csl_msla_inputs/CSL-Daily.{split}_native_hclen_b0_budget000"
        for tag in TAGS:
            paths[f"bank_{split}_{tag}"] = f"outputs/csl_msla_inputs/CSL-Daily.{split}_{tag}"
    for tag in (*TAGS, "vqvae"):
        folder = CK_NAMES.get(tag, tag)
        for name, suffix in (("checkpoint", "best.pt"), ("training_log", "train_log.json")):
            paths[f"{name}_{tag}"] = f"outputs/csl_baselines/{folder}/{suffix}"
    for path in sorted(source("outputs/csl_baselines").glob("*.log")):
        paths["historical_log_" + path.stem] = str(path)
    for tag in TAGS:
        paths[f"historical_score_{tag}"] = f"outputs/csl_mska_results/raw_fast_{tag}.log"
    fidelity = method_fidelity()
    for tag, detail in fidelity.items():
        for index, rec in enumerate(detail["official_evidence"] + detail["official_markers"]):
            paths[f"official_{tag}_{index}"] = rec["path"]
    paths["external_readme"] = f"{EVAL}/README.md"
    # Bind the entire tracked evaluator implementation dependency tree.
    for relative in git(["ls-files"], EVAL).splitlines():
        if relative.endswith((".py", ".txt", ".yaml", ".yml")):
            paths["external_source_" + relative] = f"{EVAL}/{relative}"
    provenance = read_json(source(paths["external_provenance"]))
    for name, rec in provenance["evaluator"]["bt_model_files"].items():
        paths["external_bt_"+name] = rec["path"]
    for run in provenance["runs"]:
        for name in ("pred_file", "gt_file"):
            paths["external_"+run["tag"]+"_"+name] = run[name]["path"]
        for name in ("results_json", "log"):
            paths["external_"+run["tag"]+"_"+name] = run[name]
        paths["external_"+run["tag"]+"_text_predictions"] = str(source(run["results_json"]).with_name(run["tag"]+"_text_preds.pt"))
    return paths


def seal_design(output):
    require(not Path(output).exists(), "immutable design output already exists")
    paths = dependencies()
    inputs = {name: record(path) for name, path in sorted(paths.items())}
    design = {"schema": SCHEMA, "expected_counts": COUNTS, "probe_policy": PROBE,
              "environment": environment(), "inputs": inputs,
              "admission_policy": {"local_named_reproductions": "always reject historical generic controls",
                                   "external": "only the 641-item indexed organizer PT release-bank re-score evaluated with declared --fps 12, if every frozen stored-artifact closure check passes; no authenticated producer execution or native-rate provenance"},
              "claim_ceiling": "Post-hoc audit of stored artifacts and bounded counterexamples, not fresh evaluation or untouched-test evidence.",
              "resource_policy": {"min_available_kib": MIN_AVAILABLE_KIB, "max_peak_rss_kib": MAX_RSS_KIB,
                                  "one_memory_heavy_process": True}}
    output = new_dir(output)
    write_new(output/"design.json", design)
    write_new(output/"seal.json", {"schema": SCHEMA, "design": record(output/"design.json")})
    return digest(output/"design.json")


def load_design(path, expected_hash):
    path = owned(path)
    require(len(expected_hash) == 64 and digest(path) == expected_hash, "trusted design digest mismatch")
    design = read_json(path)
    require(design["schema"] == SCHEMA and design["expected_counts"] == COUNTS
            and design["probe_policy"] == PROBE, "sealed audit policy differs")
    require(design["environment"] == environment(), "runtime differs from sealed environment")
    for rec in design["inputs"].values():
        verify_record(rec)
    require(design["inputs"]["implementation"] == record(__file__), "running implementation differs from design")
    return design


def available_memory_kib():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0])


@contextmanager
def heavy_lock():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    path = owned(OUT/"heavy.lock")
    # a+ never truncates; the lock file itself contains no scientific output.
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AuditError("another recent-control audit owns the heavy slot") from error
        try:
            require(available_memory_kib() >= MIN_AVAILABLE_KIB, "less than 11 GiB available memory")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resource_check():
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    require(peak < MAX_RSS_KIB, "audit exceeded strict <7 GiB peak RSS")
    return peak


def science(design, design_hash):
    import torch
    torch.set_num_threads(1)
    manifests, official = official_manifests()
    static, fidelity = static_audit(), method_fidelity()
    # One large archive at a time; discard before opening any output bank.
    archive = load_archive_metadata(design["inputs"]["train_archive"]["path"])
    require(isinstance(archive, dict), "raw train archive is not indexed")
    raw_ids, admitted_ids = set(archive), set(manifests["train"])
    import random
    shuffled = sorted(raw_ids)
    random.Random(0).shuffle(shuffled)
    inferred_val, inferred_fit = set(shuffled[:512]), set(shuffled[512:])
    training = {"raw_count": len(raw_ids), "admitted_count": len(admitted_ids),
                "extra_ids": sorted(raw_ids-admitted_ids), "missing_ids": sorted(admitted_ids-raw_ids),
                "raw_ids_sha256": value_hash(sorted(raw_ids)), "admitted_ids_sha256": value_hash(sorted(admitted_ids)),
                "raw_equals_admitted_train": raw_ids == admitted_ids,
                "archive_read_scope": "Restricted metadata-only pickle decoding: exact IDs, insertion order, labels and tensor shapes; training numeric arrays are not materialized or declared finite.",
                "extra_records": {sid: archive[sid] for sid in sorted(raw_ids-admitted_ids)},
                "seed0_source_reconstructed_partition": {"fit_n": len(inferred_fit), "validation_n": len(inferred_val),
                    "fit_ids_sha256": value_hash(sorted(inferred_fit)), "validation_ids_sha256": value_hash(sorted(inferred_val)),
                    "nonadmitted_fit_ids": sorted(inferred_fit-admitted_ids),
                    "nonadmitted_validation_ids": sorted(inferred_val-admitted_ids),
                    "nonadmitted_normalization_ids": sorted(set(list(archive)[:600])-admitted_ids),
                    "scope": "Reconstructed from current source and chain seed; absent historical manifest prevents execution attestation."},
                "loader_uses_unfiltered_archive": symbol(CB, "load_train"),
                "holdout_policy": "seeded shuffle, first 512 raw-archive IDs reserved for validation; all raw IDs used for vocabulary and first-600 insertion-order normalization",
                "holdout_evidence": [symbol(CB, "train_pt"), symbol(CB, "train_diff"), symbol(CB, "pose_stats")],
                "historical_seed_record": {"training_cli_default": 0, "chain_seed": 0,
                    "chain_evidence": matching_lines("scripts/run_csl_baselines_chain.sh", "--seed 0"),
                    "runtime_and_rng_state": "not preserved in checkpoints; CLI/chain text is not proof of exact historical RNG state"}}
    require(not training["missing_ids"], "raw archive is missing admitted train sources")
    declared = read_json(source(f"{V4}/archive_admission.json"))
    require(declared["raw_pose_sources"] == len(raw_ids) and declared["admitted_pose_sources"] == len(admitted_ids)
            and declared["excluded_pose_sources"] == training["extra_ids"], "prior admission record no longer agrees with raw archive")
    prior = read_json(source(f"{V4}/protocol.json"))["inputs"]
    for role, current in (("pose_archive", "train_archive"), ("train_manifest", "manifest_train"),
                          ("dev_manifest", "manifest_dev"), ("test_manifest", "manifest_test")):
        require(prior[role]["sha256"] == design["inputs"][current]["sha256"], "CSL admitted-source hash differs")
    del archive; gc.collect(); resource_check()
    banks = {tag: {} for tag in TAGS}
    for split in ("dev", "test"):
        with source(design["inputs"][f"length_reference_{split}"]["path"]).open("rb") as handle:
            reference = pickle.load(handle)
        require(set(reference) == set(manifests[split]), "non-GT reference IDs differ")
        lengths = {sid: int(row["keypoint"].shape[0]) for sid, row in reference.items()}
        require(all(t > 0 for t in lengths.values()), "nonpositive reference length")
        del reference; gc.collect()
        for tag in TAGS:
            with source(design["inputs"][f"bank_{split}_{tag}"]["path"]).open("rb") as handle:
                bank = pickle.load(handle)
            banks[tag][split] = bank_summary(bank, manifests[split], lengths)
            banks[tag][split]["length_reference_sha256"] = value_hash(lengths)
            del bank; gc.collect(); resource_check()
    checkpoints = {}
    for tag in (*TAGS, "vqvae"):
        ck = torch.load(source(design["inputs"][f"checkpoint_{tag}"]["path"]), map_location="cpu", weights_only=True)
        metric = "val_pose" if tag == "pt" else "val_eps" if tag == "signidd" else "val_rec" if tag == "vqvae" else "val_ce"
        checkpoints[tag] = checkpoint_summary(ck, read_json(source(design["inputs"][f"training_log_{tag}"]["path"])), metric)
        checkpoints[tag]["frozen_file"] = design["inputs"][f"checkpoint_{tag}"]
        if tag in ("soke", "g2p_ddm"):
            require(source(ck["vqvae"]) == source(design["inputs"]["checkpoint_vqvae"]["path"]), "VQ checkpoint pointer differs")
            checkpoints[tag]["vq_dependency_bound_now"] = design["inputs"]["checkpoint_vqvae"]
            checkpoints[tag]["historical_vq_hash_binding"] = "absent; checkpoint stores path only"
        del ck; gc.collect()
    probes = real_probes(); resource_check()
    rows = [admit_local(tag, banks[tag], checkpoints[tag], training, fidelity[tag], static, probes) for tag in TAGS]
    try:
        external = external_release_audit(read_json(source(design["inputs"]["external_provenance"]["path"])))
    except (AuditError, OSError, KeyError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        external = {"complete": False, "error_type": type(error).__name__, "error": str(error)}
    rows.append({"id": "organizer_progressive_transformer_release", "status": "ADMITTED" if external["complete"] else "REJECTED",
                 "reasons": [] if external["complete"] else ["external_stored_artifact_closure_failed: "+external["error"]],
                 "claim_ceiling": "Admitted by this audit as a 641-item organizer PT release-bank re-score evaluated with declared --fps 12 and the frozen SLRTP evaluator; no authenticated producer execution/native-rate provenance, locally reproduced generator, or recent-method superiority claim.",
                 "external_evidence": external})
    return {"schema": SCHEMA, "design_sha256": design_hash, "official_manifest_checks": official,
            "training_admission": training, "static_checks": static, "invariance_probes": probes,
            "shared_vq_checkpoint": checkpoints["vqvae"], "rows": rows,
            "no_training_corpus_generation_or_recognizer_run": True,
            "probe_exception": "Four tiny PT forward-generation calls and four tiny VQ decode calls only, using frozen checkpoints and declared synthetic inputs.",
            "overall_claim_ceiling": design["claim_ceiling"]}


def run_audit(design_path, design_hash, output):
    output = owned(no_alias_path(output))
    require(not Path(output).exists(), "immutable run output already exists")
    design = load_design(design_path, design_hash)  # fail before deserialization
    output = new_dir(output)
    start = time.monotonic()
    _, directory = run_directory(output)
    execution = {"schema": EXECUTION_SCHEMA, "run_id": secrets.token_hex(32),
                 "start_unix_ns": time.time_ns(), "output_directory": directory}
    validate_execution_identity(execution, output)
    write_new(output/"start.json", {"schema": SCHEMA, "design_sha256": design_hash,
              "environment": environment(), "execution": execution})
    try:
        with heavy_lock():
            result = science(design, design_hash)
        write_new(output/"scientific.json", result)
        write_new(output/"telemetry.json", {"elapsed_seconds": time.monotonic()-start,
                    "run_id": execution["run_id"], "completed_unix_ns": time.time_ns(),
                    "peak_rss_kib": resource_check(), "environment": environment(),
                    "worker_identity": {"requested_model": "gpt-6-astra", "observed_model": "unknown", "observed_effort": "unknown"},
                    "token_telemetry": "unknown", "hidden_exceptions": 0})
        paths = [output/name for name in RUN_FILES if name != "output_seal.json"]
        write_output_seal(output/"output_seal.json", {"schema": SCHEMA, "design_sha256": design_hash,
                    "execution": execution,
                    "artifact_inodes": {p.name: inode_identity(p) for p in paths},
                    "outputs": {p.name: record(p) for p in paths}})
    except BaseException as error:
        write_new(output/"failure.json", {"schema": SCHEMA, "error_type": type(error).__name__,
                    "execution": execution,
                    "error": str(error), "elapsed_seconds": time.monotonic()-start,
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "rows": [{"id": tag, "status": "REJECTED", "reasons": ["audit_incomplete"],
                              "claim_ceiling": "No admission; required audit did not complete."}
                             for tag in (*TAGS, "organizer_progressive_transformer_release")]})
        raise
    return result


def verify_run(design_path, design_hash, output):
    output, directory = run_directory(output)
    artifacts = run_artifact_identities(output)
    seal = read_json(output/"output_seal.json")
    require(seal["schema"] == SCHEMA and seal["design_sha256"] == design_hash, "output seal policy differs")
    require(seal["seal_inode"] == artifacts["output_seal.json"], "output seal inode binding differs")
    require(seal["artifact_inodes"] == {k: v for k, v in artifacts.items() if k != "output_seal.json"},
            "sealed execution artifact inode binding differs")
    start = read_json(output/"start.json")
    require(start["schema"] == SCHEMA and start["design_sha256"] == design_hash,
            "start metadata design binding differs")
    require(start["execution"] == seal["execution"], "seal/start execution identities differ")
    validate_execution_identity(seal["execution"], output)
    design = load_design(design_path, design_hash)
    require(start["environment"] == design["environment"], "start environment differs from design")
    require(set(seal["outputs"]) == {"start.json", "scientific.json", "telemetry.json"}, "output seal closure incomplete")
    require({p.name for p in output.iterdir()} == set(seal["outputs"]) | {"output_seal.json"}, "unexpected/missing run artifact")
    for name, rec in seal["outputs"].items():
        require(source(rec["path"]) == output/name, "output record points outside its run")
        verify_record(rec)
    telemetry = read_json(output/"telemetry.json")
    require(telemetry["run_id"] == seal["execution"]["run_id"], "telemetry run ID differs")
    require(type(telemetry["completed_unix_ns"]) is int
            and telemetry["completed_unix_ns"] >= seal["execution"]["start_unix_ns"],
            "execution completion precedes start")
    result = read_json(output/"scientific.json")
    require(result["design_sha256"] == design_hash and result["schema"] == SCHEMA, "scientific design binding differs")
    require([r["id"] for r in result["rows"]] == [*TAGS, "organizer_progressive_transformer_release"], "row set differs")
    for row in result["rows"]:
        require(row["status"] in ("ADMITTED", "REJECTED") and bool(row["claim_ceiling"]), "missing admission status/ceiling")
        require((row["status"] == "REJECTED") == bool(row["reasons"]), "admission status/reason mismatch")
        require(row["id"] not in TAGS or row["status"] == "REJECTED", "historical local control cannot enter named-method ranking")
        if row["id"] == "organizer_progressive_transformer_release":
            require((row["status"] == "ADMITTED") == bool(row["external_evidence"]["complete"]), "external completeness/status mismatch")
    require(run_directory(output)[1] == directory and run_artifact_identities(output) == artifacts,
            "execution directory/artifacts changed during verification")
    return {"verified": True, "scientific_sha256": digest(output/"scientific.json"),
            "execution": seal["execution"], "artifact_inodes": artifacts}


def compare_runs(design_path, design_hash, output, repeat):
    a, b = compare_preflight(output, repeat)
    first = verify_run(design_path, design_hash, a)
    second = verify_run(design_path, design_hash, b)
    require(first["execution"]["run_id"] != second["execution"]["run_id"],
            "repeat must have a distinct execution run ID")
    require(first["execution"]["start_unix_ns"] != second["execution"]["start_unix_ns"],
            "repeat must have a distinct execution start time")
    # Check alias metadata again after verification to reject files replaced in
    # flight. This is a consistency check, not authentication of producer work.
    compare_preflight(a, b)
    require(first["scientific_sha256"] == second["scientific_sha256"], "repeat scientific JSON differs")
    return {"verified": True, "deterministic_repeat_identical": True,
            "scientific_sha256": first["scientific_sha256"],
            "executions": [first["execution"], second["execution"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run", "verify", "compare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--design-sha256")
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    if args.action == "seal":
        print(json.dumps({"design_sha256": seal_design(args.output)}, sort_keys=True))
        return
    require(args.design is not None and args.design_sha256 is not None, "explicit trusted design path/hash required")
    if args.action == "run":
        result = run_audit(args.design, args.design_sha256, args.output)
        print(json.dumps({"rows": [{k:r[k] for k in ("id", "status")} for r in result["rows"]],
                          "scientific_sha256": digest(args.output/"scientific.json")}, sort_keys=True))
    elif args.action == "compare":
        require(args.repeat is not None, "repeat output required")
        print(json.dumps(compare_runs(args.design, args.design_sha256, args.output, args.repeat), sort_keys=True))
    else:
        print(json.dumps(verify_run(args.design, args.design_sha256, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
