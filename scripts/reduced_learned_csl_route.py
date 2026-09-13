"""Predeclared, reduced CSL allocation: dev fit -> freeze -> certify/seal -> score.

The freeze command never reads test annotations or test recognizer results.
This is a post-test one-feature extension, not the original PHOENIX router.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import pickle
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_retrieval_gloss_plans import normalize_caption

OUTPUT_ROOT = ROOT / "outputs/revision/astra_open_closure_20260906"
PREREG = OUTPUT_ROOT / "reduced_learned_csl_preregistration.json"
PREREG_SHA = "1a2b93ddea4aa617527d77164c1955d132c3637c4faea4348aba3b5087015695"
HEADS = ("ensemble_last_hyp", "body_hyp", "fuse_hyp", "left_hyp", "right_hyp")
PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.85,
              min_samples_leaf=20, random_state=0)
POLICY_ROUTE = "csl_text_native_reduced_gbdt40_v1"
SCOPE = "post-test reduced one-feature CSL allocation; not PHOENIX-method equivalence or human intelligibility"
FULL_ADAPTER_SHA = "835706d97601f801ee8da003468d3ed2b32f724c2228ae01edb346dc505b1473"
NATIVE_VERIFIER_SHA = "ae65fb2b1d486354866a06d2160c7bae570ada5e6fd4c3e9b4c5e558ef62836b"
NATIVE_HELPER_SHA = "040809dd3259ed5e7f98e78b4bee944a00403e145ccf4a17bbfbee9fc3dd8cd7"
MSKA_DATASET_SHA = "85004ee684452599a2fde75e5d432b33b85062854ea9c2370b9fca80ee1713cf"
MSKA_RECOGNITION_SHA = "d31a968d2ad6b4641f37116ae00c02e1514eb9213a03bdf2a87660071eee6621"
S2G_EVALUATOR_BY_SPLIT = {
    "dev": "59f77a5deeb1b8f5ea985b0eab251f68e31e7d575d406a52069a76d239859c92",
    "test": "57b74fa11242cc3b11fa2c190eeceb9da7a3e91db1b386305c67e0aa862bf7b9",
}
SEALED_TEST_EVALUATOR_NAME = "run_csl_mska_split_eval_57b74fa1.py"
SUPPORTED_TEST_WRAPPER_NAME = "run_csl_mska_split_eval.py"
MSKA_LOCAL_CODE_SHA = {
    "Rouge.py": "65762148a0dfdef40cd3b6482054c2a5c962144907e642d175305f2218e907fd",
    "Tokenizer.py": "a5f21a4f8dad5cd2e251178028355ed79384812a28cadba556e4b892720ffe13",
    "Visualhead.py": "1a1c2b0a646b1e35a374b5152ddf5d3c588ee3aa3ec2d047e50e62ef4a3bb88d",
    "datasets.py": MSKA_DATASET_SHA,
    "metrics.py": "cc18a21a19d9fe817a8177a6da2eb277fea9863b1de1c3f11fdd08acfe859445",
    "recognition.py": MSKA_RECOGNITION_SHA,
    "sacrebleu.py": "511541090733cdeff88e334a6ed651ef94b46cd37163666cbbfc121962a3723f",
    "utils.py": "59d44ab4f070f05b354209e7e917969ce3fbf00dc6ba84ded9145a5d7f4fb43b",
}
MSKA_LOCAL_BUNDLE_SHA = "c1fa73518427e7303eb2909d5b29b4b1204f7ed4d63437cdb16e30117ee404b4"
MSKA_GLOSS_VOCABULARY_SHA = "3b347a8a3c81ad2a2e79eaa28b89dcc0502cd09070f29175bb3e86f338afdf4d"
RAW_LABEL_ARCHIVE_RELATIVE = "data/csl-daily/CSL-Daily/sentence_label/csl2020ct_v2.pkl"
RAW_LABEL_ARCHIVE_SHA = "1fd29a33cd87550b9228f88aa35df26e6e906d0d27177648d5123f6b1babd741"


class ProtocolError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def confined(path):
    value = Path(path)
    value = (value if value.is_absolute() else ROOT/value).resolve()
    require(value != ROOT.resolve() and value.is_relative_to(ROOT.resolve()), "path escapes project root")
    return value


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, expected=None):
    path = confined(path)
    if expected is not None:
        require(digest(path) == expected, f"hash mismatch: {path.name}")
    def invalid_constant(value):
        raise ProtocolError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=no_duplicates, parse_constant=invalid_constant)


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n")


def new_directory(path):
    path = confined(path)
    require(path != OUTPUT_ROOT.resolve() and path.is_relative_to(OUTPUT_ROOT.resolve()), "outputs must be inside the versioned revision directory")
    require(not path.exists(), "use a new output directory; frozen evidence is immutable")
    path.mkdir(parents=True)
    return path


def exact_int(value, *, minimum=0):
    require(type(value) is int and value >= minimum, "expected exact nonnegative integer")
    return value


def read_prereg(path=PREREG):
    value = read_json(path, PREREG_SHA)
    require(value["schema"] == "reduced_learned_csl_pose_evaluation_preregistration_v1", "unsupported preregistration")
    expected = {"class": "sklearn.ensemble.GradientBoostingRegressor", **PARAMS}
    require(canonical(value["learned_allocation"]["estimator"]) == canonical(expected), "estimator differs from preregistration")
    require(value["learned_allocation"]["features"] == ["train_caption_donor_similarity"], "unapproved learned features")
    return value


def checked_trace(rows):
    require(isinstance(rows, list) and rows, "trace must be nonempty")
    result = {}
    for row in rows:
        sid = row.get("id")
        require(isinstance(sid, str) and sid and sid not in result, "missing/duplicate trace ID")
        value = row.get("similarity")
        require(type(value) in (float, int) and math.isfinite(value), "nonfinite or invalid similarity")
        # Deliberately project only the one preregistered feature.
        result[sid] = {"similarity": float(value), "retrieved_id": row["retrieved_id"],
                       "forbidden_source_ids": list(row["forbidden_source_ids"]),
                       "exact_caption_source_ids": list(row["exact_caption_source_ids"])}
    return result


_METRIC = None


def metric_function():
    global _METRIC
    if _METRIC is None:
        source = ROOT / "external/baselines/MSKA/metrics.py"
        spec = importlib.util.spec_from_file_location("reduced_csl_frozen_mska_metrics", source)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(source.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        require((module.WER_COST_DEL, module.WER_COST_INS, module.WER_COST_SUB) == (3, 3, 4), "native metric alignment contract changed")
        _METRIC = module.wer_single
    return _METRIC


def edits(reference, hypothesis):
    require(isinstance(reference, str) and isinstance(hypothesis, str), "reference/hypothesis must be strings")
    r, h = reference.strip().split(), hypothesis.strip().split()
    require(r, "empty reference denominator")
    # The frozen evaluator uses uint8 weighted DP. Reject possible overflow
    # instead of silently changing the metric or trusting wrapped edit counts.
    require(3*max(len(r), len(h))+min(len(r), len(h))+4 <= 255, "native MSKA uint8 alignment could overflow")
    raw = metric_function()(reference, hypothesis)
    result = {k: int(raw[k]) for k in ("num_err", "num_del", "num_ins", "num_sub", "num_ref")}
    require(result["num_ref"] == len(r) and result["num_err"] == result["num_del"]+result["num_ins"]+result["num_sub"], "metric counts inconsistent")
    require(all(v >= 0 for v in result.values()), "negative metric counts")
    return {**result, "num_hyp": len(h)}


def summarize(rows):
    require(rows, "empty result set")
    summaries = {}
    for head in HEADS:
        total = Counter()
        for row in rows.values():
            total.update(edits(row["ref"], row[head]))
        summaries[head] = {**total, "wer": 100*total["num_err"]/total["num_ref"],
                           "hyp_ref_ratio": total["num_hyp"]/total["num_ref"]}
    return summaries


def validate_result(result, ids, split, *, expected_refs=None, prereg=None,
                    input_digest=None, after_ns=None):
    # Split gate precedes access to prediction or reference values.
    require(result.get("split") == split, f"only {split} results are allowed in this stage")
    require(split in S2G_EVALUATOR_BY_SPLIT and result.get("implementation_sha256") == S2G_EVALUATOR_BY_SPLIT[split],
            f"unaudited evaluator for {split}; runner must match its frozen stage")
    ids = list(ids)
    require(ids and len(ids) == len(set(ids)), "invalid expected result IDs")
    require(type(result.get("n")) is int and result["n"] == len(ids), "result denominator differs")
    rows = result.get("per_id")
    require(isinstance(rows, dict) and set(rows) == set(ids), "result IDs differ")
    declared_heads = result.get("head_names", [])
    require(isinstance(declared_heads, list) and len(declared_heads) == len(HEADS) and
            set(declared_heads) == set(HEADS), "incomplete or unexpected head declaration")
    for sid in ids:
        row = rows[sid]
        require(isinstance(row, dict) and set(row) == {"ref", *HEADS}, f"incomplete or unexpected heads: {sid}")
        require(all(isinstance(v, str) for v in row.values()), "non-string prediction/reference")
        if expected_refs is not None:
            require(row["ref"] == expected_refs[sid], f"reference mismatch: {sid}")
    for field, expected in (("batch_size", 1), ("num_workers", 0), ("beam_size", 5)):
        require(type(result.get(field)) is int and result[field] == expected, f"unfrozen {field}")
    require(exact_int(result.get("checkpoint_state_key_count"), minimum=1) > 0, "unverified checkpoint")
    implementation = result.get("implementation_sha256")
    require(isinstance(implementation, str) and len(implementation) == 64 and
            all(c in "0123456789abcdef" for c in implementation), "missing evaluator implementation hash")
    if prereg is not None:
        require(result.get("checkpoint_sha256") == prereg["frozen_inputs"]["mska_checkpoint"]["sha256"], "checkpoint mismatch")
        require(result.get("config_sha256") == prereg["frozen_inputs"]["mska_config"]["sha256"], "config mismatch")
    if input_digest is not None:
        require(result.get("input_sha256") == input_digest, "result input hash mismatch")
        require(result.get("isolated_input") is True, "evaluation must use isolated input without shared-split mutation")
    if split == "test":
        require(after_ns is not None, "test evaluation requires a frozen certificate-seal timestamp")
        exact_int(after_ns, minimum=1)
        start = exact_int(result.get("started_unix_ns"), minimum=1)
        end = exact_int(result.get("completed_unix_ns"), minimum=1)
        require(after_ns < start <= end, "test evaluation began before certificate seal")
    computed = summarize(rows)
    require(isinstance(result.get("summary"), dict) and set(result["summary"]) == set(HEADS), "aggregate head coverage differs")
    for head in HEADS:
        claimed = result.get("summary", {}).get(head, {})
        required = {"num_err", "num_del", "num_ins", "num_sub", "num_ref", "wer"}
        if split == "test":
            required |= {"num_hyp", "hyp_ref_ratio"}
        require(isinstance(claimed, dict) and required <= set(claimed) <= required | {"num_hyp", "hyp_ref_ratio"}, "unsupported or missing aggregate fields")
        for field in ("num_err", "num_del", "num_ins", "num_sub", "num_ref"):
            require(type(claimed.get(field)) is int and claimed[field] == computed[head][field], f"reported {head} {field} differs")
        # Historical native dev results omit length fields. Always recompute
        # them, and validate them whenever the evaluator actually reports them.
        if "num_hyp" in claimed:
            require(type(claimed["num_hyp"]) is int and claimed["num_hyp"] == computed[head]["num_hyp"], "reported num_hyp differs")
        for field in ("wer", "hyp_ref_ratio"):
            if field in claimed:
                require(type(claimed[field]) in (float, int) and math.isfinite(claimed[field]) and
                        abs(claimed[field]-computed[head][field]) < 1e-10, f"reported {field} differs")
    return {sid: copy.deepcopy(rows[sid]) for sid in ids}


def fit_dev(local_result, whole_result, trace, expected_refs):
    ids = sorted(trace)
    local = validate_result(local_result, ids, "dev", expected_refs=expected_refs)
    whole = validate_result(whole_result, ids, "dev", expected_refs=expected_refs)
    x = np.asarray([[trace[sid]["similarity"]] for sid in ids], dtype=np.float64)
    require(np.isfinite(x).all(), "nonfinite learned feature")
    denominator = sum(len(local[sid]["ref"].split()) for sid in ids)
    target = np.asarray([100*(edits(local[sid]["ref"], local[sid][HEADS[0]])["num_err"]-
                              edits(whole[sid]["ref"], whole[sid][HEADS[0]])["num_err"])/denominator for sid in ids])
    require(np.isfinite(target).all(), "nonfinite target")
    model = GradientBoostingRegressor(**PARAMS).fit(x, target)
    audit = {"ids": ids, "features": x.tolist(), "targets": target.tolist(),
             "reference_tokens": denominator, "fit_split": "dev", "row_order": "lexical ID",
             "feature_names": ["train_caption_donor_similarity"], "estimator": PARAMS.copy()}
    return model, audit


def export_model(model):
    return {"schema": "reduced_csl_gbdt_trees_v1", "sklearn_version": sklearn.__version__,
            "estimator": model.get_params(), "initial_prediction": float(model.init_.constant_[0, 0]),
            "n_features": int(model.n_features_in_),
            "trees": [{"left": tree.tree_.children_left.tolist(), "right": tree.tree_.children_right.tolist(),
                       "feature": tree.tree_.feature.tolist(), "threshold": tree.tree_.threshold.tolist(),
                       "value": tree.tree_.value[:, 0, 0].tolist()} for tree in model.estimators_[:, 0]]}


def predict_export(model, features):
    x = np.asarray(features, dtype=np.float32)
    require(x.ndim == 2 and x.shape[1] == 1 and np.isfinite(x).all(), "model requires one finite feature")
    predictions = np.full(len(x), model["initial_prediction"], dtype=np.float64)
    for tree in model["trees"]:
        for i, row in enumerate(x):
            node = 0
            while tree["left"][node] != -1:
                node = tree["left"][node] if row[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            predictions[i] += model["estimator"]["learning_rate"] * tree["value"][node]
    require(np.isfinite(predictions).all(), "nonfinite model predictions")
    return predictions


def admit(ids, scores, whole_lengths, local_lengths):
    require(ids and all(isinstance(s, str) and s for s in ids) and len(set(ids)) == len(ids), "missing/duplicate allocation IDs")
    require(len(scores) == len(whole_lengths) == len(local_lengths) == len(ids), "allocation shapes differ")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in scores), "scores must be finite")
    r = [exact_int(v, minimum=1) for v in whole_lengths]
    f = [exact_int(v, minimum=1) for v in local_lengths]
    capacity, used = 2*sum(f), 0
    mask = {sid: False for sid in ids}
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    for i in order:
        cost = 3*r[i]+2*f[i]
        if used+cost <= capacity:
            mask[ids[i]], used = True, used+cost
    replay = sum(r[i] for i, sid in enumerate(ids) if mask[sid])
    emitted = sum(r[i] if mask[sid] else f[i] for i, sid in enumerate(ids))
    require(5*replay <= 2*emitted, "integer budget failure")
    return mask, {"ranked_ids": [ids[i] for i in order], "capacity": capacity, "used": used,
                  "replay_clips": sum(mask.values()), "replay_frames": replay, "emitted_frames": emitted,
                  "integer_budget": f"{5*replay} <= {2*emitted}", "optimality_claimed": False}


def checked_bank(bank, ids):
    import torch
    require(isinstance(bank, dict) and set(bank) == set(ids), "candidate bank IDs differ")
    for sid, value in bank.items():
        require(torch.is_tensor(value) and value.dtype == torch.float32 and value.ndim == 3 and
                value.shape[1:] == (133, 3) and len(value) > 0 and bool(torch.isfinite(value).all()), f"invalid native pose: {sid}")
    return bank


def materialize(mask, local, whole, ledger, trace, training_ids):
    ids = list(mask)
    checked_bank(local, ids); checked_bank(whole, ids)
    require(set(ledger) == set(trace) == set(ids), "recipe/trace IDs differ")
    require(not set(ids) & training_ids, "query/training source overlap")
    poses, rows = {}, {}
    for sid in ids:
        require(type(mask[sid]) is bool, "route mask must be Boolean")
        donor = trace[sid]["retrieved_id"]
        require(donor in training_ids and donor not in trace[sid]["exact_caption_source_ids"], "unadmitted or exact-caption donor")
        recipe = ([{"source_id": donor, "start": 0, "end": len(whole[sid]), "out_len": len(whole[sid])}]
                  if mask[sid] else copy.deepcopy(ledger[sid]["local_recipe"]))
        mass = Counter()
        for unit in recipe:
            start, end, n = (exact_int(unit[k], minimum=1 if k in ("end", "out_len") else 0) for k in ("start", "end", "out_len"))
            require(start < end and unit["source_id"] in training_ids, "invalid or unadmitted source interval")
            if not mask[sid]:
                require(unit["source_id"] not in trace[sid]["forbidden_source_ids"], "forbidden local source")
            mass[unit["source_id"]] += n
        poses[sid] = (whole[sid] if mask[sid] else local[sid]).clone()
        require(sum(mass.values()) == len(poses[sid]), "source recipe length differs from candidate")
        rows[sid] = {"whole_replay": mask[sid], "recipe": recipe, "source_mass": dict(mass),
                     "frames": len(poses[sid]), "local_recipe": copy.deepcopy(ledger[sid]["local_recipe"]),
                     "local_forbidden_sources": list(trace[sid]["forbidden_source_ids"]),
                     "omitted_local_plan_tokens": list(ledger[sid]["omitted_local_plan_tokens"])}
    replay = sum(len(poses[s]) for s in ids if mask[s])
    require(5*replay <= 2*sum(len(v) for v in poses.values()), "materialized release exceeds exact budget")
    return poses, rows


def committed_file(path, expected):
    path = confined(path)
    require(digest(path) == expected, f"hash mismatch: {path.name}")
    return path


def input_record(path):
    path = confined(path)
    return {"path": str(path), "sha256": digest(path)}


def resolve_test_evaluator_record(record):
    """Resolve only the exact sealed role after the supported API split.

    The original freeze recorded the then-live path.  Its byte-identical
    implementation now lives at a same-depth immutable path, while the old
    name is a test-only wrapper.  No generic hash fallback is permitted.
    """
    require(isinstance(record, dict) and set(record) == {"path", "sha256"},
            "test evaluator record schema differs")
    require(record["sha256"] == S2G_EVALUATOR_BY_SPLIT["test"],
            "test evaluator role has an unexpected hash")
    recorded = confined(record["path"])
    historical_path = confined(ROOT/"scripts"/SUPPORTED_TEST_WRAPPER_NAME)
    snapshot = confined(ROOT/"scripts"/SEALED_TEST_EVALUATOR_NAME)
    require(recorded in {historical_path, snapshot},
            "test evaluator role has an unsupported recorded path")
    committed_file(snapshot, S2G_EVALUATOR_BY_SPLIT["test"])
    wrapper = committed_file(historical_path, digest(historical_path))
    require(wrapper != snapshot, "supported wrapper must be distinct from the sealed snapshot")
    return {
        "schema": "sealed_test_evaluator_relocation_v1",
        "recorded": dict(record),
        "resolved_snapshot": input_record(snapshot),
        "supported_test_only_wrapper": input_record(wrapper),
        "development_pin_unchanged": S2G_EVALUATOR_BY_SPLIT["dev"],
        "generic_hash_fallback": False,
    }


def source_context(prereg_path=PREREG, whole_provenance=None):
    prereg = read_prereg(prereg_path)
    frozen = prereg["frozen_inputs"]
    protocol_path = confined(frozen["v4_protocol"]["path"])
    protocol = read_json(protocol_path, frozen["v4_protocol"]["sha256"])
    v4 = protocol_path.parent
    require(protocol.get("version") == "CSL-text-native-fixed40-v4" and canonical(protocol.get("budget")) == b"[2,5]", "source construction contract differs")
    prescores = {s: read_json(frozen[f"{s}_prescore"]["path"], frozen[f"{s}_prescore"]["sha256"]) for s in ("dev", "test")}
    raw_traces = {s: read_json(v4/f"{s}_trace.json", prescores[s][f"{s}_trace.json"]) for s in ("dev", "test")}
    traces = {s: checked_trace(raw_traces[s]) for s in ("dev", "test")}
    require(not set(traces["dev"]) & set(traces["test"]), "dev/test query overlap")
    for split in ("dev", "test"):
        require(len(traces[split]) == prereg["splits"][split], "official split count differs")
    wp = confined(whole_provenance or OUTPUT_ROOT/"csl_whole_candidates/provenance.json")
    whole = read_json(wp)
    require(whole.get("protocol_sha256") == frozen["v4_protocol"]["sha256"], "whole bank construction protocol differs")
    require(whole.get("admission_sha256") == digest(v4/"archive_admission.json"), "whole bank admission differs")
    require(whole.get("archive_sha256") == protocol["inputs"]["pose_archive"]["sha256"] and
            whole.get("train_manifest_sha256") == protocol["inputs"]["train_manifest"]["sha256"] and
            whole.get("no_query_annotations_or_poses") is True, "whole bank source boundary differs")
    for split in ("dev", "test"):
        require(whole["splits"][split]["n"] == len(traces[split]) and
                whole["splits"][split]["trace_sha256"] == prescores[split][f"{split}_trace.json"], "whole donor trace/count differs")
    return prereg, protocol, v4, prescores, raw_traces, traces, whole, wp


def verified_raw_label_archive():
    """Hash only: routing must not deserialize this mixed-split label archive."""
    return input_record(committed_file(ROOT/RAW_LABEL_ARCHIVE_RELATIVE, RAW_LABEL_ARCHIVE_SHA))


def local_recognition_code_closure():
    """Pin the recursively discovered repository-local Python import closure."""
    mska = confined(ROOT/"external/baselines/MSKA")
    runner = committed_file(ROOT/"scripts"/SEALED_TEST_EVALUATOR_NAME,
                            S2G_EVALUATOR_BY_SPLIT["test"])
    for relative, expected in MSKA_LOCAL_CODE_SHA.items():
        committed_file(mska/relative, expected)

    def local_modules(module):
        parts = module.split(".")
        found = set()
        # Include package initializers as Python executes them on dotted imports.
        for length in range(1, len(parts)+1):
            initializer = mska.joinpath(*parts[:length])/"__init__.py"
            if initializer.is_file(): found.add(initializer.relative_to(mska).as_posix())
        candidate = mska.joinpath(*parts).with_suffix(".py")
        if candidate.is_file(): found.add(candidate.relative_to(mska).as_posix())
        return found

    def dependencies(path):
        found, nonlocal_roots = set(), set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                require(node.level == 0, "unapproved relative import in frozen recognition closure")
                names = [node.module] if node.module else []
                names += [f"{node.module}.{alias.name}" for alias in node.names if node.module and alias.name != "*"]
            else:
                continue
            for name in names:
                resolved = local_modules(name)
                found.update(resolved)
                if not resolved and not local_modules(name.split(".")[0]):
                    nonlocal_roots.add(name.split(".")[0])
        return found, nonlocal_roots

    seeds, nonlocal_roots = dependencies(runner)
    require(seeds == {"datasets.py", "Tokenizer.py", "recognition.py", "metrics.py"}, "runner local import roots differ")
    pending, edges = list(seeds), {}
    while pending:
        relative = pending.pop()
        if relative in edges: continue
        require(relative in MSKA_LOCAL_CODE_SHA, "unapproved local recognition dependency or import shadow")
        found, outside = dependencies(mska/relative)
        edges[relative] = sorted(found)
        nonlocal_roots.update(outside)
        pending.extend(found)
    require(set(edges) == set(MSKA_LOCAL_CODE_SHA), "recognition import closure membership differs")
    actual = {relative: digest(mska/relative) for relative in sorted(edges)}
    bundle = hashlib.sha256(canonical(actual)).hexdigest()
    require(bundle == MSKA_LOCAL_BUNDLE_SHA, "recognition import-closure bundle differs")
    vocabulary = committed_file(mska/"data/CSL-Daily/gloss2ids.pkl", MSKA_GLOSS_VOCABULARY_SHA)
    return {"schema": "reduced_csl_local_recognition_import_closure_v1", "entrypoint": input_record(runner),
            "files": {relative: input_record(mska/relative) for relative in sorted(edges)},
            "relative_path_sha256": actual, "bundle_sha256": bundle,
            "bundle_algorithm": "SHA256 of UTF-8 canonical JSON relative-path to SHA256 mapping",
            "local_import_edges": {relative: edges[relative] for relative in sorted(edges)},
            "live_gloss_vocabulary": input_record(vocabulary),
            "nonlocal_import_roots": sorted(nonlocal_roots),
            "scope": "recursive static Import/ImportFrom closure, including nested statements, in the MSKA repository",
            "third_party_environment_hashed": False, "runtime_execution_attested": False,
            "poison_caption_test_scope": "native dataset __getitem__/collate methods only; not all imports"}


def pose_only_metadata_contract(prereg):
    """Permit distinct caption metadata only for the audited, frozen S2G path."""
    import yaml
    record = prereg["frozen_inputs"]["mska_config"]
    config_path = committed_file(record["path"], record["sha256"])
    cfg = yaml.safe_load(config_path.read_text())
    require(cfg.get("task") == "S2G" and cfg.get("do_translation") is False and
            cfg.get("do_recognition") is True and cfg["model"]["RecognitionNetwork"].get("input_type") == "keypoint",
            "caption metadata separation requires frozen pose-only S2G evaluation")
    closure = local_recognition_code_closure()
    return {"task": "S2G", "caption_read_by_dataset": False, "config": input_record(config_path),
            "local_recognition_import_closure": closure,
            "accepted_evaluator_sha256_by_split": dict(S2G_EVALUATOR_BY_SPLIT)}


def canonical_annotation_lineage(official_path, ids, expected_refs, expected_texts):
    """Bind two distinct authorities; never normalize or replace query captions."""
    with Path(official_path).open("rb") as handle:
        archive = pickle.load(handle)
    require(isinstance(archive, dict) and set(archive) == set(expected_refs) == set(expected_texts) and
            set(ids) <= set(archive), "canonical full-split annotation membership differs")
    official = {sid: {field: archive[sid][field] for field in ("name", "text", "gloss")} for sid in ids}
    del archive  # Do not retain canonical GT poses alongside candidate banks.
    differences = []
    for sid in sorted(ids):
        row = official[sid]
        require(all(isinstance(v, str) for v in row.values()), "canonical annotations must be strings")
        require(row["name"] == sid and row["gloss"] == expected_refs[sid], "canonical name/gloss annotations differ from frozen split")
        require(isinstance(expected_texts[sid], str), "declared query caption must be a string")
        if row["text"] != expected_texts[sid]:
            differences.append({"id": sid, "generation_query_text": expected_texts[sid], "mska_metadata_text": row["text"]})
    audit = {"schema": "reduced_csl_annotation_lineage_v1", "n": len(ids),
             "generation_and_bootstrap_caption_authority": "preregistered v4 JSON manifest; unchanged",
             "adapter_caption_authority": "exact canonical MSKA metadata; ignored by pinned S2G loader",
             "reference_authority": "exact agreement between declared gloss and canonical MSKA gloss",
             "caption_normalization_applied": False, "caption_mismatch_count": len(differences),
             "caption_mismatches": differences, "name_mismatch_count": 0, "gloss_mismatch_count": 0,
             "canonical_manifest": input_record(official_path),
             "canonical_metadata_sha256": hashlib.sha256(canonical(official)).hexdigest(),
             "generation_caption_sha256": hashlib.sha256(canonical({sid: expected_texts[sid] for sid in ids})).hexdigest()}
    return official, audit


def check_adapter_input(provenance, *, split, ids, source_path, pose_digest,
                        prescore_path, prescore_digest, adapter_path, adapter_digest,
                        expected_refs, expected_texts):
    """Recheck data transformation, not merely a self-consistent provenance pair."""
    import torch
    source_path = committed_file(source_path, pose_digest)
    require(confined(provenance.get("source", "")) == source_path, "adapter source path differs")
    committed_file(adapter_path, adapter_digest)
    require(provenance.get("implementation_sha256") == adapter_digest, "adapter implementation is not the frozen implementation")
    prescore = read_json(prescore_path, prescore_digest)
    expected_bank = prescore.get(source_path.name)
    if expected_bank is None and isinstance(prescore.get("splits"), dict):
        record = prescore["splits"].get(split, {})
        require(confined(record.get("bank", "")) == source_path and record.get("n") == len(ids), "whole-bank frozen path/count differs")
        expected_bank = record.get("sha256")
    require(expected_bank == pose_digest, "source bank is not bound by frozen prescore evidence")
    official_path = ROOT/f"external/baselines/MSKA/data/CSL-Daily/CSL-Daily.{split}"
    require(confined(provenance.get("manifest", "")) == confined(official_path), "adapter metadata path is not the canonical split")
    committed_file(official_path, provenance["manifest_sha256"])
    require(expected_refs is not None and expected_texts is not None and set(ids) <= set(expected_refs) & set(expected_texts), "frozen split annotations are required")
    official, annotation_audit = canonical_annotation_lineage(official_path, ids, expected_refs, expected_texts)
    input_path = committed_file(provenance["output"], provenance["output_sha256"])
    source = checked_bank(torch.load(source_path, weights_only=True, map_location="cpu"), ids)
    with input_path.open("rb") as handle:
        adapted = pickle.load(handle)
    require(isinstance(adapted, dict) and list(adapted) == list(source), "adapter membership/order differs from frozen bank")
    for sid in source:
        row = adapted[sid]
        require(isinstance(row, dict) and set(row) == {"name", "text", "gloss", "keypoint", "num_frames"}, "adapter row schema differs")
        require({field: row[field] for field in ("name", "text", "gloss")} == official[sid], "adapter annotations differ from exact canonical metadata")
        require(type(row["num_frames"]) is int and row["num_frames"] == len(source[sid]), "adapter emitted length differs")
        pose = row["keypoint"]
        require(torch.is_tensor(pose) and pose.dtype == torch.float32 and pose.shape == source[sid].shape and
                pose.detach().cpu().contiguous().numpy().tobytes() == source[sid].detach().cpu().contiguous().numpy().tobytes(),
                f"adapter pose differs from frozen source tensor: {sid}")
    return input_path, annotation_audit


def check_evaluator_input_record(result, result_path, input_path, input_digest, *, split):
    require(split in S2G_EVALUATOR_BY_SPLIT, "unknown evaluation stage")
    require(confined(result.get("result_path", "")) == result_path, "evaluator result path differs")
    log_path = confined(result.get("log_path", ""))
    require(log_path == result_path.with_suffix(".log"), "evaluator log path differs")
    try:
        with log_path.open() as handle:
            header = json.loads(handle.readline(), object_pairs_hook=no_duplicates)
    except (OSError, ValueError) as exc:
        raise ProtocolError("malformed evaluator log header") from exc
    require(isinstance(header, dict), "evaluator log header must be an object")
    expected_fields = {"input", "input_sha256"}
    if split == "test":
        expected_fields |= {"implementation_sha256", "started_unix_ns"}
    require(set(header) == expected_fields, "evaluator log header fields differ")
    require(isinstance(header.get("input"), str) and confined(header["input"]) == input_path and
            header["input_sha256"] == input_digest, "evaluator input path/hash is not the exact adapter output")
    if split == "test":
        require(header["implementation_sha256"] == result.get("implementation_sha256") == S2G_EVALUATOR_BY_SPLIT["test"],
                "test evaluator implementation is not the pinned runner")
        header_started = exact_int(header["started_unix_ns"], minimum=1)
        result_started = exact_int(result.get("started_unix_ns"), minimum=1)
        require(header_started == result_started, "test evaluator start timestamp differs between log and result")
    if "input_path" in result:
        require(confined(result["input_path"]) == input_path, "reported evaluator input path differs")
    require(result.get("input_sha256") == input_digest, "reported evaluator input hash differs")


def load_evaluation(result_path, provenance_path, *, split, route, ids,
                    source_path, pose_digest, prescore_path, prescore_digest,
                    adapter_path, adapter_digest, prereg, expected_refs, expected_texts, after_ns=None):
    result_path = confined(result_path)
    if split == "dev":
        require(result_path.name.startswith("mska_dev_"), "freeze accepts only explicitly named dev result files")
    provenance = read_json(provenance_path)
    require(provenance.get("split") == split and provenance.get("route") == route, "evaluation branch provenance differs")
    require(type(provenance.get("n")) is int and provenance["n"] == len(ids), "provenance count differs")
    require(provenance.get("source_sha256") == pose_digest and
            provenance.get("prescore_hashes_sha256") == prescore_digest, "evaluation candidate commitment differs")
    pose_only_contract = pose_only_metadata_contract(prereg)
    input_path, annotation_audit = check_adapter_input(provenance, split=split, ids=ids, source_path=source_path,
        pose_digest=pose_digest, prescore_path=prescore_path, prescore_digest=prescore_digest,
        adapter_path=adapter_path, adapter_digest=adapter_digest, expected_refs=expected_refs, expected_texts=expected_texts)
    result = read_json(result_path)
    require(result.get("implementation_sha256") == S2G_EVALUATOR_BY_SPLIT[split], f"unaudited evaluator for {split}; runner must match its frozen stage")
    check_evaluator_input_record(result, result_path, input_path, provenance["output_sha256"], split=split)
    require(result.get("canonical_split_sha256") == provenance["manifest_sha256"], "evaluator canonical split differs from adapter metadata")
    rows = validate_result(result, ids, split, expected_refs=expected_refs, prereg=prereg,
                           input_digest=provenance["output_sha256"], after_ns=after_ns)
    # This validation record is newly derived; never rewrite the scored result file.
    return {**result, "validated_annotation_lineage": {**annotation_audit, "pose_only_contract": pose_only_contract}}, rows


def freeze(args):
    """Fit on dev labels/results; test inputs are only committed trace/poses/recipes."""
    import torch
    prereg_path = confined(args.preregistration)
    raw_label_archive = verified_raw_label_archive()
    prereg, protocol, v4, prescores, raw_traces, traces, whole, wp = source_context(prereg_path, args.whole_provenance)
    training_input, dev_input = (protocol["inputs"][k] for k in ("train_manifest", "dev_manifest"))
    training_rows = read_json(training_input["path"], training_input["sha256"])
    training_ids = {row["id"] for row in training_rows}
    require(len(training_ids) == len(training_rows) == prereg["splits"]["train"], "training registry has duplicates/count drift")
    require(not training_ids & (set(traces["dev"]) | set(traces["test"])), "query/training source overlap")
    dev_rows = read_json(dev_input["path"], dev_input["sha256"])
    refs = {row["id"]: row["gloss"] for row in dev_rows}
    texts = {row["id"]: row["text"] for row in dev_rows}
    adapter_path = ROOT/"scripts/prepare_csl_mska_eval_inputs.py"
    require(len(refs) == len(dev_rows) and set(refs) == set(traces["dev"]), "dev annotation IDs differ")
    dev_local, _ = load_evaluation(args.dev_local_result, args.dev_local_provenance, split="dev", route="local",
        ids=sorted(refs), source_path=v4/"dev_local.pt", pose_digest=prescores["dev"]["dev_local.pt"],
        prescore_path=prereg["frozen_inputs"]["dev_prescore"]["path"],
        prescore_digest=prereg["frozen_inputs"]["dev_prescore"]["sha256"],
        adapter_path=adapter_path, adapter_digest=FULL_ADAPTER_SHA, prereg=prereg, expected_refs=refs, expected_texts=texts)
    dev_whole, _ = load_evaluation(args.dev_whole_result, args.dev_whole_provenance, split="dev", route="whole",
        ids=sorted(refs), source_path=whole["splits"]["dev"]["bank"], pose_digest=whole["splits"]["dev"]["sha256"],
        prescore_path=wp, prescore_digest=digest(wp), adapter_path=adapter_path, adapter_digest=FULL_ADAPTER_SHA,
        prereg=prereg, expected_refs=refs, expected_texts=texts)
    require(read_json(args.dev_local_provenance)["manifest_sha256"] == read_json(args.dev_whole_provenance)["manifest_sha256"], "dev branch annotation snapshots differ")
    require(dev_local.get("implementation_sha256") == dev_whole.get("implementation_sha256") and
            dev_local.get("checkpoint_state_key_count") == dev_whole.get("checkpoint_state_key_count"), "dev evaluator implementations differ")
    model, fit_audit = fit_dev(dev_local, dev_whole, traces["dev"], refs)
    ids = sorted(traces["test"])
    features = [[traces["test"][sid]["similarity"]] for sid in ids]
    scores = model.predict(np.asarray(features, dtype=np.float64))
    model_data = export_model(model)
    require(np.array_equal(scores, predict_export(model_data, features)), "exported model fails exact score reconstruction")
    local_path = committed_file(v4/"test_local.pt", prescores["test"]["test_local.pt"])
    whole_path = committed_file(whole["splits"]["test"]["bank"], whole["splits"]["test"]["sha256"])
    local = checked_bank(torch.load(local_path, weights_only=True, map_location="cpu"), ids)
    whole_bank = checked_bank(torch.load(whole_path, weights_only=True, map_location="cpu"), ids)
    rows = read_json(v4/"test_ledger.json", prescores["test"]["test_ledger.json"])
    mask, allocation = admit(ids, scores.tolist(), [len(whole_bank[s]) for s in ids], [len(local[s]) for s in ids])
    poses, ledger = materialize(mask, local, whole_bank, rows, traces["test"], training_ids)
    fixed_mask_list = read_json(v4/"test_mask.json", prescores["test"]["test_mask.json"])
    original_ids = [r["id"] for r in raw_traces["test"]]
    require(len(fixed_mask_list) == len(original_ids) and all(type(v) is bool for v in fixed_mask_list), "fixed mask malformed")
    fixed_mask = dict(zip(original_ids, fixed_mask_list))
    fixed, _ = materialize(fixed_mask, local, whole_bank, rows, traces["test"], training_ids)
    fixed_path = committed_file(v4/"test_fixed40.pt", prescores["test"]["test_fixed40.pt"])
    original_fixed = checked_bank(torch.load(fixed_path, weights_only=True, map_location="cpu"), ids)
    require(all(torch.equal(fixed[s], original_fixed[s]) for s in ids), "fixed-mask candidate reconstruction differs")
    inputs = {"preregistration": input_record(args.preregistration), "v4_protocol": input_record(v4/"protocol.json"),
              "admission": input_record(v4/"archive_admission.json"), "dev_prescore": input_record(prereg["frozen_inputs"]["dev_prescore"]["path"]),
              "test_prescore": input_record(prereg["frozen_inputs"]["test_prescore"]["path"]), "whole_provenance": input_record(wp),
              "test_local": input_record(local_path), "test_whole": input_record(whole_path), "test_fixed40": input_record(fixed_path),
              "dev_manifest": input_record(dev_input["path"]), "train_manifest": input_record(training_input["path"]),
              "native_metric": input_record(ROOT/"external/baselines/MSKA/metrics.py"),
              "full_adapter": input_record(adapter_path)}
    closure = dev_local["validated_annotation_lineage"]["pose_only_contract"]["local_recognition_import_closure"]
    inputs["raw_label_archive"] = raw_label_archive
    inputs["test_evaluator"] = closure["entrypoint"]
    inputs["mska_gloss_vocabulary"] = closure["live_gloss_vocabulary"]
    inputs.update({f"mska_code_{relative}": record for relative, record in closure["files"].items()})
    for label in ("dev_local_result", "dev_whole_result", "dev_local_provenance", "dev_whole_provenance"):
        inputs[label] = input_record(getattr(args, label))
    out = new_directory(args.out_dir)
    # Retain the exact code bytes as well as their hash, avoiding historical
    # script drift; the snapshot is restored to its original scripts path to run.
    (out/"implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(out/"model_parameters.json", model_data)
    write_json(out/"dev_fit_audit.json", fit_audit)
    write_json(out/"dev_annotation_lineage.json", {"local": dev_local["validated_annotation_lineage"],
                                                "whole": dev_whole["validated_annotation_lineage"]})
    write_json(out/"recognition_import_closure.json", closure)
    write_json(out/"raw_label_lineage_commitment.json", {
        "schema": "reduced_csl_raw_label_lineage_commitment_v1", "archive": raw_label_archive,
        "dev_manifest": inputs["dev_manifest"], "archive_sha256_verified": True,
        "archive_deserialized_during_freeze": False, "semantic_recheck_during_freeze": False,
        "purpose": "bind the raw archive used for the separate dev-lineage diagnostic without loading mixed-split annotations into routing"})
    write_json(out/"test_scores.json", {sid: {"similarity": features[i][0], "predicted_value": float(scores[i]),
                                            "local_frames": len(local[sid]), "whole_frames": len(whole_bank[sid])} for i, sid in enumerate(ids)})
    write_json(out/"test_mask.json", mask)
    write_json(out/"test_fixed_mask.json", fixed_mask)
    write_json(out/"test_ledger.json", ledger)
    torch.save(poses, out/"test_learned40.pt")
    subset_ids = ids[:20]
    require(len(subset_ids) == 20, "first20 audit requires at least twenty test requests")
    write_json(out/"first20_ids.json", subset_ids)
    for route, bank in (("fixed40", fixed), ("learned40", poses)):
        torch.save({sid: bank[sid] for sid in subset_ids}, out/f"test_first20_{route}.pt")
    allocation.update(schema="reduced_csl_allocation_v1", policy_route=POLICY_ROUTE, scope=SCOPE,
                      ids=ids, preregistration_sha256=digest(prereg_path), inputs=inputs,
                      model_sha256=digest(out/"model_parameters.json"), scores_sha256=digest(out/"test_scores.json"),
                      mask_sha256=digest(out/"test_mask.json"), ledger_sha256=digest(out/"test_ledger.json"),
                      pose_sha256=digest(out/"test_learned40.pt"), test_annotation_files_read=False,
                      test_recognizer_results_read=False, source_certificate_required_before_test_scoring=True)
    write_json(out/"allocation.json", allocation)
    output_hashes = {p.name: digest(p) for p in sorted(out.iterdir())}
    write_json(out/"test_prescore_hashes.json", output_hashes)
    output_hashes["test_prescore_hashes.json"] = digest(out/"test_prescore_hashes.json")
    report = {"schema": "reduced_csl_freeze_v1", "scope": SCOPE, "outputs": output_hashes,
              "inputs": inputs, "frozen_unix_ns": time.time_ns(), "test_ids": ids,
              "test_annotations_or_scores_read": False, "certification_pending": True,
              "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version()},
              "implementation_sha256": digest(__file__)}
    write_json(out/"freeze.json", report)
    return report


def frozen_release(directory):
    directory = confined(directory)
    frozen = read_json(directory/"freeze.json")
    require(frozen.get("schema") == "reduced_csl_freeze_v1", "unsupported frozen release")
    for filename, expected in frozen["outputs"].items():
        require(Path(filename).name == filename, "invalid frozen output name")
        committed_file(directory/filename, expected)
    for label, record in frozen["inputs"].items():
        if label == "test_evaluator":
            resolve_test_evaluator_record(record)
        else:
            committed_file(record["path"], record["sha256"])
    prereg = read_prereg(frozen["inputs"]["preregistration"]["path"])
    for local, registered in (("v4_protocol", "v4_protocol"), ("test_prescore", "test_prescore")):
        require(frozen["inputs"][local]["sha256"] == prereg["frozen_inputs"][registered]["sha256"], "frozen source contract differs from preregistration")
    return directory, frozen


def certificate_context(frozen):
    """Load only the committed training archive once for both release audits."""
    prereg = read_prereg(frozen["inputs"]["preregistration"]["path"])
    protocol_record = frozen["inputs"]["v4_protocol"]
    protocol = read_json(protocol_record["path"], prereg["frozen_inputs"]["v4_protocol"]["sha256"])
    require(protocol.get("version") == "CSL-text-native-fixed40-v4" and canonical(protocol.get("budget")) == b"[2,5]", "certificate source protocol differs")
    training = protocol["inputs"]["train_manifest"]
    require(training == frozen["inputs"]["train_manifest"], "frozen training registry differs from source protocol")
    training_rows = read_json(training["path"], training["sha256"])
    admitted = {row["id"] for row in training_rows}
    require(len(admitted) == len(training_rows) == prereg["splits"]["train"], "certificate training registry count differs")
    prescore_record = frozen["inputs"]["test_prescore"]
    prescore = read_json(prescore_record["path"], prereg["frozen_inputs"]["test_prescore"]["sha256"])
    trace = checked_trace(read_json(Path(protocol_record["path"]).parent/"test_trace.json", prescore["test_trace.json"]))
    require(frozen["test_ids"] == sorted(trace) and not set(trace) & admitted, "certificate release membership is not the frozen test split")
    raw = protocol["inputs"]["pose_archive"]
    archive_path = committed_file(raw["path"], raw["sha256"])
    with archive_path.open("rb") as handle:
        archive = pickle.load(handle)
    require(isinstance(archive, dict), "raw archive must be an ID mapping")
    admission = read_json(frozen["inputs"]["admission"]["path"], frozen["inputs"]["admission"]["sha256"])
    require(admission.get("excluded_pose_sources") == sorted(set(archive)-admitted) and
            type(admission.get("admitted_pose_sources")) is int and admission["admitted_pose_sources"] == len(admitted) and
            admission.get("train_sources_missing_poses") == sorted(admitted-set(archive)) == [], "archive admission differs from committed training registry")
    return {"archive": archive, "admitted": admitted, "archive_digest": raw["sha256"],
            "training_digest": training["sha256"], "protocol": protocol}


def check_certificate(manifest_path, report_path, frozen, directory, *, learned, key, context=None):
    import torch
    from scripts import csl_text_route_certificate as cert
    manifest_path, report_path = confined(manifest_path), confined(report_path)
    manifest, report = read_json(manifest_path), read_json(report_path)
    require(report.get("verdict") == "PASS" and report.get("manifest_sha256") == digest(manifest_path), "certificate verification is absent or inconsistent")
    require(report.get("implementation_sha256") == digest(cert.__file__) == NATIVE_VERIFIER_SHA and
            report.get("helper_sha256") == digest(cert.ROOT/"budget_sage/audit/certificate.py") == NATIVE_HELPER_SHA, "verification code hash differs from accepted verifier")
    expected_pose = frozen["outputs"]["test_learned40.pt"] if learned else frozen["inputs"]["test_fixed40"]["sha256"]
    require(report.get("pose_bank_sha256") == manifest.get("pose_bank_sha256") == expected_pose, "certificate belongs to another pose bank")
    expected_route = POLICY_ROUTE if learned else cert.POLICY["route"]
    require(manifest["policy"]["route"] == expected_route, "certificate route differs")
    commitments = {"protocol_sha256": frozen["inputs"]["v4_protocol"]["sha256"],
                   "archive_admission_sha256": frozen["inputs"]["admission"]["sha256"],
                   "prescore_sha256": frozen["outputs"]["test_prescore_hashes.json"] if learned else frozen["inputs"]["test_prescore"]["sha256"],
                   "allocation_protocol_sha256": frozen["outputs"]["allocation.json"] if learned else frozen["inputs"]["v4_protocol"]["sha256"]}
    require(canonical(manifest.get("construction_commitments")) == canonical(report.get("construction_commitments")) == canonical(commitments), "certificate construction commitments differ")
    prescore_path = directory/"test_prescore_hashes.json" if learned else Path(frozen["inputs"]["test_prescore"]["path"])
    prescore = read_json(prescore_path, commitments["prescore_sha256"])
    pose_path = directory/"test_learned40.pt" if learned else Path(frozen["inputs"]["test_fixed40"]["path"])
    require(prescore.get(pose_path.name) == expected_pose, "certificate pose is absent from frozen prescore records")
    committed_file(pose_path, expected_pose)
    if learned:
        allocation = read_json(directory/"allocation.json", commitments["allocation_protocol_sha256"])
        require(allocation.get("policy_route") == POLICY_ROUTE and allocation.get("preregistration_sha256") == PREREG_SHA and
                allocation.get("pose_sha256") == expected_pose and allocation.get("mask_sha256") == frozen["outputs"]["test_mask.json"] and
                allocation.get("model_sha256") == frozen["outputs"]["model_parameters.json"], "allocation declaration differs from frozen route")
        require(manifest_path.stat().st_mtime_ns >= frozen["frozen_unix_ns"], "learned certificate predates route freeze")
    context = context or certificate_context(frozen)
    poses = torch.load(pose_path, weights_only=True, map_location="cpu")
    require(set(poses) == set(frozen["test_ids"]), "certificate pose membership differs")
    ledger_path = directory/"test_ledger.json" if learned else prescore_path.parent/"test_ledger.json"
    ledger = read_json(ledger_path, prescore["test_ledger.json"])
    mask = read_json(directory/("test_mask.json" if learned else "test_fixed_mask.json"))
    require(set(ledger) == set(mask) == set(poses), "certificate recipe/mask membership differs")
    for sid in poses:
        require(type(mask[sid]) is bool and ledger[sid].get("whole_replay") is mask[sid], "certificate recipe differs from frozen allocation")
        require(canonical(manifest["certificates"][sid]["ledger"]) == canonical(cert.converted_ledger(ledger[sid])), "certificate source recipe differs from frozen release")
    try:
        recomputed = cert.verify(poses, manifest, context["archive"], context["admitted"],
            context["archive_digest"], context["training_digest"], expected_pose, key,
            construction_commitments=commitments)
    except cert.CertificateError as exc:
        raise ProtocolError(f"independent reconstruction verification failed: {exc}") from exc
    metadata_fields = {"implementation_sha256", "helper_sha256", "manifest_sha256", "construction_commitments", "peak_rss_kb"}
    require(set(report) == set(recomputed) | metadata_fields, "verifier report has unsupported or missing fields")
    require(canonical({name: report[name] for name in recomputed}) == canonical(recomputed), "verifier scientific report differs from independent reconstruction")
    exact_int(report["peak_rss_kb"])
    fresh = {"schema": "reduced_csl_seal_reconstruction_v1", "recomputed": recomputed,
             "construction_commitments": commitments, "verifier_sha256": NATIVE_VERIFIER_SHA,
             "helper_sha256": NATIVE_HELPER_SHA, "consumed_report_sha256": digest(report_path),
             "manifest_sha256": digest(manifest_path), "frozen_recipe_binding_checked": True}
    return {"manifest": input_record(manifest_path), "verification_report": input_record(report_path)}, fresh


def seal(args):
    import os
    directory, frozen = frozen_release(args.frozen_dir)
    key = os.environ.get(args.key_env, "").encode()
    require(key, "cooperative certificate key required")
    context = certificate_context(frozen)
    learned, learned_audit = check_certificate(args.manifest, args.verification_report, frozen, directory, learned=True, key=key, context=context)
    fixed, fixed_audit = check_certificate(args.fixed_manifest, args.fixed_verification_report, frozen, directory, learned=False, key=key, context=context)
    for branch, records, fresh in (("learned", learned, learned_audit), ("fixed", fixed, fixed_audit)):
        path = directory/f"seal_{branch}_reconstruction.json"
        write_json(path, fresh)
        records["independent_reconstruction"] = input_record(path)
    value = {"schema": "reduced_csl_test_scoring_seal_v1", "freeze_sha256": digest(directory/"freeze.json"),
             "created_unix_ns": time.time_ns(), "learned_certificate": learned, "fixed_certificate": fixed,
             "local_cooperative_ordering_only": True, "scope": SCOPE}
    write_json(directory/"test_scoring_seal.json", value)
    return value


def sealed_release(directory):
    directory, frozen = frozen_release(directory)
    seal = read_json(directory/"test_scoring_seal.json")
    require(seal.get("schema") == "reduced_csl_test_scoring_seal_v1" and
            seal.get("freeze_sha256") == digest(directory/"freeze.json"), "missing or changed test-scoring seal")
    require(exact_int(seal.get("created_unix_ns"), minimum=1) > frozen["frozen_unix_ns"], "invalid seal ordering")
    for branch in ("learned_certificate", "fixed_certificate"):
        for record in seal[branch].values():
            committed_file(record["path"], record["sha256"])
    return directory, frozen, seal


def annotation_rows(protocol, split):
    record = protocol["inputs"][f"{split}_manifest"]
    rows = read_json(record["path"], record["sha256"])
    result = {r["id"]: {"ref": r["gloss"], "text": r["text"]} for r in rows}
    require(len(result) == len(rows), "duplicate annotation IDs")
    return result


def prepare_subsets(args):
    """Attach held-out metadata only after the full route's certificate seal."""
    import torch
    directory, frozen, seal_record = sealed_release(args.frozen_dir)
    require(digest(__file__) == frozen["outputs"]["implementation_snapshot.py"], "subset adapter implementation differs from frozen snapshot")
    pose_only_contract = pose_only_metadata_contract(read_prereg(frozen["inputs"]["preregistration"]["path"]))
    protocol = read_json(frozen["inputs"]["v4_protocol"]["path"])
    annotations = annotation_rows(protocol, "test")
    ids = read_json(directory/"first20_ids.json")
    require(ids == sorted(frozen["test_ids"])[:20] and len(ids) == 20, "first20 membership differs")
    require(set(annotations) == set(frozen["test_ids"]), "test annotation membership differs")
    official_path = confined(args.manifest)
    require(official_path == confined(ROOT/"external/baselines/MSKA/data/CSL-Daily/CSL-Daily.test"), "subset metadata must use canonical test archive")
    official, annotation_audit = canonical_annotation_lineage(official_path, frozen["test_ids"],
        {sid: row["ref"] for sid, row in annotations.items()}, {sid: row["text"] for sid, row in annotations.items()})
    out = new_directory(args.out_dir)
    result = {}
    for route in ("fixed40", "learned40"):
        bank_path = directory/f"test_first20_{route}.pt"
        bank = checked_bank(torch.load(bank_path, weights_only=True, map_location="cpu"), ids)
        rows = {}
        for sid in ids:
            rows[sid] = {"name": sid, "text": official[sid]["text"], "gloss": official[sid]["gloss"],
                         "keypoint": bank[sid], "num_frames": len(bank[sid])}
        output = out/f"CSL-Daily.test_first20_{route}"
        with output.open("xb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        provenance = {"split": "test", "route": route, "n": len(rows), "subset_ids": ids,
                      "source": str(bank_path), "source_sha256": digest(bank_path),
                      "prescore_hashes_sha256": frozen["outputs"]["test_prescore_hashes.json"],
                      "manifest": str(official_path), "manifest_sha256": digest(official_path),
                      "output": str(output), "output_sha256": digest(output),
                      "full_release_sha256": frozen["outputs"]["test_learned40.pt"] if route == "learned40" else frozen["inputs"]["test_fixed40"]["sha256"],
                      "seal_sha256": digest(directory/"test_scoring_seal.json"), "prepared_unix_ns": time.time_ns(),
                      "implementation_sha256": digest(__file__),
                      "annotation_lineage": {**annotation_audit, "pose_only_contract": pose_only_contract}}
        write_json(out/f"CSL-Daily.test_first20_{route}.provenance.json", provenance)
        result[route] = provenance
    return result


def compose(local, whole, mask):
    require(set(local) == set(whole) == set(mask), "composition membership differs")
    result = {}
    for sid in mask:
        require(type(mask[sid]) is bool, "mask must contain exact Booleans")
        require(local[sid]["ref"] == whole[sid]["ref"], "branch reference mismatch")
        require(set(local[sid]) == set(whole[sid]) == {"ref", *HEADS}, "branch head coverage differs")
        result[sid] = copy.deepcopy(whole[sid] if mask[sid] else local[sid])
    return result


def first20_identity(composed, directly_scored):
    expected = sorted(composed)[:20]
    require(len(expected) == 20 and set(directly_scored) == set(expected), "first20 exact membership required")
    mismatch = []
    for sid in expected:
        require(directly_scored[sid]["ref"] == composed[sid]["ref"], "first20 reference mismatch")
        for head in HEADS:
            if directly_scored[sid].get(head) != composed[sid][head]:
                mismatch.append({"id": sid, "head": head, "composed": composed[sid][head], "direct": directly_scored[sid].get(head)})
    return {"passed": not mismatch, "ids": expected, "heads": list(HEADS), "mismatches": mismatch}


def group_bootstrap(fixed, learned, captions, *, replicates=10000, seed=30373):
    require(set(fixed) == set(learned) == set(captions), "bootstrap memberships differ")
    exact_int(replicates, minimum=1); exact_int(seed)
    groups = {}
    for sid in sorted(fixed):
        require(fixed[sid]["ref"] == learned[sid]["ref"], "bootstrap reference mismatch")
        group = normalize_caption(captions[sid])
        require(group, "empty normalized caption group")
        groups.setdefault(group, []).append(sid)
    statistics = []
    for group in sorted(groups):
        a = b = n = 0
        for sid in groups[group]:
            ea, eb = edits(fixed[sid]["ref"], fixed[sid][HEADS[0]]), edits(learned[sid]["ref"], learned[sid][HEADS[0]])
            a += ea["num_err"]; b += eb["num_err"]; n += ea["num_ref"]
        statistics.append((a, b, n))
    values = np.asarray(statistics, dtype=np.int64)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        summed = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        differences[i] = 100*(int(summed[1])-int(summed[0]))/int(summed[2])
    total = values.sum(axis=0)
    point = 100*(int(total[1])-int(total[0]))/int(total[2])
    low, high = np.percentile(differences, [2.5, 97.5], method="linear").tolist()
    return {"replicates": replicates, "seed": seed, "groups": len(groups), "requests": len(fixed),
            "sampling_unit": "normalized public-caption group; include all group members",
            "difference_learned_minus_fixed_wer": point, "ci95": [low, high], "percentile_method": "linear",
            "learned_benefit_confirmed_by_preregistered_falsifier": bool(point < 0 and high < 0),
            "replicate_sha256": hashlib.sha256(differences.astype("<f8").tobytes()).hexdigest(),
            "group_membership_sha256": hashlib.sha256(canonical(groups)).hexdigest()}


def analyze(args):
    directory, frozen, seal_record = sealed_release(args.frozen_dir)
    prereg = read_prereg(frozen["inputs"]["preregistration"]["path"])
    protocol = read_json(frozen["inputs"]["v4_protocol"]["path"])
    annotations = annotation_rows(protocol, "test")
    ids = frozen["test_ids"]
    require(set(annotations) == set(ids), "analysis test annotation membership differs")
    refs = {sid: annotations[sid]["ref"] for sid in ids}
    texts = {sid: annotations[sid]["text"] for sid in ids}
    adapter_path = frozen["inputs"]["full_adapter"]["path"]
    require(frozen["inputs"]["full_adapter"]["sha256"] == FULL_ADAPTER_SHA, "full adapter differs from accepted frozen implementation")
    after = seal_record["created_unix_ns"]
    rows, evidence, evaluator_contract, annotation_lineages = {}, {}, None, {}
    evidence["test_evaluator_relocation"] = resolve_test_evaluator_record(
        frozen["inputs"]["test_evaluator"]
    )
    def check_evaluator(result):
        nonlocal evaluator_contract
        contract = {key: result[key] for key in ("implementation_sha256", "checkpoint_sha256", "config_sha256", "checkpoint_state_key_count")}
        if evaluator_contract is None:
            evaluator_contract = contract
        require(contract == evaluator_contract, "test evaluator implementation/weights/configuration differ across banks")
    for route in ("local", "whole"):
        pose_key = f"test_{route}"
        prescore_record = frozen["inputs"]["test_prescore" if route == "local" else "whole_provenance"]
        result_path, provenance_path = getattr(args, f"test_{route}_result"), getattr(args, f"test_{route}_provenance")
        evaluated, rows[route] = load_evaluation(result_path, provenance_path, split="test", route=route, ids=ids,
            source_path=frozen["inputs"][pose_key]["path"], pose_digest=frozen["inputs"][pose_key]["sha256"],
            prescore_path=prescore_record["path"], prescore_digest=prescore_record["sha256"],
            adapter_path=adapter_path, adapter_digest=FULL_ADAPTER_SHA,
            prereg=prereg, expected_refs=refs, expected_texts=texts, after_ns=after)
        check_evaluator(evaluated)
        annotation_lineages[route] = evaluated["validated_annotation_lineage"]
        evidence[f"{route}_result"] = input_record(result_path)
        evidence[f"{route}_provenance"] = input_record(provenance_path)
    route_rows, identities = {}, {}
    for route, filename in (("fixed40", "test_fixed_mask.json"), ("learned40", "test_mask.json")):
        route_rows[route] = compose(rows["local"], rows["whole"], read_json(directory/filename))
        prefix = "fixed" if route == "fixed40" else "learned"
        result_path, provenance_path = getattr(args, f"{prefix}_subset_result"), getattr(args, f"{prefix}_subset_provenance")
        evaluated, direct = load_evaluation(result_path, provenance_path, split="test", route=route, ids=sorted(ids)[:20],
            source_path=directory/f"test_first20_{route}.pt", pose_digest=frozen["outputs"][f"test_first20_{route}.pt"],
            prescore_path=directory/"test_prescore_hashes.json", prescore_digest=frozen["outputs"]["test_prescore_hashes.json"],
            adapter_path=directory/"implementation_snapshot.py", adapter_digest=frozen["outputs"]["implementation_snapshot.py"],
            prereg=prereg, expected_refs=refs, expected_texts=texts, after_ns=after)
        check_evaluator(evaluated)
        annotation_lineages[f"{route}_first20"] = evaluated["validated_annotation_lineage"]
        identities[route] = first20_identity(route_rows[route], direct)
        evidence[f"{prefix}_subset_result"] = input_record(result_path)
        evidence[f"{prefix}_subset_provenance"] = input_record(provenance_path)
    composition_pass = all(v["passed"] for v in identities.values())
    out = new_directory(args.out_dir)
    write_json(out/"first20_identity.json", identities)
    if not composition_pass:
        write_json(out/"composition_failure.json", {"status": "RETAINED_FAILURE", "requires": "direct full evaluation of both frozen routed banks", "identity": identities})
        require(all(getattr(args, f"{prefix}_full_{field}") for prefix in ("fixed", "learned") for field in ("result", "provenance")), "first20 identity failed; failure retained; directly evaluate both full frozen routes")
        for prefix, route in (("fixed", "fixed40"), ("learned", "learned40")):
            result_path, provenance_path = getattr(args, f"{prefix}_full_result"), getattr(args, f"{prefix}_full_provenance")
            evaluated, route_rows[route] = load_evaluation(result_path, provenance_path, split="test", route=route, ids=ids,
                source_path=frozen["inputs"]["test_fixed40"]["path"] if prefix == "fixed" else directory/"test_learned40.pt",
                pose_digest=frozen["inputs"]["test_fixed40"]["sha256"] if prefix == "fixed" else frozen["outputs"]["test_learned40.pt"],
                prescore_path=frozen["inputs"]["test_prescore"]["path"] if prefix == "fixed" else directory/"test_prescore_hashes.json",
                prescore_digest=frozen["inputs"]["test_prescore"]["sha256"] if prefix == "fixed" else frozen["outputs"]["test_prescore_hashes.json"],
                adapter_path=adapter_path, adapter_digest=FULL_ADAPTER_SHA,
                prereg=prereg, expected_refs=refs, expected_texts=texts, after_ns=after)
            check_evaluator(evaluated)
            annotation_lineages[f"{route}_full"] = evaluated["validated_annotation_lineage"]
            evidence[f"{prefix}_full_result"] = input_record(result_path)
            evidence[f"{prefix}_full_provenance"] = input_record(provenance_path)
    all_rows = {**rows, **route_rows}
    bootstrap = group_bootstrap(route_rows["fixed40"], route_rows["learned40"], {sid: annotations[sid]["text"] for sid in ids})
    scores = read_json(directory/"test_scores.json")
    lengths = {"local": sum(scores[s]["local_frames"] for s in ids), "whole": sum(scores[s]["whole_frames"] for s in ids)}
    for route, filename in (("fixed40", "test_fixed_mask.json"), ("learned40", "test_mask.json")):
        mask = read_json(directory/filename)
        lengths[route] = sum(scores[s]["whole_frames" if mask[s] else "local_frames"] for s in ids)
    for route, predictions in all_rows.items():
        write_json(out/f"{route}_predictions.json", predictions)
    report = {"schema": "reduced_csl_analysis_v1", "scope": SCOPE, "n": len(ids),
              "coverage": {"emitted": len(ids), "refused": 0, "missing_predictions": 0},
              "composition_identity_pass": composition_pass, "used_direct_full_fallback": not composition_pass,
              "first20_identity": identities, "all_head_metrics": {route: summarize(p) for route, p in all_rows.items()},
              "emitted_frames": lengths, "primary_paired_bootstrap": bootstrap, "evidence": evidence,
              "freeze_sha256": digest(directory/"freeze.json"), "seal_sha256": digest(directory/"test_scoring_seal.json"),
              "native_metric_sha256": digest(ROOT/"external/baselines/MSKA/metrics.py"),
              "evaluator_contract": evaluator_contract,
              "adapter_annotation_lineages": annotation_lineages,
              "implementation_sha256": digest(__file__), "completed_unix_ns": time.time_ns()}
    write_json(out/"analysis.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    f = stages.add_parser("freeze", help="dev-only fit and test route freeze; no test annotations or scores")
    f.add_argument("--preregistration", default=str(PREREG))
    f.add_argument("--whole-provenance", default=str(OUTPUT_ROOT/"csl_whole_candidates/provenance.json"))
    for branch in ("local", "whole"):
        f.add_argument(f"--dev-{branch}-result", required=True)
        f.add_argument(f"--dev-{branch}-provenance", required=True)
    f.add_argument("--out-dir", required=True)
    s = stages.add_parser("seal", help="authorize test scoring after native full-release certificate verification")
    s.add_argument("--frozen-dir", required=True)
    for name in ("manifest", "verification-report", "fixed-manifest", "fixed-verification-report"):
        s.add_argument(f"--{name}", required=True)
    s.add_argument("--key-env", default="BUDGET_SAGE_MANIFEST_KEY")
    sub = stages.add_parser("prepare-subsets", help="attach official metadata to precisely the frozen first20 outputs")
    sub.add_argument("--frozen-dir", required=True)
    sub.add_argument("--manifest", default=str(ROOT/"external/baselines/MSKA/data/CSL-Daily/CSL-Daily.test"))
    sub.add_argument("--out-dir", required=True)
    a = stages.add_parser("analyze", help="post-seal all-head evaluation; retain null and failed comparisons")
    a.add_argument("--frozen-dir", required=True)
    a.add_argument("--out-dir", required=True)
    for branch in ("local", "whole"):
        for field in ("result", "provenance"):
            a.add_argument(f"--test-{branch}-{field}", required=True)
    for branch in ("fixed", "learned"):
        for field in ("result", "provenance"):
            a.add_argument(f"--{branch}-subset-{field}", required=True)
            a.add_argument(f"--{branch}-full-{field}")
    return p


def main():
    args = parser().parse_args()
    result = {"freeze": freeze, "seal": seal, "prepare-subsets": prepare_subsets, "analyze": analyze}[args.stage](args)
    print(json.dumps({"stage": args.stage, "schema": result.get("schema"), "scope": SCOPE}, sort_keys=True))


if __name__ == "__main__":
    main()
