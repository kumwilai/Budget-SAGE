"""Versioned text-only CSL native source-routing feasibility, not full-method efficacy."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import resource
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_retrieval_gloss_plans import project_queries, project_training, sha256
from scripts.build_csl_pg_rast_native_hrnet import resample
from scripts.clean_frame_budget_baseline import _ColumnScore
from scripts.strict_route_only_cac import FEATURE_NAMES, select_replay_frames as admit_frames

PLANNER = ROOT / "outputs/csl_extension/scripts/build_retrieval_gloss_plans_csl.py"
DEFAULT_OUT = ROOT / "outputs/revision/astra_open_closure_20260906/csl_text_transfer_v4"


def planner(queries, training):
    # Existing port caches caption normalization; 'both' fixes .6/.4 weights.
    spec = importlib.util.spec_from_file_location("csl_fixed_planner", PLANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_plans(project_queries(queries), training, "both")


def array(value):
    if isinstance(value, dict):
        value = value["keypoint"]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (133, 3) or len(result) < 1:
        raise ValueError("source must be non-empty native [T,133,3]")
    return result


def validate_source_split(training, spans, pose, heldout_ids):
    allowed = {str(row["id"]) for row in training}
    for name, ids in (("pose", set(pose)), ("span", set(spans))):
        if ids - allowed:
            raise ValueError(f"{name} source outside train manifest: {sorted(ids-allowed)[:3]}")
        if ids & heldout_ids:
            raise ValueError(f"{name} source overlaps held-out IDs")


def build_bank(pose, spans, max_per_gloss=64):
    """Same capped units/highconf_len formula as the native diagnostic."""
    bank, skipped = {}, Counter()
    for sid, units in spans.items():
        if sid not in pose:
            skipped["source_missing"] += len(units)
            continue
        source = array(pose[sid])
        if not np.isfinite(source).all():
            raise ValueError(f"nonfinite training pose: {sid}")
        for unit in units:
            start, end = max(0, int(unit["start"])), min(len(source), int(unit["end"]))
            if end - start < 4:
                skipped["span_shorter_than_four"] += 1
                continue
            if start == 0 and end == len(source):
                skipped["full_clip_span_not_local"] += 1
                continue
            bank.setdefault(str(unit["gloss"]), []).append((sid, start, end))
    rng = np.random.default_rng(0)
    lengths = {}
    for gloss, units in bank.items():
        if len(units) > max_per_gloss:
            units = [units[int(i)] for i in rng.choice(len(units), max_per_gloss, replace=False)]
        median = float(np.median([end-start for _, start, end in units]))
        lengths[gloss] = max(4, int(round(max(4, int(round(median))) / 4)) * 4)
        scored = []
        for sid, start, end in units:
            segment = array(pose[sid])[start:end]
            quality = float(.75 * segment[:, 91:133, 2].mean()
                            + .25 * segment[:, :13, 2].mean()
                            - .10 * abs(math.log((end-start) / max(1., median))))
            scored.append((quality, sid, start, end))
        bank[gloss] = sorted(scored, reverse=True)
    return bank, lengths, dict(skipped)


def reconstruct(recipe, pose, forbidden=()):
    pieces = []
    forbidden = set(forbidden)
    for unit in recipe:
        sid, start, end, length = (unit[k] for k in ("source_id", "start", "end", "out_len"))
        if sid in forbidden:
            raise ValueError("forbidden source interval")
        if sid not in pose:
            raise ValueError("source absent from training archive")
        source = array(pose[sid])
        if any(type(x) is not int for x in (start, end, length)) or not 0 <= start < end <= len(source) or length < 1:
            raise ValueError("invalid source interval or output length")
        pieces.append(resample(source[start:end], length))
    if not pieces:
        raise ValueError("empty source recipe")
    return np.concatenate(pieces).astype(np.float32)


def assemble_local(glosses, bank, lengths, pose, forbidden):
    recipe, omitted = [], []
    for index, gloss in enumerate(glosses):
        selected = next((u for u in bank.get(gloss, []) if u[1] not in forbidden), None)
        if selected is None:
            omitted.append({"index": index, "gloss": gloss})
            continue
        _, sid, start, end = selected
        recipe.append({"source_id": sid, "start": start, "end": end,
                       "out_len": lengths[gloss], "gloss": gloss, "plan_index": index})
    if not recipe:
        return None, [], omitted
    return reconstruct(recipe, pose, forbidden), recipe, omitted


def select_replay_frames(candidates, budget="0.40"):
    ids = [c["id"] for c in candidates]
    features = np.zeros((len(ids), len(FEATURE_NAMES)))
    features[:, 0] = [c["similarity"] for c in candidates]
    replay = np.asarray([c["whole_len"] for c in candidates], dtype=np.int64)
    local = np.asarray([c["local_len"] for c in candidates], dtype=np.int64)
    mask, info = admit_frames(_ColumnScore(0), ids, features, np.ones(len(ids), bool), replay, local, budget)
    return mask.astype(bool).tolist(), info


def generate(queries, training, pose, spans, heldout_ids):
    queries = project_queries(queries)
    training = project_training(training)
    validate_source_split(training, spans, pose, heldout_ids)
    plans, trace = planner(queries, training)
    bank, lengths, skipped = build_bank(pose, spans)
    pending, refused = [], []
    for row, tr in zip(queries, trace):
        sid = row["id"]
        forbidden = set(tr["forbidden_source_ids"])
        local, recipe, omitted = assemble_local(plans[sid], bank, lengths, pose, forbidden)
        if local is None:
            refused.append({"id": sid, "reason": "no_permitted_local_unit", "omitted": omitted})
            continue
        donor = tr["retrieved_id"]
        if donor in tr["exact_caption_source_ids"] or donor not in pose:
            raise ValueError("invalid whole-clip donor")
        pending.append({"id": sid, "local": local, "recipe": recipe,
                        "omitted": omitted, "donor": donor, "forbidden": forbidden,
                        "similarity": tr["similarity"], "local_len": len(local),
                        "whole_len": len(array(pose[donor])), "plan_len": len(plans[sid])})
    if not pending:
        raise ValueError("all requests refused; no valid release")
    mask, budget = select_replay_frames(pending)
    local_bank, route_bank, ledger, mass = {}, {}, {}, Counter()
    exact = 0
    for item, replay in zip(pending, mask):
        sid = item["id"]
        local = item["local"]
        if not np.array_equal(local, reconstruct(item["recipe"], pose, item["forbidden"])):
            raise AssertionError("local tensor reconstruction failed")
        recipe = ([{"source_id": item["donor"], "start": 0, "end": item["whole_len"],
                    "out_len": item["whole_len"]}] if replay else item["recipe"])
        selected = array(pose[item["donor"]]) if replay else local
        if not np.array_equal(selected, reconstruct(recipe, pose)):
            raise AssertionError("selected tensor reconstruction failed")
        exact += 1
        local_bank[sid] = torch.from_numpy(local)
        route_bank[sid] = torch.from_numpy(selected)
        source_mass = Counter()
        for unit in recipe:
            source_mass[unit["source_id"]] += unit["out_len"]
        if sum(source_mass.values()) != len(selected):
            raise AssertionError("source mass does not conserve output frames")
        mass.update(source_mass)
        ledger[sid] = {"whole_replay": replay, "recipe": recipe, "local_recipe": item["recipe"],
                       "omitted_local_plan_tokens": item["omitted"],
                       "source_mass": dict(source_mass), "frames": len(selected),
                       "local_forbidden_sources": sorted(item["forbidden"])}
    emitted_frames = sum(len(v) for v in route_bank.values())
    replay_frames = sum(len(route_bank[item["id"]]) for item, flag in zip(pending, mask) if flag)
    if 5*replay_frames > 2*emitted_frames or emitted_frames != sum(mass.values()):
        raise AssertionError("exact replay budget or mass check failed")
    summary = {"requests": len(queries), "emitted": len(pending), "refused": refused,
               "budget": budget, "source_mass_frames": sum(mass.values()),
               "unique_sources": len(mass), "exact_local_reconstructions": exact,
               "exact_selected_reconstructions": exact, "forbidden_local_frames": 0,
               "omitted_plan_tokens": sum(len(p["omitted"]) for p in pending) + sum(len(r["omitted"]) for r in refused),
               "planned_tokens": sum(map(len, plans.values())), "bank_glosses": len(bank),
               "bank_skipped_units": skipped, "local_total_frames": sum(len(v) for v in local_bank.values()),
               "mean_route_frames": emitted_frames/len(pending),
               "mean_local_frames": sum(len(v) for v in local_bank.values())/len(pending),
               "integer_budget": f"{5*replay_frames} <= {2*emitted_frames}"}
    return plans, trace, local_bank, route_bank, ledger, mask, summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def score_plans(rows, plans):
    from jiwer import process_words
    score = process_words([str(r["gloss"]) for r in rows], [" ".join(plans[r["id"]]) for r in rows])
    return {"metric": "donor-gloss-plan WER; NOT pose recognizer WER", "n": len(rows),
            "wer_percent": 100*score.wer, "substitutions": score.substitutions,
            "deletions": score.deletions, "insertions": score.insertions,
            "reference_tokens": sum(map(len, score.references)),
            "hypothesis_tokens": sum(map(len, score.hypotheses))}


def confined_output_path(value):
    candidate = Path(value)
    out = (candidate if candidate.is_absolute() else ROOT/candidate).resolve()
    if out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        raise ValueError("output path must be a directory inside the project root")
    return out


def run(args):
    out = confined_output_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("use a new output directory; a protocol is already frozen here")
    manifest = ROOT / "data/csl-daily"
    source_paths = {"train_manifest": manifest/"csl_daily_train.json", "spans": Path(args.spans),
                    "pose_archive": Path(args.pose_archive), "implementation": Path(__file__),
                    "planner": PLANNER, "allocator": ROOT/"scripts/strict_route_only_cac.py",
                    "resampling": ROOT/"scripts/build_csl_pg_rast_native_hrnet.py"}
    for split in ("dev", "test"):
        source_paths[f"{split}_manifest"] = manifest/f"csl_daily_{split}.json"
    protocol = {"version": "CSL-text-native-fixed40-v4", "feasibility_only": True,
                "revision_note": "v4 adds resolved output-root confinement; v3 scientific settings unchanged",
                "scope": "text planning/native gloss assembly/fixed frame admission; not PHOENIX semantic guards, phrase lattice, learned router, authentication, or sign efficacy",
                "query_fields": ["id", "text"], "weights_word_char": [.6, .4],
                "selection": "highconf_len", "max_per_gloss": 64, "seed": 0,
                "length": "train median, round to integer then nearest multiple of four",
                "joins": "plain concatenation", "budget": [2, 5], "ranking": "donor similarity; low-ID tie break",
                "missing_unit": "omit token; refuse if no local units; count every refusal and omission",
                "archive_admission": "raw pose and span source IDs intersect declared train-manifest IDs before generator validation",
                "local_unit_rule": "exclude intervals spanning an entire source clip; local candidates use strict subintervals",
                "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in source_paths.items()},
                "versions": {"numpy": np.__version__, "torch": torch.__version__}}
    write_json(out/"protocol.json", protocol)
    training = json.loads(source_paths["train_manifest"].read_text())
    queries = {s: project_queries(json.loads(source_paths[f"{s}_manifest"].read_text())) for s in ("dev", "test")}
    heldout = {r["id"] for rows in queries.values() for r in rows}
    with source_paths["pose_archive"].open("rb") as handle:
        pose = pickle.load(handle)
    spans = json.loads(source_paths["spans"].read_text())
    train_ids = {row["id"] for row in training}
    admission = {"raw_pose_sources": len(pose), "raw_span_sources": len(spans),
                 "excluded_pose_sources": sorted(set(pose)-train_ids),
                 "excluded_span_sources": sorted(set(spans)-train_ids),
                 "train_sources_missing_poses": sorted(train_ids-set(pose))}
    if admission["train_sources_missing_poses"]:
        raise ValueError("declared training donor has no pose; inspect archive admission")
    pose = {sid: value for sid, value in pose.items() if sid in train_ids}
    spans = {sid: value for sid, value in spans.items() if sid in train_ids}
    admission["admitted_pose_sources"] = len(pose)
    admission["admitted_span_sources"] = len(spans)
    write_json(out/"archive_admission.json", admission)
    results = {}
    for split in ("dev", "test"):
        plans, trace, local, route, ledger, mask, summary = generate(queries[split], training, pose, spans, heldout)
        torch.save(local, out/f"{split}_local.pt")
        torch.save(route, out/f"{split}_fixed40.pt")
        for name, value in (("plans", plans), ("trace", trace), ("ledger", ledger), ("mask", mask), ("structure", summary)):
            write_json(out/f"{split}_{name}.json", value)
        frozen = {p.name: sha256(p) for p in sorted(out.glob(f"{split}_*"))}
        write_json(out/f"{split}_prescore_hashes.json", frozen)
        # Query gloss labels are accessed only here, after pose/output hashes.
        summary["plan_gloss_wer"] = score_plans(json.loads(source_paths[f"{split}_manifest"].read_text()), plans)
        summary["prescore_hashes_sha256"] = sha256(out/f"{split}_prescore_hashes.json")
        results[split] = summary
        write_json(out/f"{split}_summary.json", summary)
        print(json.dumps({"split": split, **{k: summary[k] for k in ("requests", "emitted", "integer_budget", "plan_gloss_wer")}}, ensure_ascii=False), flush=True)
        del local, route, plans, trace, ledger
    results["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results["protocol_sha256"] = sha256(out/"protocol.json")
    write_json(out/"summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--spans", default=str(ROOT/"data/csl-daily/gloss_spans_ctc_train.json"))
    parser.add_argument("--pose_archive", default=str(ROOT/"external/baselines/MSKA/data/CSL-Daily/CSL-Daily.train"))
    run(parser.parse_args())
