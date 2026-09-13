"""Run exactly one validated CSL MSKA input, restoring the shared split safely."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
MSKA = ROOT / "external/baselines/MSKA"
DEFAULT_CONFIG = MSKA / "configs/csl-daily_s2g.yaml"
DEFAULT_CHECKPOINT = MSKA / "pretrained_models/CSL-Daily_SLR/best.pth"
MSKA_DATA = MSKA / "data/CSL-Daily"
REQUIRED_HEADS = {
    "body_hyp", "fuse_hyp", "left_hyp", "right_hyp", "ensemble_last_hyp",
}


def confined(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    q = p.resolve(strict=False)
    try:
        q.relative_to(ROOT.resolve())
    except ValueError as e:
        raise ValueError(f"path escapes project root: {path}") from e
    return q


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _check_hash(path: Path, expected: str | None, label: str) -> str:
    got = sha256(path)
    if expected and got != expected:
        raise ValueError(f"{label} SHA256 mismatch: {got} != {expected}")
    return got


@contextmanager
def cpu_cuda_compat(torch):
    """Temporarily neutralize MSKA's unconditional Tensor.cuda calls on CPU."""
    if torch.cuda.is_available():
        yield False
        return
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        yield True
    finally:
        torch.Tensor.cuda = original


def evaluate_ids(input_pickle: Path, config_path: Path, checkpoint: Path,
                 batch_size: int = 1, num_workers: int = 0,
                 beam_size: int = 5) -> dict[str, Any]:
    """Direct MSKA evaluation, returning ID-keyed refs/hyps and edit counts."""
    import yaml
    import torch
    sys.path.insert(0, str(MSKA))
    old = os.getcwd(); os.chdir(MSKA)
    cpu_compat = False
    try:
        from datasets import S2T_Dataset
        from Tokenizer import GlossTokenizer_S2G
        from recognition import Recognition
        from metrics import wer_single
        cfg = yaml.safe_load(config_path.read_text())
        cfg["data"]["dev_label_path"] = str(input_pickle)
        cfg["data"]["test_label_path"] = str(input_pickle)
        cfg["gloss"]["gloss2id_file"] = str(MSKA_DATA / "gloss2ids.pkl")
        cfg["model"]["RecognitionNetwork"]["GlossTokenizer"]["gloss2id_file"] = cfg["gloss"]["gloss2id_file"]
        class Args: rank = 0; distributed = False; gpu = 0; run = None
        tok = GlossTokenizer_S2G(cfg["gloss"])
        ds = S2T_Dataset(path=str(input_pickle), tokenizer=tok, config=cfg, args=Args(), phase="val", training_refurbish=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=ds.collate_fn, pin_memory=False, shuffle=False)
        model = Recognition(cfg=cfg["model"]["RecognitionNetwork"], args=Args()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k[len("recognition_network."):]: v for k, v in ck["model"].items() if k.startswith("recognition_network.")}
        expected = set(model.state_dict())
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise RuntimeError(f"strict recognition checkpoint mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
        model.load_state_dict(state, strict=True); model.eval()
        checkpoint_state_key_count = len(state)
        result: dict[str, dict[str, str]] = {}
        expected_heads: set[str] | None = None
        with cpu_cuda_compat(torch) as cpu_compat, torch.no_grad():
            for batch_index, src in enumerate(loader, start=1):
                out = model(src)
                batch_heads = {
                    key.replace("gloss_logits", "") + "hyp"
                    for key in out if "gloss_logits" in key
                }
                if expected_heads is None:
                    expected_heads = batch_heads
                elif batch_heads != expected_heads:
                    raise RuntimeError(
                        "model output heads changed across batches: "
                        f"expected={sorted(expected_heads)} got={sorted(batch_heads)}"
                    )
                if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                    print(
                        f"MSKA progress: {batch_index}/{len(loader)} batches",
                        flush=True,
                    )
                for key, logits in out.items():
                    if "gloss_logits" not in key: continue
                    head = key.replace("gloss_logits", "")
                    decoded = model.decode(gloss_logits=logits, beam_size=beam_size, input_lengths=out["input_lengths"])
                    for sid, hyp, ref in zip(src["name"], tok.convert_ids_to_tokens(decoded), src["gloss"]):
                        row = result.setdefault(str(sid), {"ref": str(ref)})
                        row[head + "hyp"] = " ".join(hyp)
        input_rows = pickle.load(input_pickle.open("rb"))
        expected_ids = {str(k) for k in input_rows}
        if set(result) != expected_ids:
            raise RuntimeError("evaluator did not return exactly the input IDs")
        if not expected_heads:
            raise RuntimeError("model returned no gloss-logit heads")
        missing_required = REQUIRED_HEADS - expected_heads
        if missing_required:
            raise RuntimeError(
                "model omitted preregistered heads: "
                f"missing={sorted(missing_required)}"
            )
        for sid, row in result.items():
            if str(row.get("ref", "")) != str(input_rows[sid if sid in input_rows else next(k for k in input_rows if str(k) == sid)]["gloss"]):
                raise RuntimeError(f"inconsistent reference for {sid}")
            missing_heads = expected_heads - set(row)
            if missing_heads:
                raise RuntimeError(
                    f"incomplete model-head coverage for {sid}: "
                    f"missing={sorted(missing_heads)}"
                )
        # Report every model head and use the fixed ensemble_last head as primary.
        counts = {}
        for sid, row in result.items():
            for k, hyp in row.items():
                if not k.endswith("hyp"): continue
                e = wer_single(row["ref"], hyp)
                counts.setdefault(k, {"num_err": 0, "num_del": 0, "num_ins": 0, "num_sub": 0, "num_ref": 0})
                for n in counts[k]: counts[k][n] += int(e[n])
        primary = "ensemble_last_hyp"
        if primary not in counts:
            raise RuntimeError("missing fixed ensemble_last head")
        for row in result.values(): row["ensemble_last_hyp"] = row[primary]
        counts["ensemble_last_hyp"] = counts[primary].copy()
        summary = {}
        for key, values in counts.items():
            num_hyp = sum(len(row[key].split()) for row in result.values())
            num_ref = values["num_ref"]
            summary[key] = {
                **values,
                "wer": 100.0 * values["num_err"] / num_ref if num_ref else 0.0,
                "num_hyp": num_hyp,
                "hyp_ref_ratio": num_hyp / num_ref if num_ref else 0.0,
            }
        return {"n": len(result), "per_id": result, "summary": summary,
                "head_names": sorted(expected_heads),
                "cpu_cuda_compat": cpu_compat,
                "checkpoint_state_key_count": checkpoint_state_key_count}
    finally:
        os.chdir(old)


def run_one(input_pickle: str | Path, split: str, out_dir: str | Path,
            config: str | Path = DEFAULT_CONFIG, checkpoint: str | Path = DEFAULT_CHECKPOINT,
            input_sha256: str | None = None, config_sha256: str | None = None,
            checkpoint_sha256: str | None = None,
            batch_size: int = 1, num_workers: int = 0, beam_size: int = 5,
            evaluator: Callable[..., dict] = evaluate_ids) -> dict[str, Any]:
    if split not in {"dev", "test"}: raise ValueError("split must be dev/test")
    if batch_size < 1 or num_workers < 0 or beam_size < 1:
        raise ValueError("batch size/worker count/beam size are invalid")
    inp, cfg, ckpt = confined(input_pickle), confined(config), confined(checkpoint)
    out = confined(out_dir); out.mkdir(parents=True, exist_ok=True)
    if not inp.is_file() or not cfg.is_file() or not ckpt.is_file(): raise FileNotFoundError("input/config/checkpoint missing")
    in_hash = _check_hash(inp, input_sha256, "input")
    cfg_hash = _check_hash(cfg, config_sha256, "config")
    ck_hash = _check_hash(ckpt, checkpoint_sha256, "checkpoint")
    canonical = MSKA_DATA / f"CSL-Daily.{split}"
    canonical_before = sha256(canonical)
    run_id = f"{split}_{in_hash[:12]}"
    log = out / f"mska_{run_id}.log"; result_path = out / f"mska_{run_id}.json"
    started = time.perf_counter()
    started_unix_ns = time.time_ns()
    implementation_hash = sha256(Path(__file__).resolve())
    try:
        with log.open("w") as f:
            f.write(json.dumps({
                "input": str(inp),
                "input_sha256": in_hash,
                "implementation_sha256": implementation_hash,
                "started_unix_ns": started_unix_ns,
            }) + "\n")
        result = evaluator(inp, cfg, ckpt, batch_size, num_workers, beam_size)
        completed_unix_ns = time.time_ns()
        input_ids = {str(k) for k in pickle.load(inp.open("rb"))}
        if set(result.get("per_id", {})) != input_ids:
            raise RuntimeError("result IDs do not exactly match supplied input")
        head_names = set(result.get("head_names", []))
        if not head_names:
            raise RuntimeError("result does not declare evaluated model heads")
        for sid, row in result["per_id"].items():
            missing_heads = head_names - set(row)
            if missing_heads:
                raise RuntimeError(
                    f"result has incomplete model-head coverage for {sid}: "
                    f"missing={sorted(missing_heads)}"
                )
        result.update({"split": split, "input_sha256": in_hash, "config_sha256": cfg_hash, "checkpoint_sha256": ck_hash,
                       "input_path": str(inp),
                       "implementation_sha256": implementation_hash,
                       "canonical_split_sha256": canonical_before,
                       "isolated_input": True,
                       "started_unix_ns": started_unix_ns,
                       "completed_unix_ns": completed_unix_ns,
                       "result_path": str(result_path), "log_path": str(log),
                       "batch_size": batch_size, "num_workers": num_workers,
                       "beam_size": beam_size,
                       "elapsed_seconds": time.perf_counter() - started,
                       "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        canonical_after = sha256(canonical)
        if canonical_after != canonical_before:
            raise RuntimeError(
                "canonical MSKA split changed during isolated evaluation: "
                f"{canonical_after} != {canonical_before}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--split", required=True, choices=["dev", "test"]); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG)); ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT)); ap.add_argument("--input-sha256", required=True); ap.add_argument("--config-sha256", required=True); ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--beam-size", type=int, default=5)
    a = ap.parse_args(); result = run_one(
        a.input, a.split, a.out_dir, a.config, a.checkpoint,
        a.input_sha256, a.config_sha256, a.checkpoint_sha256,
        a.batch_size, a.num_workers, a.beam_size,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_id"}, indent=2))


if __name__ == "__main__": main()
