"""SignJEPA-CPF exact generated-frame knapsack.

Given a source-assisted base route bank and a SignJEPA generated candidate bank,
choose which clips should be generated so that a minimum generated-frame mass is
met while the corpus WER edit-distance numerator is minimized.  Unlike the
fast CPF greedy policy, this is an exact dynamic program for the additive WER
objective under a generated-frame budget.  BLEU is then evaluated on the final
mixed corpus as a non-additive secondary metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SLRTP = ROOT / "external/SLRTP-Sign-Production-Evaluation"
sys.path.insert(0, str(SLRTP))

from budget_sage.generation.cpf_sage import aggregate_ledgers  # noqa: E402
from external_metrics.sacrebleu import raw_corpus_bleu  # noqa: E402
from scripts.cpf_sage_counterfactual_sim import build_ledgers  # noqa: E402


WER_TR = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    bleu = raw_corpus_bleu(sys_stream=hyps, ref_streams=[refs]).scores
    return {
        "bleu1": float(bleu[0]),
        "bleu2": float(bleu[1]),
        "bleu3": float(bleu[2]),
        "bleu4": float(bleu[3]),
        "wer": 100.0
        * float(
            jiwer.wer(
                refs,
                hyps,
                reference_transform=WER_TR,
                hypothesis_transform=WER_TR,
            )
        ),
    }


def word_error_counts(ref: str, hyp: str) -> tuple[int, int]:
    out = jiwer.process_words(
        [ref],
        [hyp],
        reference_transform=WER_TR,
        hypothesis_transform=WER_TR,
    )
    err = int(out.substitutions + out.deletions + out.insertions)
    nref = int(out.hits + out.substitutions + out.deletions)
    return err, nref


def solve_min_error_at_generated_target(
    weights: list[int],
    gen_errors: list[int],
    base_errors: list[int],
    target_frames: int,
) -> set[int]:
    """Exact DP: minimize total corpus error with generated weight >= target.

    State is clipped at target_frames, so all solutions meeting or exceeding
    the target share the same terminal bin.  Backpointers recover one optimal
    generated set under deterministic tie-breaking.
    """

    n = len(weights)
    target = int(max(0, target_frames))
    inf = 10**12
    dp = np.full(target + 1, inf, dtype=np.int64)
    dp[0] = 0
    take = np.zeros((n, target + 1), dtype=np.bool_)
    prev_state = np.zeros((n, target + 1), dtype=np.int32)

    for i, (w, ge, be) in enumerate(zip(weights, gen_errors, base_errors)):
        ndp = dp + int(be)
        nprev = np.arange(target + 1, dtype=np.int32)
        ntake = np.zeros(target + 1, dtype=np.bool_)
        w = int(max(0, w))
        ge = int(ge)
        if w > 0:
            for s in range(target + 1):
                if dp[s] >= inf:
                    continue
                ns = min(target, s + w)
                cand = int(dp[s] + ge)
                # Prefer generated only when it strictly improves the objective;
                # this makes exact ties select the lower-generated route.
                if cand < int(ndp[ns]):
                    ndp[ns] = cand
                    nprev[ns] = s
                    ntake[ns] = True
        dp = ndp
        take[i] = ntake
        prev_state[i] = nprev

    chosen: set[int] = set()
    state = target
    for i in range(n - 1, -1, -1):
        if take[i, state]:
            chosen.add(i)
        state = int(prev_state[i, state])
    return chosen


def greedy_min_error(weights: list[int], gen_errors: list[int], base_errors: list[int], target_frames: int) -> set[int]:
    rows = []
    for i, (w, ge, be) in enumerate(zip(weights, gen_errors, base_errors)):
        if w <= 0:
            continue
        delta = int(be) - int(ge)
        rows.append((-(delta / max(1, int(w))), -delta, i))
    rows.sort()
    chosen: set[int] = set()
    frames = 0
    for _score, _delta, i in rows:
        chosen.add(i)
        frames += int(weights[i])
        if frames >= target_frames:
            break
    return chosen


def materialize(
    ids: list[str],
    chosen_idx: set[int],
    refs: list[str],
    base_text: list[str],
    gen_text: list[str],
    base_pose: dict[str, torch.Tensor],
    gen_pose: dict[str, torch.Tensor],
    base_ledgers: dict[str, Any],
    gen_ledgers: dict[str, Any],
    tag: str,
    write_pose: bool,
) -> dict[str, Any]:
    chosen_sids = {ids[i] for i in chosen_idx}
    hyps = [gen_text[i] if i in chosen_idx else base_text[i] for i in range(len(ids))]
    active_ledgers = [gen_ledgers[sid] if sid in chosen_sids else base_ledgers[sid] for sid in ids]
    agg = aggregate_ledgers(active_ledgers)
    row = {
        "tag": tag,
        "rewritten_clips": len(chosen_idx),
        "metrics": metrics(refs, hyps),
        "ledger": agg,
        "rewritten_ids": sorted(chosen_sids),
    }
    if write_pose:
        pose = {sid: (gen_pose[sid].clone() if sid in chosen_sids else base_pose[sid].clone()) for sid in ids}
        torch.save(pose, SLRTP / f"results/{tag}.pt")
        torch.save(hyps, SLRTP / f"results/{tag}_text_preds.pt")
        row["pose_path"] = str(SLRTP / f"results/{tag}.pt")
        row["text_pred_path"] = str(SLRTP / f"results/{tag}_text_preds.pt")
    return row


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SignJEPA-CPF exact generated-frame knapsack",
        "",
        "| split | target gen % | solver | rewrites | BLEU-4 | WER | gen % | charged % |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        m = row["metrics"]
        led = row["ledger"]
        lines.append(
            f"| {payload['split']} | {100*row['target_generated_frame_fraction']:.0f} | {row['solver']} | "
            f"{row['rewritten_clips']} | {m['bleu4']:.2f} | {m['wer']:.2f} | "
            f"{led['generated_frame_pct']:.2f} | {led['charged_source_frame_pct']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--base_tag", default="phase58_b40_CAC_a0_00")
    ap.add_argument("--gen_tag", default="sign_jepa_v11_sentence_budget_governor")
    ap.add_argument("--generated_targets", type=float, nargs="+", default=[0.36, 0.38, 0.40, 0.42, 0.44])
    ap.add_argument("--out_tag", default="phase58_signjepa_cpf_knapsack")
    ap.add_argument("--out", default="outputs/learned_confidence/signjepa_cpf_knapsack.json")
    ap.add_argument("--write_pose", action="store_true")
    args = ap.parse_args()

    gt = torch.load(
        SLRTP / f"pretrained/SLRTP-Sign-Production-Evaluation-Data/data/{args.split}.pt",
        map_location="cpu",
        weights_only=False,
    )
    ids = list(gt.keys())
    refs = [gt[sid]["text"] for sid in ids]
    base_pose = torch.load(SLRTP / f"results/{args.base_tag}_{args.split}.pt", map_location="cpu", weights_only=False)
    gen_pose = torch.load(SLRTP / f"results/{args.gen_tag}_{args.split}.pt", map_location="cpu", weights_only=False)
    base_text = list(torch.load(SLRTP / f"results/{args.base_tag}_{args.split}_text_preds.pt", map_location="cpu", weights_only=False))
    gen_text = list(torch.load(SLRTP / f"results/{args.gen_tag}_{args.split}_text_preds.pt", map_location="cpu", weights_only=False))
    base_ledgers, gen_ledgers = build_ledgers(args.split, ids, base_pose, gen_pose)

    weights: list[int] = []
    base_errors: list[int] = []
    gen_errors: list[int] = []
    ref_words = 0
    for i, sid in enumerate(ids):
        weights.append(max(0, int(gen_ledgers[sid].generated_frames - base_ledgers[sid].generated_frames)))
        be, nr = word_error_counts(refs[i], base_text[i])
        ge, _ = word_error_counts(refs[i], gen_text[i])
        base_errors.append(be)
        gen_errors.append(ge)
        ref_words += nr
    total_frames = int(sum(ledger.total_frames for ledger in base_ledgers.values()))

    rows: list[dict[str, Any]] = []
    for frac in args.generated_targets:
        target_frames = int(round(float(frac) * float(total_frames)))
        for solver_name, solver in (
            ("exact_wer_dp", solve_min_error_at_generated_target),
            ("greedy_delta_per_frame", greedy_min_error),
        ):
            chosen = solver(weights, gen_errors, base_errors, target_frames)
            tag = f"{args.out_tag}_{solver_name}_g{int(round(100*frac)):02d}_{args.split}"
            row = materialize(
                ids, chosen, refs, base_text, gen_text,
                base_pose, gen_pose, base_ledgers, gen_ledgers,
                tag=tag,
                write_pose=args.write_pose,
            )
            row.update(
                {
                    "solver": solver_name,
                    "target_generated_frame_fraction": float(frac),
                    "target_generated_frames": int(target_frames),
                    "corpus_ref_words": int(ref_words),
                    "selected_error_sum": int(
                        sum(gen_errors[i] if i in chosen else base_errors[i] for i in range(len(ids)))
                    ),
                }
            )
            rows.append(row)

    payload = {
        "name": "SignJEPA-CPF exact generated-frame knapsack",
        "split": args.split,
        "base_tag": args.base_tag,
        "gen_tag": args.gen_tag,
        "definition": (
            "Use SignJEPA as a generated candidate bank and solve an exact dynamic program "
            "for minimum SLRTP-cycle WER numerator under a minimum generated-frame budget."
        ),
        "rows": rows,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    out_path.with_suffix(".md").write_text(markdown(payload))
    print(out_path.with_suffix(".md").read_text())
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
