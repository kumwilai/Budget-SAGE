"""Clean native-178 text-retrieval baseline for SLRTP evaluation.

This is a MoE-style sanity probe: retrieve a whole train signing clip from
source German text similarity, then submit that native SLRTP-178 pose as the
prediction. It uses only train texts/poses and eval source texts; eval
references are not read for selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _load_slrtp_split(split: str, data_root: Path) -> dict:
    path = data_root / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _texts_and_keys(data: dict) -> tuple[list[str], list[str]]:
    keys = list(data.keys())
    texts = [str(data[k].get("text", "")) for k in keys]
    return keys, texts


def _build_features(train_texts: list[str], eval_texts: list[str]):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    xw_tr = word.fit_transform(train_texts)
    xw_ev = word.transform(eval_texts)
    xc_tr = char.fit_transform(train_texts)
    xc_ev = char.transform(eval_texts)
    x_tr = normalize(hstack([0.60 * xw_tr, 0.40 * xc_tr]).tocsr(), norm="l2")
    x_ev = normalize(hstack([0.60 * xw_ev, 0.40 * xc_ev]).tocsr(), norm="l2")
    return x_tr, x_ev


def _retrieve_top1(
    x_train,
    x_eval,
    train_texts: list[str],
    eval_texts: list[str],
    max_sim: float = 1.01,
    exclude_exact: bool = False,
) -> tuple[list[int], list[float]]:
    sims = x_eval @ x_train.T
    picks: list[int] = []
    scores: list[float] = []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            picks.append(0)
            scores.append(0.0)
            continue
        order = row.data.argsort()[::-1]
        chosen = None
        eval_norm = " ".join(eval_texts[i].lower().split())
        for local in order:
            j = int(row.indices[int(local)])
            score = float(row.data[int(local)])
            if score > max_sim:
                continue
            if exclude_exact and " ".join(train_texts[j].lower().split()) == eval_norm:
                continue
            chosen = (j, score)
            break
        if chosen is None:
            chosen = (0, 0.0)
        picks.append(chosen[0])
        scores.append(chosen[1])
    return picks, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--data_root", default=(
        "external/SLRTP-Sign-Production-Evaluation/pretrained/"
        "SLRTP-Sign-Production-Evaluation-Data/data"
    ))
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="")
    ap.add_argument("--max_sim", type=float, default=1.01,
                    help="Reject train candidates with text similarity above this value.")
    ap.add_argument("--exclude_exact", action="store_true",
                    help="Reject exact normalized train/eval sentence matches.")
    args = ap.parse_args()

    data_root = ROOT / args.data_root
    train = _load_slrtp_split("train", data_root)
    ev = _load_slrtp_split(args.split, data_root)
    train_keys, train_texts = _texts_and_keys(train)
    eval_keys, eval_texts = _texts_and_keys(ev)

    x_train, x_eval = _build_features(train_texts, eval_texts)
    picks, scores = _retrieve_top1(
        x_train,
        x_eval,
        train_texts=train_texts,
        eval_texts=eval_texts,
        max_sim=args.max_sim,
        exclude_exact=args.exclude_exact,
    )

    pred = {}
    trace = []
    for key, eval_text, j, score in zip(eval_keys, eval_texts, picks, scores):
        train_key = train_keys[j]
        pred[key] = train[train_key]["poses_3d"].clone()
        trace.append({
            "id": key,
            "text": eval_text,
            "retrieved_id": train_key,
            "retrieved_text": train_texts[j],
            "score": score,
            "T": int(pred[key].shape[0]),
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    trace_path = ROOT / (args.trace or f"{out.with_suffix('').as_posix()}_trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

    lens = torch.tensor([v.shape[0] for v in pred.values()], dtype=torch.float32)
    print(f"train={len(train_keys)} {args.split}={len(eval_keys)}")
    print(f"saved -> {out}")
    print(f"trace -> {trace_path}")
    print(f"T mean={lens.mean().item():.1f} p10/50/90="
          f"{[round(x, 1) for x in torch.quantile(lens, torch.tensor([0.1, 0.5, 0.9])).tolist()]}")
    print(f"similarity mean={sum(scores)/max(len(scores), 1):.4f} "
          f"max={max(scores) if scores else 0.0:.4f}")


if __name__ == "__main__":
    main()
