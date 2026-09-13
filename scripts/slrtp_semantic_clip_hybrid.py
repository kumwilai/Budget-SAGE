"""Guarded semantic-clip retrieval + native PG-RAST++ fallback.

This combines two clean native-178 sources without using eval references:

1. Whole-clip train retrieval by German source-text similarity.
2. Predicted-gloss native PG-RAST++ fallback.

The clip branch is used only when the train sentence is sufficiently similar
and does not contradict explicit text slots such as day/month/weather/region.
This keeps the strong lexical recall of sentence retrieval while avoiding the
most obvious "wrong weather fact" failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


DAY = {
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag",
    "sonntag",
}
MONTH = {
    "januar", "februar", "maerz", "märz", "april", "mai", "juni", "juli",
    "august", "september", "oktober", "november", "dezember",
}
WEATHER = {
    "regen", "regnet", "schauer", "schnee", "schneit", "sonne", "sonnig",
    "wolke", "wolken", "wolkig", "bewoelkt", "bewölkt", "wind", "windig",
    "sturm", "nebel", "frost", "gewitter", "temperatur", "grad", "warm",
    "waermer", "wärmer", "kalt", "kaelter", "kälter", "heiss", "heiß",
    "tief", "hoch",
}
REGION = {
    "nord", "norden", "sued", "süd", "sueden", "süden", "ost", "osten",
    "west", "westen", "alpen", "kueste", "küste", "deutschland", "bayern",
    "rhein", "meer", "see",
}


def norm(s: str) -> str:
    return (s.lower().replace("ä", "ae").replace("ö", "oe")
            .replace("ü", "ue").replace("ß", "ss"))


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", norm(s)))


def slots(text: str) -> dict[str, set[str]]:
    t = toks(text)
    return {
        "day": t & {norm(x) for x in DAY},
        "month": t & {norm(x) for x in MONTH},
        "weather": t & {norm(x) for x in WEATHER},
        "region": t & {norm(x) for x in REGION},
        "number": {x for x in t if x.isdigit()},
    }


def slot_conflict(target: str, retrieved: str, strict: bool = True) -> bool:
    a = slots(target)
    b = slots(retrieved)
    for key in ("day", "month", "number"):
        if a[key] and b[key] and not (a[key] & b[key]):
            return True
    if not strict:
        return False
    # For weather/region, require at least one shared explicit slot when both
    # texts mention that type. This avoids swapping "rain in the Alps" for
    # unrelated "sun in the north" while still allowing generic forecasts.
    for key in ("weather", "region"):
        if a[key] and b[key] and not (a[key] & b[key]):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_pt", required=True)
    ap.add_argument("--retrieval_trace", required=True)
    ap.add_argument("--fallback_pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace_out", default="")
    ap.add_argument("--min_sim", type=float, default=0.50)
    ap.add_argument("--loose_slots", action="store_true")
    args = ap.parse_args()

    retrieval = torch.load(ROOT / args.retrieval_pt, map_location="cpu", weights_only=True)
    fallback = torch.load(ROOT / args.fallback_pt, map_location="cpu", weights_only=True)
    trace = json.load(open(ROOT / args.retrieval_trace))
    trace_by_id = {r["id"]: r for r in trace}

    pred = {}
    out_trace = []
    use_retrieval = 0
    conflict_n = 0
    low_sim_n = 0
    for key in fallback.keys():
        r = trace_by_id[key]
        conflict = slot_conflict(r["text"], r["retrieved_text"], strict=not args.loose_slots)
        low_sim = float(r["score"]) < args.min_sim
        choose_retrieval = (not conflict) and (not low_sim)
        if choose_retrieval:
            pred[key] = retrieval[key]
            use_retrieval += 1
            source = "retrieval"
        else:
            pred[key] = fallback[key]
            source = "pgrastpp"
            conflict_n += int(conflict)
            low_sim_n += int(low_sim)
        out_trace.append({
            **r,
            "source": source,
            "slot_conflict": conflict,
            "low_sim": low_sim,
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    trace_out = ROOT / (args.trace_out or f"{out.with_suffix('').as_posix()}_trace.json")
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    trace_out.write_text(json.dumps(out_trace, indent=2, ensure_ascii=False))

    lens = torch.tensor([v.shape[0] for v in pred.values()], dtype=torch.float32)
    n = len(pred)
    print(f"saved -> {out}")
    print(f"trace -> {trace_out}")
    print(f"retrieval used {use_retrieval}/{n} ({use_retrieval/max(n,1)*100:.1f}%)")
    print(f"fallback reasons: conflict={conflict_n} low_sim={low_sim_n}")
    print(f"T mean={lens.mean().item():.1f} p10/50/90="
          f"{[round(x, 1) for x in torch.quantile(lens, torch.tensor([0.1, 0.5, 0.9])).tolist()]}")


if __name__ == "__main__":
    main()
