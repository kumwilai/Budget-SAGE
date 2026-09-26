#!/usr/bin/env python3
"""Signer round analysis: relative contrasts only, cluster bootstrap over raters.

    python3 scripts/analyse_signer_round.py <responses.csv> [--json out.json]

The pre-registered Gate 0 band for this round is `relative_only`, so no absolute
score is reported. Every contrast is paired inside a rater and resampled at the
rater level, 2000 resamples, seed 42.

P288 and P289 are researcher system tests and are excluded by id.
"""
import argparse, csv, json, collections, statistics, random, sys

csv.field_size_limit(10**9)
TEST_IDS = {"P288", "P289"}
KEY_PATH = "outputs/human_eval_kit_signer_2026-09-14/condition_key.json"
N_BOOT, SEED = 2000, 42


def load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = []
    for x in rows:
        if x["participant"] in TEST_IDS:
            continue
        try:
            d = json.loads(x["payload"]).get("payload", {})
        except Exception:
            continue
        d["_kind"] = x["kind"]
        out.append(d)
    return out


def boot_mean(by_rater, n=N_BOOT, seed=SEED):
    """Cluster bootstrap of a mean over per-rater lists of values."""
    keys = sorted(by_rater)
    flat = [v for k in keys for v in by_rater[k]]
    if not flat:
        return None, None, None, 0, 0
    point = statistics.mean(flat)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        vals = []
        for _ in keys:
            vals.extend(by_rater[keys[rng.randrange(len(keys))]])
        if vals:
            draws.append(statistics.mean(vals))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return point, lo, hi, len(flat), len(keys)


def fmt(name, r, pct=False):
    p, lo, hi, n, k = r
    if p is None:
        return f"{name:<34} no data"
    s = 100.0 if pct else 1.0
    u = "%" if pct else ""
    return (f"{name:<34} {p*s:+7.3f}{u} [{lo*s:+7.3f}, {hi*s:+7.3f}]"
            f"   n={n:<4} raters={k}") if not pct else (
            f"{name:<34} {p*s:6.1f}{u} [{lo*s:5.1f}, {hi*s:5.1f}]"
            f"   n={n:<4} raters={k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    key = json.load(open(KEY_PATH))["by_file"]
    def cond(code):
        return key.get(code, {}).get("condition")

    recs = load(a.csv)
    ratings = [d for d in recs if d["_kind"] == "rating"]
    sessions = [d for d in recs if d["_kind"] == "session"]

    ended = collections.defaultdict(list)
    for d in sessions:
        if d.get("ended"):
            ended[d["participant"]].append(d["ended"])

    res = {"file": a.csv}

    # ---------- roster ----------
    segs = collections.defaultdict(collections.Counter)
    for d in ratings:
        segs[d["participant"]][d.get("seg", "?")] += 1
    entrants = sorted(segs)
    stopped = [p for p in entrants if "gate0_stop" in ended.get(p, [])]
    main = [p for p in entrants
            if sum(segs[p][s] for s in ("b1a", "b1b", "b2", "b3")) > 0]
    res["roster"] = {"entrants": len(entrants), "gate0_stop": len(stopped),
                     "with_main_blocks": sorted(main),
                     "complete": sorted(p for p in entrants
                                        if "complete" in ended.get(p, []))}

    # ---------- b1b: paired single ratings, hybrid minus archive ----------
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in ratings:
        if d.get("seg") != "b1b":
            continue
        c = cond(d.get("code"))
        if c in ("hyb", "ref", "bud"):
            cells[(d["participant"], d["item"])][c].append(d)

    scales = {"naturalness": "natural", "continuity": "contin",
              "clarity": "clarity"}
    res["contrasts"] = {}
    for label, fld in scales.items():
        by_rater = collections.defaultdict(list)
        for (pid, item), v in cells.items():
            if "hyb" in v and "ref" in v:
                h = [x[fld] for x in v["hyb"] if x.get(fld) is not None]
                r = [x[fld] for x in v["ref"] if x.get(fld) is not None]
                if h and r:
                    by_rater[pid].append(statistics.mean(h) - statistics.mean(r))
        res["contrasts"][label] = boot_mean(by_rater)

    # pace on a 3-point scale, and the rate of "I saw a seam"
    PACE = {"slow": 1.0, "ok": 2.0, "fast": 3.0}
    for label, get in (("pace_1to3", lambda x: PACE.get(x.get("pace"))),
                       ("seam_seen_rate",
                        lambda x: 1.0 if x.get("seam") == "yes" else 0.0)):
        by_rater = collections.defaultdict(list)
        for (pid, item), v in cells.items():
            if "hyb" in v and "ref" in v:
                h = [y for y in map(get, v["hyb"]) if y is not None]
                r = [y for y in map(get, v["ref"]) if y is not None]
                if h and r:
                    by_rater[pid].append(statistics.mean(h) - statistics.mean(r))
        res["contrasts"][label] = boot_mean(by_rater)

    # The all-generated anchor rated inside the same block, against the same
    # rater's archive clips. Not paired within sentence, because the anchor is
    # a single clip, so this is a rater-level contrast.
    by_rater = collections.defaultdict(list)
    n_allg = 0
    per_allg = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in ratings:
        if d.get("seg") != "b1b":
            continue
        c = cond(d.get("code"))
        if c in ("allg1_anchor", "ref") and d.get("natural") is not None:
            per_allg[d["participant"]][c].append(d["natural"])
    for pid, v in per_allg.items():
        if "allg1_anchor" in v and "ref" in v:
            by_rater[pid].append(statistics.mean(v["allg1_anchor"])
                                 - statistics.mean(v["ref"]))
            n_allg += len(v["allg1_anchor"])
    res["allgen_minus_archive_naturalness"] = boot_mean(by_rater)
    res["allgen_rating_count"] = n_allg

    # Gate 0 speed manipulation check, over EVERY entrant: an artificially
    # sped-up ground-truth clip minus the unmodified one. An interval that
    # contains zero means the check failed, which is reported as a limitation.
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in ratings:
        if d.get("seg") != "gate0":
            continue
        c = cond(d.get("code"))
        if c in ("fast_ctl", "gt_anchor") and d.get("natural") is not None:
            per[d["participant"]][c].append(d["natural"])
    by_rater = collections.defaultdict(list)
    for pid, v in per.items():
        if "fast_ctl" in v and "gt_anchor" in v:
            by_rater[pid].append(statistics.mean(v["fast_ctl"])
                                 - statistics.mean(v["gt_anchor"]))
    res["speed_check_fast_minus_groundtruth"] = boot_mean(by_rater)

    # ---------- b1a: forced choice, hybrid preferred over archive ----------
    by_rater = collections.defaultdict(list)
    for d in ratings:
        if d.get("seg") != "b1a" or d.get("kind") not in ("primary", "retest"):
            continue
        cf, cs, ch = cond(d.get("first")), cond(d.get("second")), d.get("choice")
        if ch not in ("first", "second"):
            continue
        picked = cf if ch == "first" else cs
        if {cf, cs} == {"hyb", "ref"}:
            by_rater[d["participant"]].append(1.0 if picked == "hyb" else 0.0)
    res["forced_choice_hyb_over_ref"] = boot_mean(by_rater)

    # ---------- b2: detection, "which one is generated" ----------
    det = {"b2_real": collections.defaultdict(list),
           "b2_pos": collections.defaultdict(list),
           "b2_neg": collections.defaultdict(list)}
    GEN = {"hyb", "b2_hyb", "b3_hyb0", "b3_hybwm", "allg1_anchor",
           "b2ctlpos2_allg1", "b3_allg0", "b3_allg1", "b3_allg2"}
    for d in ratings:
        if d.get("seg") != "b2":
            continue
        k = d.get("kind")
        if k not in det:
            continue
        cf, cs, g = cond(d.get("first")), cond(d.get("second")), d.get("gen")
        if g not in ("first", "second"):
            continue
        picked = cf if g == "first" else cs
        gen_side = [c for c in (cf, cs) if c in GEN]
        if k == "b2_neg":
            correct = 1.0          # both sides identical: any answer is the control
        elif len(gen_side) == 1:
            correct = 1.0 if picked == gen_side[0] else 0.0
        else:
            continue
        det[k][d["participant"]].append(correct)
    res["detect_hybrid_vs_archive"] = boot_mean(det["b2_real"])
    res["detect_allgen_vs_groundtruth"] = boot_mean(det["b2_pos"])
    res["detect_control_same_vs_same"] = boot_mean(det["b2_neg"])

    # ---------- attention ----------
    ok = tot = 0
    per = collections.defaultdict(lambda: [0, 0])
    for d in ratings:
        if d.get("kind") != "attention":
            continue
        tot += 1
        per[d["participant"]][1] += 1
        if d.get("choice") == d.get("a_first"):
            ok += 1
            per[d["participant"]][0] += 1
    res["attention"] = {"correct": ok, "total": tot,
                        "raters_all_correct": sum(1 for v in per.values()
                                                  if v[0] == v[1]),
                        "raters": len(per)}

    # ---------- print ----------
    print(f"file: {a.csv}")
    r = res["roster"]
    print(f"entrants {r['entrants']}   gate0_stop {r['gate0_stop']}   "
          f"with main blocks {len(r['with_main_blocks'])} "
          f"{r['with_main_blocks']}")
    print(f"marked complete: {r['complete']}")
    print()
    print("hybrid minus archive, paired within rater and sentence:")
    for lbl in ("naturalness", "continuity", "clarity", "pace_1to3",
                "seam_seen_rate"):
        print("  " + fmt(lbl, res["contrasts"][lbl]))
    print()
    print("  " + fmt("forced choice hyb over ref",
                     res["forced_choice_hyb_over_ref"], pct=True))
    print("  " + fmt("detect hybrid vs archive",
                     res["detect_hybrid_vs_archive"], pct=True))
    print("  " + fmt("all-gen minus archive (natural)",
                     res["allgen_minus_archive_naturalness"]),
          f'  all-G ratings={res["allgen_rating_count"]}')
    print("  " + fmt("detect all-gen vs ground truth",
                     res["detect_allgen_vs_groundtruth"], pct=True))
    print("  " + fmt("detect control same vs same",
                     res["detect_control_same_vs_same"], pct=True))
    print("  " + fmt("gate-0 speed check (all entrants)",
                     res["speed_check_fast_minus_groundtruth"]))
    print()
    print(f"  attention {res['attention']['correct']} of "
          f"{res['attention']['total']} items, "
          f"{res['attention']['raters_all_correct']} of "
          f"{res['attention']['raters']} raters perfect")

    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=list)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
