"""PG-RAST: per-gloss real exemplar retrieval + native-T concat with
boundary cosine blending.

For each dev/test clip:
  1. Read its predicted B1 gloss list (deployable spans_pred_v2.json) OR
     its true gloss sequence (from manifest, optional)
  2. For each gloss in order, retrieve an exemplar from the bank:
       - default: weighted random from top-K most similar by signer/handshape
       - simple: random per-gloss
  3. Concatenate at the EXEMPLAR's NATIVE duration (preserves motion energy)
  4. Linear-blend ±BLEND_W frames at each gloss boundary

Output: phoenix14t.<split> in PT-150 format ready for audit + MSKA WER.

Usage
  python -u scripts/pg_rast_assemble.py \\
    --bank outputs/sota_chase/phase24_pgrast/exemplars.pt \\
    --tag pgrast_naive --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data.datasets import PhoenixT2SDataset


HAND_XY_DIMS_201 = []
for _base in (75, 138):
    for _j in range(21):
        HAND_XY_DIMS_201.extend([_base + 3 * _j, _base + 3 * _j + 1])


def _ex_len(ex):
    return int(ex[2] - ex[1])


def _get_exemplar_segment(exemplar_poses, ex):
    sid, s, e = ex
    return exemplar_poses[sid][s:e].astype(np.float32)


def _adjust_duration(seg, target_T, mode="hold", trim_mode="middle"):
    """Resize a retrieved segment without mixing it with any generated pose.

    `hold` is the legacy behavior. `loop` preserves more per-frame motion than
    last-frame replication when a selected exemplar is shorter than target_T.
    """
    if target_T is None:
        return seg
    target_T = int(max(2, target_T))
    T = int(seg.shape[0])
    if T == target_T:
        return seg
    if T < 2:
        return np.repeat(seg[:1], target_T, axis=0)
    if mode == "window":
        mode = "loop"
    if T < target_T:
        n_pad = target_T - T
        if mode == "hold":
            return np.concatenate([seg, np.tile(seg[-1:], (n_pad, 1))], axis=0)
        if mode == "loop":
            chunks = [seg]
            src = seg[1:] if T > 2 else seg
            while sum(c.shape[0] for c in chunks) < target_T:
                chunks.append(src)
            return np.concatenate(chunks, axis=0)[:target_T]
        if mode == "pingpong":
            chunks = [seg]
            forward = True
            while sum(c.shape[0] for c in chunks) < target_T:
                if forward:
                    nxt = seg[-2:0:-1] if T > 3 else seg[::-1]
                else:
                    nxt = seg[1:-1] if T > 3 else seg
                chunks.append(nxt)
                forward = not forward
            return np.concatenate(chunks, axis=0)[:target_T]
        raise ValueError(f"Unknown duration mode: {mode}")
    # T > target_T
    if T <= target_T + 4:
        return seg[:target_T]
    if trim_mode == "center":
        mid = T // 2
        half = target_T // 2
        s = max(0, mid - half)
        return seg[s:s + target_T]
    if trim_mode == "headtail":
        # Keep onset and offset frames; drop the center where holds often live.
        n_head = target_T // 2
        n_tail = target_T - n_head
        return np.concatenate([seg[:n_head], seg[-n_tail:]], axis=0)
    raise ValueError(f"Unknown trim mode: {trim_mode}")


def _adjust_duration_window(seg, target_T, prev_seg=None, boundary_w=4,
                            energy_floor=True):
    """For over-long exemplars, choose a real-frame window that best connects
    to the previous segment while preserving internal velocity. Short segments
    use loop padding. This emits only retrieved exemplar frames.
    """
    if target_T is None:
        return seg
    target_T = int(max(2, target_T))
    T = int(seg.shape[0])
    if T <= target_T:
        return _adjust_duration(seg, target_T, mode="loop", trim_mode="center")
    if T <= target_T + 4:
        return seg[:target_T]
    if prev_seg is None:
        return _adjust_duration(seg, target_T, mode="hold", trim_mode="center")

    # Limit enumeration to keep assembly fast; include endpoints and center.
    max_start = T - target_T
    starts = set(np.linspace(0, max_start, num=min(max_start + 1, 9), dtype=int).tolist())
    starts.add(max_start // 2)
    full_vel = float(np.mean((seg[1:, HAND_XY_DIMS_201] - seg[:-1, HAND_XY_DIMS_201]) ** 2))
    best = None
    for s in sorted(starts):
        win = seg[s:s + target_T]
        cost = _boundary_cost(prev_seg, win, boundary_w=boundary_w)
        if energy_floor and win.shape[0] > 1 and full_vel > 1e-8:
            win_vel = float(np.mean((win[1:, HAND_XY_DIMS_201] - win[:-1, HAND_XY_DIMS_201]) ** 2))
            # Penalize low-motion windows so seam optimization cannot choose
            # holds. High-motion windows are allowed; they are real frames.
            cost += max(0.0, 0.75 - win_vel / full_vel)
        rec = (cost, s, win)
        if best is None or rec[:2] < best[:2]:
            best = rec
    return best[2]


def _boundary_cost(prev_seg, cur_seg, boundary_w=4):
    if prev_seg is None or prev_seg.shape[0] < 2 or cur_seg.shape[0] < 2:
        return 0.0
    w = min(boundary_w, prev_seg.shape[0], cur_seg.shape[0])
    if w < 1:
        return 0.0
    dims = [d for d in HAND_XY_DIMS_201 if d < prev_seg.shape[1] and d < cur_seg.shape[1]]
    if not dims:
        dims = list(range(min(prev_seg.shape[1], cur_seg.shape[1])))
    tail = prev_seg[-w:, dims]
    head = cur_seg[:w, dims]
    pose_cost = float(np.mean(np.abs(tail - head)))
    if w > 1:
        prev_vel = prev_seg[-1:, dims] - prev_seg[-2:-1, dims]
        cur_vel = cur_seg[1:2, dims] - cur_seg[:1, dims]
        vel_cost = float(np.mean(np.abs(prev_vel - cur_vel)))
    else:
        vel_cost = 0.0
    return pose_cost + 0.5 * vel_cost


def _rle_codes(seq):
    arr = [int(x) for x in np.asarray(seq).reshape(-1).tolist()]
    out = []
    last = None
    for x in arr:
        if x != last:
            out.append(x)
            last = x
    return out


def _lcs_dist(a, b):
    if not a or not b:
        return 1.0
    # Two-row LCS DP; normalized distance in [0, 1].
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def _edit_dist_norm(a, b):
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[-1] + 1,
                prev[j - 1] + (0 if x == y else 1),
            ))
        prev = cur
    return prev[-1] / max(len(a), len(b), 1)


def _bigram_dist(a, b):
    if len(a) < 2 or len(b) < 2:
        return _lcs_dist(a, b)
    sa = set(zip(a[:-1], a[1:]))
    sb = set(zip(b[:-1], b[1:]))
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return 1.0 - len(sa & sb) / union


def _upc_seq_cost(target_codes, ex_codes, mode):
    target = _rle_codes(target_codes)
    ex = _rle_codes(ex_codes)
    if not target or not ex:
        return 1.0
    if mode == "set":
        target_set = set(target)
        ex_set = set(ex)
        union = len(target_set | ex_set)
        return 1.0 - len(target_set & ex_set) / union if union else 0.0
    if mode == "rle_lcs":
        return _lcs_dist(target, ex)
    if mode == "edit":
        return _edit_dist_norm(target, ex)
    if mode == "bigram":
        return _bigram_dist(target, ex)
    if mode == "hybrid":
        return 0.45 * _lcs_dist(target, ex) + 0.35 * _edit_dist_norm(target, ex) + 0.20 * _bigram_dist(target, ex)
    raise ValueError(f"Unknown upc_cost_mode: {mode}")


def _active_hand_weights(target_upc):
    if target_upc is None:
        return {"lh": 0.5, "rh": 0.5}
    scores = {}
    for hand in ("lh", "rh"):
        codes = target_upc.get(f"{hand}_codes")
        rle = _rle_codes(codes if codes is not None else [])
        scores[hand] = max(len(rle) - 1, 0)
    if scores["lh"] == scores["rh"]:
        return {"lh": 0.5, "rh": 0.5}
    active = "lh" if scores["lh"] > scores["rh"] else "rh"
    passive = "rh" if active == "lh" else "lh"
    return {active: 0.7, passive: 0.3}


def _upc_mismatch(exemplar_upc, ex, target_upc, mode="set", active_weight=False):
    if exemplar_upc is None or target_upc is None:
        return 0.0
    sid, s, e = ex
    info = exemplar_upc.get((sid, int(s), int(e)))
    if info is None:
        return 0.5
    costs = []
    weights = _active_hand_weights(target_upc) if active_weight else {"lh": 0.5, "rh": 0.5}
    for hand in ("lh", "rh"):
        target_codes = target_upc.get(f"{hand}_codes")
        if target_codes is None or len(target_codes) == 0:
            continue
        ex_codes = info.get(f"{hand}_codes")
        if ex_codes is not None and len(ex_codes) > 0:
            costs.append((weights[hand], _upc_seq_cost(target_codes, ex_codes, mode)))
            continue
        dom = int(info.get(f"{hand}_dom", -1))
        if dom < 0:
            continue
        # Cost 0 if the exemplar dominant code appears in this target span;
        # otherwise use 1. This is intentionally coarse so UPC cannot become
        # another posterior-mean scoring path.
        costs.append((weights[hand], 0.0 if dom in set(map(int, target_codes)) else 1.0))
    if not costs:
        return 0.0
    wsum = sum(w for w, _c in costs)
    return float(sum(w * c for w, c in costs) / max(wsum, 1e-6))


def _candidate_records(ex_list, exemplar_poses, target_T, rng,
                       sid_to_signer=None, prefer_signer=None,
                       strict_signer=False, candidate_k=16,
                       duration_mode="hold", trim_mode="center",
                       length_weight=1.0, signer_weight=0.25,
                       exemplar_upc=None, target_upc=None, upc_weight=0.0,
                       upc_cost_mode="set", upc_active_hand=False,
                       exemplar_context=None, target_prev=None,
                       target_next=None, context_weight=0.0):
    if not ex_list:
        return []
    cand_pool = ex_list
    if sid_to_signer is not None and prefer_signer is not None:
        signer_match = [ex for ex in ex_list if sid_to_signer.get(ex[0]) == prefer_signer]
        if signer_match and strict_signer:
            cand_pool = signer_match
    target_len = int(target_T) if target_T is not None else None
    if target_len is not None:
        cand_pool = sorted(cand_pool, key=lambda ex: abs(_ex_len(ex) - target_len))
    else:
        cand_pool = list(cand_pool)
        rng.shuffle(cand_pool)
    cand_pool = cand_pool[:max(1, int(candidate_k))]

    out = []
    for ex in cand_pool:
        raw = _get_exemplar_segment(exemplar_poses, ex)
        if raw.shape[0] < 2:
            continue
        seg = _adjust_duration(raw, target_len, mode=duration_mode, trim_mode=trim_mode)
        raw_len = _ex_len(ex)
        if target_len is None:
            length_cost = 0.0
        else:
            length_cost = abs(raw_len - target_len) / max(float(target_len), 1.0)
        signer_cost = 0.0
        if sid_to_signer is not None and prefer_signer is not None:
            signer_cost = 0.0 if sid_to_signer.get(ex[0]) == prefer_signer else 1.0
        upc_cost = _upc_mismatch(
            exemplar_upc, ex, target_upc,
            mode=upc_cost_mode,
            active_weight=upc_active_hand,
        )
        context_cost = 0.0
        if exemplar_context is not None and context_weight > 0:
            ctx = exemplar_context.get((ex[0], int(ex[1]), int(ex[2])), {})
            parts = []
            if target_prev is not None and ctx.get("prev") is not None:
                parts.append(0.0 if ctx.get("prev") == target_prev else 1.0)
            if target_next is not None and ctx.get("next") is not None:
                parts.append(0.0 if ctx.get("next") == target_next else 1.0)
            context_cost = float(np.mean(parts)) if parts else 0.0
        unary = (length_weight * length_cost +
                 signer_weight * signer_cost +
                 upc_weight * upc_cost +
                 context_weight * context_cost)
        tie = (str(ex[0]), int(ex[1]), int(ex[2]))
        meta = {
            "sid": ex[0], "start": int(ex[1]), "end": int(ex[2]),
            "raw_len": int(raw_len), "out_len": int(seg.shape[0]),
            "score": float(unary), "length_cost": float(length_cost),
            "boundary_cost": 0.0, "signer_cost": float(signer_cost),
            "upc_cost": float(upc_cost), "context_cost": float(context_cost),
        }
        out.append((unary, tie, ex, seg, meta))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def _beam_select_segments(glosses, gloss_to_exemplars, exemplar_poses,
                          per_gloss_targets, rng, fallback_random_clip=True,
                          sid_to_signer=None, prefer_signer=None,
                          strict_signer=False, candidate_k=16,
                          beam_size=8, duration_mode="hold",
                          trim_mode="center", boundary_w=4,
                          length_weight=1.0, boundary_weight=0.2,
                          signer_weight=0.25, exemplar_upc=None,
                          target_upc_segments=None, upc_weight=0.0,
                          upc_cost_mode="set", upc_active_hand=False,
                          exemplar_context=None, context_weight=0.0):
    beam = [(0.0, (), [], [])]  # score, tie tuple, segments, trace
    for idx, (g, target_T) in enumerate(zip(glosses, per_gloss_targets)):
        ex_list = gloss_to_exemplars.get(g)
        if not ex_list:
            if fallback_random_clip:
                some_g = rng.choice(list(gloss_to_exemplars.keys()))
                ex_list = gloss_to_exemplars[some_g]
            else:
                continue
        target_upc = None if target_upc_segments is None else target_upc_segments[idx]
        target_prev = glosses[idx - 1] if idx > 0 else None
        target_next = glosses[idx + 1] if idx + 1 < len(glosses) else None
        cands = _candidate_records(
            ex_list, exemplar_poses, target_T, rng,
            sid_to_signer=sid_to_signer,
            prefer_signer=prefer_signer,
            strict_signer=strict_signer,
            candidate_k=candidate_k,
            duration_mode=duration_mode,
            trim_mode=trim_mode,
            length_weight=length_weight,
            signer_weight=signer_weight,
            exemplar_upc=exemplar_upc,
            target_upc=target_upc,
            upc_weight=upc_weight,
            upc_cost_mode=upc_cost_mode,
            upc_active_hand=upc_active_hand,
            exemplar_context=exemplar_context,
            target_prev=target_prev,
            target_next=target_next,
            context_weight=context_weight,
        )
        if not cands:
            continue
        next_beam = []
        for prev_score, prev_tie, prev_segments, prev_trace in beam:
            prev_seg = prev_segments[-1] if prev_segments else None
            for unary, tie, _ex, seg, meta in cands:
                boundary = _boundary_cost(prev_seg, seg, boundary_w=boundary_w)
                new_meta = dict(meta)
                new_meta["gloss"] = g
                new_meta["target_T"] = int(target_T) if target_T is not None else None
                new_meta["boundary_cost"] = float(boundary)
                new_meta["score"] = float(unary + boundary_weight * boundary)
                next_beam.append((
                    prev_score + unary + boundary_weight * boundary,
                    prev_tie + (tie,),
                    prev_segments + [seg],
                    prev_trace + [new_meta],
                ))
        next_beam.sort(key=lambda r: (r[0], r[1]))
        beam = next_beam[:max(1, int(beam_size))]
    if not beam:
        return [], []
    return beam[0][2], beam[0][3]


def _target_upc_for_segments(target_upc_clip, per_gloss_targets):
    if not target_upc_clip:
        return [None] * len(per_gloss_targets)
    lh = target_upc_clip.get("lh_codes")
    rh = target_upc_clip.get("rh_codes")
    if lh is None or rh is None:
        return [None] * len(per_gloss_targets)
    total_T = max(sum(int(t or 0) for t in per_gloss_targets), 1)
    out = []
    cursor = 0
    for t in per_gloss_targets:
        t = int(t or max(total_T // max(len(per_gloss_targets), 1), 1))
        s = int(round(cursor / total_T * len(lh)))
        e = int(round((cursor + t) / total_T * len(lh)))
        if e <= s:
            e = min(len(lh), s + 1)
        out.append({
            "lh_codes": np.asarray(lh[s:e], dtype=np.int16),
            "rh_codes": np.asarray(rh[s:e], dtype=np.int16),
        })
        cursor += t
    return out


def _pick_exemplar_scored(ex_list, exemplar_poses, target_T, rng,
                          prev_seg=None, sid_to_signer=None,
                          prefer_signer=None, strict_signer=False,
                          candidate_k=16, duration_mode="hold",
                          trim_mode="center", boundary_w=4,
                          length_weight=1.0, boundary_weight=0.2,
                          signer_weight=0.25, exemplar_upc=None,
                          target_upc=None, upc_weight=0.0,
                          upc_cost_mode="set", upc_active_hand=False,
                          exemplar_context=None, target_prev=None,
                          target_next=None, context_weight=0.0):
    best = None
    for unary, tie, ex, seg, meta in _candidate_records(
        ex_list, exemplar_poses, target_T, rng,
        sid_to_signer=sid_to_signer,
        prefer_signer=prefer_signer,
        strict_signer=strict_signer,
        candidate_k=candidate_k,
        duration_mode=duration_mode,
        trim_mode=trim_mode,
        length_weight=length_weight,
        signer_weight=signer_weight,
        exemplar_upc=exemplar_upc,
        target_upc=target_upc,
        upc_weight=upc_weight,
        upc_cost_mode=upc_cost_mode,
        upc_active_hand=upc_active_hand,
        exemplar_context=exemplar_context,
        target_prev=target_prev,
        target_next=target_next,
        context_weight=context_weight,
    ):
        if duration_mode == "window":
            raw = _get_exemplar_segment(exemplar_poses, ex)
            seg = _adjust_duration_window(raw, target_T, prev_seg=prev_seg,
                                          boundary_w=boundary_w)
            meta = {**meta, "out_len": int(seg.shape[0])}
        boundary = _boundary_cost(prev_seg, seg, boundary_w=boundary_w)
        score = unary + boundary_weight * boundary
        rec = (score, tie, ex, seg, {**meta, "score": float(score),
                                     "boundary_cost": float(boundary)})
        if best is None or (rec[0], rec[1]) < (best[0], best[1]):
            best = rec
    if best is None:
        return None, None
    return best[3], best[4]


def _choose_auto_signer(glosses, per_gloss_targets, gloss_to_exemplars, sid_to_signer):
    """Pick one train signer with best coverage + length fit for this clip."""
    if not sid_to_signer:
        return None
    signers = sorted(set(sid_to_signer.values()))
    best = None
    for signer in signers:
        coverage = 0
        length_err = 0.0
        for gloss, target_T in zip(glosses, per_gloss_targets):
            exs = [ex for ex in gloss_to_exemplars.get(gloss, [])
                   if sid_to_signer.get(ex[0]) == signer]
            if not exs:
                length_err += 10.0
                continue
            coverage += 1
            if target_T is not None:
                target_T = max(int(target_T), 1)
                length_err += min(abs(_ex_len(ex) - target_T) / target_T for ex in exs)
        rec = (-coverage, length_err, signer)
        if best is None or rec < best:
            best = rec
    return best[2] if best is not None else None


def _build_exemplar_context(gloss_to_exemplars):
    by_sid = defaultdict(list)
    for gloss, entries in gloss_to_exemplars.items():
        for sid, s, e in entries:
            by_sid[sid].append((int(s), int(e), gloss))
    ctx = {}
    for sid, rows in by_sid.items():
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        for i, (s, e, gloss) in enumerate(rows):
            prev_g = rows[i - 1][2] if i > 0 else None
            next_g = rows[i + 1][2] if i + 1 < len(rows) else None
            ctx[(sid, int(s), int(e))] = {"gloss": gloss, "prev": prev_g, "next": next_g}
    return ctx


def _pick_exemplar_for_target(ex_list, exemplar_poses, target_T, rng,
                               sid_to_signer=None, prefer_signer=None):
    """Pick the exemplar whose native length is closest to target_T (or
    larger), optionally restricted to those from `prefer_signer` first."""
    if not ex_list:
        return None
    # Optional signer-consistency filter
    cand_pool = ex_list
    if sid_to_signer is not None and prefer_signer is not None:
        signer_match = [(sid, s, e) for (sid, s, e) in ex_list
                        if sid_to_signer.get(sid) == prefer_signer]
        if signer_match:
            cand_pool = signer_match
    candidates = [(sid, s, e, e - s) for (sid, s, e) in cand_pool]
    if target_T is None:
        sid, s, e, _ = rng.choice(candidates)
        return exemplar_poses[sid][s:e].astype(np.float32)
    over = [c for c in candidates if c[3] >= target_T - 2]
    if over:
        sid, s, e, L = rng.choice(over)
    else:
        sid, s, e, L = max(candidates, key=lambda c: c[3])
    return exemplar_poses[sid][s:e].astype(np.float32)


def assemble_clip(glosses, gloss_to_exemplars, exemplar_poses, blend_w=4,
                  rng=None, fallback_random_clip=True,
                  pred_spans=None,
                  sid_to_signer=None, prefer_signer=None,
                  selection="legacy", candidate_k=16,
                  beam_size=8,
                  duration_mode="hold", trim_mode="center",
                  boundary_w=4, length_weight=1.0,
                  boundary_weight=0.2, signer_weight=0.25,
                  strict_signer=False, exemplar_upc=None,
                  target_upc_clip=None, upc_weight=0.0,
                  upc_cost_mode="set", upc_active_hand=False,
                  exemplar_context=None, context_weight=0.0,
                  boundary_mode="overlap", return_trace=False):
    """Concat exemplars per gloss, with optional cosine blend at boundaries.

    pred_spans: optional list of [(gloss_in_pred, T_predicted), ...] to
    drive each gloss's target duration. If exemplar shorter than target,
    we time-stretch to target (via linear interp). Better than pure native
    because MSKA WER cares about length-axis.
    """
    rng = rng or random.Random(0)
    segments = []
    if pred_spans is not None:
        # Map gloss → its predicted T (in order; if same gloss appears
        # multiple times, we use queue order)
        pq = defaultdict(deque)
        for g, T in pred_spans:
            pq[g].append(int(T))
        # Build per-gloss target list ordered as glosses appears
        per_gloss_targets = []
        for g in glosses:
            if pq.get(g):
                per_gloss_targets.append(pq[g].popleft())
            else:
                per_gloss_targets.append(None)
    else:
        per_gloss_targets = [None] * len(glosses)

    target_upc_segments = _target_upc_for_segments(target_upc_clip, per_gloss_targets)
    trace = []
    if selection == "beam":
        segments, trace = _beam_select_segments(
            glosses, gloss_to_exemplars, exemplar_poses, per_gloss_targets, rng,
            fallback_random_clip=fallback_random_clip,
            sid_to_signer=sid_to_signer,
            prefer_signer=prefer_signer,
            strict_signer=strict_signer,
            candidate_k=candidate_k,
            beam_size=beam_size,
            duration_mode=duration_mode,
            trim_mode=trim_mode,
            boundary_w=boundary_w,
            length_weight=length_weight,
            boundary_weight=boundary_weight,
            signer_weight=signer_weight,
            exemplar_upc=exemplar_upc,
            target_upc_segments=target_upc_segments,
            upc_weight=upc_weight,
            upc_cost_mode=upc_cost_mode,
            upc_active_hand=upc_active_hand,
            exemplar_context=exemplar_context,
            context_weight=context_weight,
        )
        if not segments:
            out = np.zeros((4, 201), dtype=np.float32)
            return (out, trace) if return_trace else out
        # Fall through to the shared boundary blending block.
        if len(segments) == 1:
            return (segments[0], trace) if return_trace else segments[0]
        out = [segments[0]]
        for i in range(1, len(segments)):
            prev = out[-1]
            cur = segments[i]
            w = min(blend_w, prev.shape[0] // 2, cur.shape[0] // 2)
            if w < 1:
                out.append(cur); continue
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w))).astype(np.float32)[:, None]
            prev_tail = prev[-w:]
            cur_head = cur[:w]
            blend = (1.0 - ramp) * prev_tail + ramp * cur_head
            prev[-w:] = blend
            if boundary_mode == "overlap":
                cur = cur[w:]
            elif boundary_mode != "preserve":
                raise ValueError(f"Unknown boundary_mode: {boundary_mode}")
            out.append(cur)
        final = np.concatenate(out, axis=0)
        return (final, trace) if return_trace else final

    prev_seg = None
    for seg_idx, (g, target_T) in enumerate(zip(glosses, per_gloss_targets)):
        ex_list = gloss_to_exemplars.get(g)
        if not ex_list:
            if fallback_random_clip:
                some_g = rng.choice(list(gloss_to_exemplars.keys()))
                ex_list = gloss_to_exemplars[some_g]
            else:
                continue
        if selection == "legacy":
            seg = _pick_exemplar_for_target(ex_list, exemplar_poses, target_T, rng,
                                              sid_to_signer=sid_to_signer,
                                              prefer_signer=prefer_signer)
            meta = None
            if target_T is not None and seg is not None and seg.shape[0] < target_T:
                n_pad = target_T - seg.shape[0]
                seg = np.concatenate([seg, np.tile(seg[-1:], (n_pad, 1))], axis=0)
            elif target_T is not None and seg is not None and seg.shape[0] > target_T + 4:
                mid = seg.shape[0] // 2
                half = target_T // 2
                seg = seg[max(0, mid - half): mid + (target_T - half)]
        else:
            seg, meta = _pick_exemplar_scored(
                ex_list, exemplar_poses, target_T, rng,
                prev_seg=prev_seg,
                sid_to_signer=sid_to_signer,
                prefer_signer=prefer_signer,
                strict_signer=strict_signer,
                candidate_k=candidate_k,
                duration_mode=duration_mode,
                trim_mode=trim_mode,
                boundary_w=boundary_w,
                length_weight=length_weight,
                boundary_weight=boundary_weight,
                signer_weight=signer_weight,
                exemplar_upc=exemplar_upc,
                target_upc=target_upc_segments[seg_idx],
                upc_weight=upc_weight,
                upc_cost_mode=upc_cost_mode,
                upc_active_hand=upc_active_hand,
                exemplar_context=exemplar_context,
                target_prev=glosses[seg_idx - 1] if seg_idx > 0 else None,
                target_next=glosses[seg_idx + 1] if seg_idx + 1 < len(glosses) else None,
                context_weight=context_weight,
            )
        if seg is None or seg.shape[0] < 2:
            continue
        if meta is not None:
            meta["gloss"] = g
            meta["target_T"] = int(target_T) if target_T is not None else None
            trace.append(meta)
        segments.append(seg)
        prev_seg = seg
    if not segments:
        out = np.zeros((4, 201), dtype=np.float32)
        return (out, trace) if return_trace else out
    if len(segments) == 1:
        return (segments[0], trace) if return_trace else segments[0]

    # Cosine-blend boundaries
    out = [segments[0]]
    for i in range(1, len(segments)):
        prev = out[-1]
        cur = segments[i]
        w = min(blend_w, prev.shape[0] // 2, cur.shape[0] // 2)
        if w < 1:
            out.append(cur); continue
        # cosine ramp from 0 → 1 over w frames at the boundary
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w))).astype(np.float32)[:, None]
        prev_tail = prev[-w:]
        cur_head = cur[:w]
        blend = (1.0 - ramp) * prev_tail + ramp * cur_head
        prev[-w:] = blend
        if boundary_mode == "overlap":
            # Legacy Phase-24 behavior: overlap the boundary by dropping the
            # current head. This smooths transitions but shortens each clip by
            # roughly blend_w * (n_glosses - 1), which can raise DEL.
            cur = cur[w:]
        elif boundary_mode == "preserve":
            # Keep total segment duration. The previous tail becomes a
            # transition into the current segment, but no current frames are
            # removed. This is still pure exemplar motion, not v55 mixing.
            pass
        else:
            raise ValueError(f"Unknown boundary_mode: {boundary_mode}")
        out.append(cur)
    final = np.concatenate(out, axis=0)
    return (final, trace) if return_trace else final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="outputs/sota_chase/phase24_pgrast/exemplars.pt")
    ap.add_argument("--tag", default="pgrast_naive")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--body_adapter", default="first8")
    ap.add_argument("--blend_w", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_pred_glosses", default="",
                    help="JSON of per-clip predicted gloss list. Empty=use GT glosses (oracle).")
    ap.add_argument("--per_gloss_spans_json", default="outputs/sota_chase/phase3e_pergloss/spans_pred_v2.json",
                    help="Predicted per-gloss span lengths to target. Empty=native exemplar length.")
    ap.add_argument("--signer_consistent", action="store_true",
                    help="Pick exemplars all from the same train signer per dev clip "
                         "(prefer dev clip's own signer if available in train).")
    ap.add_argument("--auto_signer", action="store_true",
                    help="Choose the train signer with best clip-level coverage/length fit.")
    ap.add_argument("--selection", choices=["legacy", "scored", "beam"], default="legacy",
                    help="legacy reproduces Phase-24 PG-RAST; scored is PG-RAST++ top-K selection.")
    ap.add_argument("--candidate_k", type=int, default=16)
    ap.add_argument("--beam_size", type=int, default=8)
    ap.add_argument("--duration_mode", choices=["hold", "loop", "pingpong", "window"], default="hold")
    ap.add_argument("--trim_mode", choices=["center", "headtail"], default="center")
    ap.add_argument("--boundary_w", type=int, default=4)
    ap.add_argument("--boundary_mode", choices=["overlap", "preserve"], default="overlap")
    ap.add_argument("--length_weight", type=float, default=1.0)
    ap.add_argument("--boundary_weight", type=float, default=0.2)
    ap.add_argument("--signer_weight", type=float, default=0.25)
    ap.add_argument("--strict_signer", action="store_true",
                    help="When a preferred signer has candidates for a gloss, restrict to that signer.")
    ap.add_argument("--target_upc_path", default="",
                    help="Optional per-clip v55 UPC-code dump for coarse token compatibility scoring.")
    ap.add_argument("--upc_weight", type=float, default=0.0)
    ap.add_argument("--upc_cost_mode", choices=["set", "rle_lcs", "edit", "bigram", "hybrid"],
                    default="set")
    ap.add_argument("--upc_active_hand", action="store_true",
                    help="Weight the target hand with more UPC transitions more heavily.")
    ap.add_argument("--context_weight", type=float, default=0.0,
                    help="Prefer exemplars whose original prev/next gloss context matches the target sequence.")
    ap.add_argument("--save_trace", action="store_true")
    ap.add_argument("--use_gt_total_T", action="store_true",
                    help="Diagnostic Upper Bound mode: scale per-gloss spans so the "
                         "per-clip total = item['length'] (the GT clip length). "
                         "Mirrors Sign-IDD's GT-T leak for apples-to-apples WER cells. "
                         "Result MUST be labelled Diagnostic Upper Bound, NOT deployable.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading exemplar bank: {args.bank}")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    g2e = bank["gloss_to_exemplars"]
    pose_pool = bank["exemplar_poses"]
    exemplar_upc = bank.get("exemplar_upc")
    print(f"  {bank['n_glosses']} glosses, {bank['n_exemplars']} exemplars, "
          f"{len(pose_pool)} source clips")
    exemplar_context = _build_exemplar_context(g2e) if args.context_weight > 0 else None
    if exemplar_context is not None:
        print(f"  built exemplar context for {len(exemplar_context)} spans")

    target = PhoenixT2SDataset("data/phoenix", args.split, max_frames=500,
                                augment=False, return_face=False)
    print(f"Assembling {len(target.samples)} {args.split} clips ...")

    # Build sid -> signer mapping if signer_consistent
    sid_to_signer = None
    if args.signer_consistent or args.auto_signer:
        train_ds = PhoenixT2SDataset("data/phoenix", "train", max_frames=500,
                                       augment=False, return_face=False)
        sid_to_signer = {s["id"]: s["signer"] for s in train_ds.samples}
        print(f"  signer_consistent: built map for {len(sid_to_signer)} train clips")

    pred_glosses_lookup = None
    if args.use_pred_glosses:
        with open(args.use_pred_glosses) as f:
            pred_glosses_lookup = json.load(f)

    pred_spans_lookup = None
    if args.per_gloss_spans_json:
        with open(args.per_gloss_spans_json) as f:
            sp = json.load(f)
        pred_spans_lookup = sp.get("spans", sp)

    target_upc_lookup = None
    if args.target_upc_path:
        target_upc_lookup = torch.load(args.target_upc_path, map_location="cpu",
                                       weights_only=False)
        print(f"  loaded target UPC for {len(target_upc_lookup)} clips: {args.target_upc_path}")

    preds = {}
    traces = {}
    misses = 0
    for item in target.samples:
        sid = item["id"]
        glosses = (pred_glosses_lookup or {}).get(sid) or item["gloss"].split()
        spans_for_clip = None
        if pred_spans_lookup is not None:
            spans_for_clip = pred_spans_lookup.get(sid)
            if spans_for_clip is None:
                for prefix in (args.split, "dev", "test", "train"):
                    spans_for_clip = pred_spans_lookup.get(f"{prefix}/{sid}")
                    if spans_for_clip is not None:
                        break
        n_missing = sum(1 for g in glosses if not g2e.get(g))
        if n_missing == len(glosses):
            misses += 1
        # Diagnostic Upper Bound mode: rescale per-gloss spans so total = GT-T,
        # mirroring Sign-IDD's GT-T leak so the WER cell is apples-to-apples.
        if args.use_gt_total_T and spans_for_clip is not None:
            gt_T = int(item.get("length", 0) or 0)
            cur_total = sum(int(t) for _, t in spans_for_clip)
            if gt_T > 0 and cur_total > 0 and cur_total != gt_T:
                scale = gt_T / cur_total
                rescaled = []
                acc = 0
                for j, (g, t) in enumerate(spans_for_clip):
                    if j == len(spans_for_clip) - 1:
                        new_t = max(2, gt_T - acc)
                    else:
                        new_t = max(2, int(round(int(t) * scale)))
                        acc += new_t
                    rescaled.append((g, new_t))
                spans_for_clip = rescaled
        per_gloss_targets = None
        if spans_for_clip is not None:
            pq = defaultdict(deque)
            for g, T in spans_for_clip:
                pq[g].append(int(T))
            per_gloss_targets = []
            for g in glosses:
                per_gloss_targets.append(pq[g].popleft() if pq.get(g) else None)
        if args.auto_signer:
            prefer_signer = _choose_auto_signer(
                glosses, per_gloss_targets or [None] * len(glosses),
                g2e, sid_to_signer,
            )
        else:
            prefer_signer = item.get("signer") if args.signer_consistent else None
        pose, trace = assemble_clip(
            glosses, g2e, pose_pool,
            blend_w=args.blend_w, rng=rng,
            pred_spans=spans_for_clip,
            sid_to_signer=sid_to_signer,
            prefer_signer=prefer_signer,
            selection=args.selection,
            candidate_k=args.candidate_k,
            beam_size=args.beam_size,
            duration_mode=args.duration_mode,
            trim_mode=args.trim_mode,
            boundary_w=args.boundary_w,
            length_weight=args.length_weight,
            boundary_weight=args.boundary_weight,
            signer_weight=args.signer_weight,
            strict_signer=args.strict_signer,
            exemplar_upc=exemplar_upc,
            target_upc_clip=(target_upc_lookup or {}).get(sid),
            upc_weight=args.upc_weight,
            upc_cost_mode=args.upc_cost_mode,
            upc_active_hand=args.upc_active_hand,
            exemplar_context=exemplar_context,
            context_weight=args.context_weight,
            boundary_mode=args.boundary_mode,
            return_trace=True,
        )
        preds[sid] = torch.from_numpy(pose.astype(np.float32))
        if args.save_trace:
            traces[sid] = {
                "prefer_signer": prefer_signer,
                "glosses": glosses,
                "segments": trace,
            }
    print(f"  {misses} clips with NO matching glosses (used random exemplars)")

    # Save PT-201 dump for downstream
    out_dir = Path("outputs/sota_chase/phase24_pgrast")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = {k: v.numpy().astype(np.float16) for k, v in preds.items()}
    out_pt_path = out_dir / f"v55_pt201_{args.split}_{args.tag}.pt"
    torch.save(out_pt, out_pt_path)
    print(f"\nSaved {len(out_pt)} preds → {out_pt_path}")
    if args.save_trace:
        trace_path = out_dir / f"trace_{args.split}_{args.tag}.json"
        with open(trace_path, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"  trace → {trace_path}")

    # Stats: mean T
    Ts = [v.shape[0] for v in preds.values()]
    print(f"  T mean={np.mean(Ts):.1f} p10/50/90={np.percentile(Ts,[10,50,90])}")

    print(f"\nNow run:")
    print(f"  python scripts/transport_to_phoenix14t.py "
          f"--transported_pt {out_pt_path} --tag {args.tag} "
          f"--split {args.split} --body_adapter {args.body_adapter}")


if __name__ == "__main__":
    main()
