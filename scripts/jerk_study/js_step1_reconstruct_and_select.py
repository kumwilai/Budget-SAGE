import time, json, math, sys
sys.path.insert(0, "/tmp/claude-1000/-home-kumwilai-research-signgen-t2m/ef04ff5e-1c47-4276-b131-90e6a9dbc109/scratchpad")
from js_common import *
t0 = time.time()
bank = torch.load(BANK, map_location="cpu", weights_only=False); pool = bank["exemplar_poses"]; g2e = bank["gloss_to_exemplars"]
trace = json.load(open(TRACE)); local = torch.load(LOCAL_PT, map_location="cpu", weights_only=False)
plans_raw = json.load(open(PLANS)); ids = [str(s) for s in plans_raw]
plans = {s: (v.split() if isinstance(v, str) else [str(t) for t in v]) for s, v in plans_raw.items()}
# exclusions exactly as _load_exclusions
rows = json.load(open(DONOR)); rows = list(rows.values()) if isinstance(rows, dict) else rows
excl = {}
for r in rows:
    rid = r.get("retrieved_id")
    if r.get("source") == "retrieval" and rid:
        f = {str(v) for v in r.get("forbidden_source_ids", [])}; f.add(str(rid)); excl[str(r["id"])] = f
assert all(excl.get(s) for s in ids)
print("loaded", time.time() - t0)
# 1) reconstruct frozen output from trace -> raw archive segments; verify bit-exact
raw_segments = {}; mism = 0; maxdiff = 0.0
for sid in ids:
    segs = [pool[s["source"]][s["start"]:s["end"]].astype(np.float32) for s in trace[sid]["segments"]]
    for s, seg in zip(trace[sid]["segments"], segs): assert seg.shape[0] == s["out_len"] == s["raw_len"]
    raw_segments[sid] = segs
    rec, _ = frozen_blend(segs); ref = local[sid].reshape(local[sid].shape[0], -1).numpy()
    d = float(np.abs(rec - ref).max()); maxdiff = max(maxdiff, d); mism += int(d > 0)
print(f"reconstruction: clips={len(ids)} mismatching={mism} max_abs_diff={maxdiff}")
torch.save(raw_segments, OUT / "frozen_local_test_raw_segments.pt")
# join stats and near-join jerk on xyz (paper scale) to confirm finding 1 holds in xyz
near = []; far = []
for sid in ids:
    segs = raw_segments[sid]; pose = local[sid].reshape(local[sid].shape[0], 178, 3)[:, 8:50].numpy().astype(np.float64)
    if pose.shape[0] < 4: continue
    j = np.linalg.norm(np.diff(pose, 3, axis=0), axis=2).mean(1)  # per third-diff window t..t+3
    J = set(); pos = join_positions(segs)
    for p, w in zip(pos, [min(4, a.shape[0]//2, b.shape[0]//2) for a, b in zip(segs[:-1], segs[1:])]):
        J.update(range(p - w, p + 1))  # blended frames p-w..p-1 plus first new frame p
    for t in range(len(j)):
        (near if any(u in J for u in range(t, t + 4)) else far).append(j[t])
print(f"xyz near-join median {np.median(near):.5f} n={len(near)} ; elsewhere {np.median(far):.5f} n={len(far)} ; GT med {GT_MED['hand_jerk']:.5f}")
# 2) faithful re-implementation of selection, generalized
def build_phrase_bank(min_phrase=2, max_phrase=4):
    by_sid = defaultdict(list)
    for gloss, rws in g2e.items():
        for sid, s, e in rws: by_sid[sid].append((int(s), int(e), str(gloss)))
    for sid in by_sid: by_sid[sid].sort(key=lambda r: (r[0], r[1], r[2]))
    p2c = defaultdict(list)
    for sid, rws in by_sid.items():
        n = len(rws)
        for i in range(n):
            for L in range(min_phrase, max_phrase + 1):
                if i + L > n: continue
                s = rws[i][0]; e = rws[i+L-1][1]
                if e <= s + 1: continue
                p2c[tuple(r[2] for r in rws[i:i+L])].append((sid, s, e, e - s))
    for k in list(p2c): p2c[k] = sorted(set(p2c[k]))
    return p2c
p2c = build_phrase_bank()
def cost_terms(prev, cur, bw=4):
    if prev is None or prev.shape[0] < 2 or cur.shape[0] < 2: return 0.0, 0.0, 0.0, 0.0
    w = min(bw, prev.shape[0], cur.shape[0]); d = HAND_XY
    pose_c = float(np.mean(np.abs(prev[-w:, d] - cur[:w, d])))
    pv = prev[-1:, d] - prev[-2:-1, d]; cv = cur[1:2, d] - cur[:1, d]
    vel_c = float(np.mean(np.abs(pv - cv))) if w > 1 else 0.0
    acc_c = 0.0
    if prev.shape[0] >= 3 and cur.shape[0] >= 3:
        pa = prev[-1, d] - 2*prev[-2, d] + prev[-3, d]; ca = cur[2, d] - 2*cur[1, d] + cur[0, d]
        acc_c = float(np.mean(np.abs(pa - ca)))
    # direct: xyz hand jerk of the actual post-blend join window (what the metric sees)
    seg, _ = frozen_blend([prev[-8:], cur[:8]]); p = seg.reshape(-1, 178, 3)[:, 8:50]
    jerk_c = float(np.linalg.norm(np.diff(p, 3, axis=0), axis=2).mean()) if p.shape[0] >= 4 else 0.0
    return pose_c, vel_c, acc_c, jerk_c
def choose(cands, prev, counts, last, excluded, cap, L, cfg):
    filt = [c for c in cands if c[0] not in excluded and counts[c[0]] + L <= cap]
    if not filt: return None
    filt.sort(key=lambda r: (counts[r[0]], r[0] == last, r[0], r[1], r[2]))
    pool_ = filt[:max(1, cfg["k"])]; best = None
    for sid, s, e, rl in pool_:
        seg = pool[sid][s:e].astype(np.float32)
        if seg.shape[0] < 2: continue
        pc, vc, ac, jc = cost_terms(prev, seg)
        boundary = (jc / GT_MED["hand_jerk"]) if cfg.get("direct") else (pc + cfg["vw"] * vc + cfg.get("aw", 0.0) * ac)
        score = cfg["bw"] * boundary + counts[sid] / max(sum(counts.values()), 1) + (0.5 if sid == last else 0.0)
        rec = (score, (sid, int(s), int(e)), seg)
        if best is None or rec[:2] < best[:2]: best = rec
    return best
def assemble(sid, cfg):
    gl = plans[sid]; n = len(gl); cap = max(2, int(math.ceil(0.65 * max(n, 1))))
    segs = []; picks = []; counts = Counter(); last = None; i = 0
    while i < n:
        ch = None; cl = 1
        for L in range(min(4, n - i), 1, -1):
            c = p2c.get(tuple(gl[i:i+L]))
            if not c: continue
            r = choose(c, segs[-1] if segs else None, counts, last, excl[sid], cap, L, cfg)
            if r is not None: ch, cl = r, L; break
        if ch is None:
            c = [(ex[0], int(ex[1]), int(ex[2]), int(ex[2]-ex[1])) for ex in g2e.get(gl[i], [])]
            ch = choose(c, segs[-1] if segs else None, counts, last, excl[sid], cap, 1, cfg); cl = 1
        if ch is None: i += 1; continue
        segs.append(ch[2]); picks.append(ch[1]); counts[ch[1][0]] += cl; last = ch[1][0]; i += cl
    return segs, picks
gt = load_gt()
frozen_picks = {sid: [(s["source"], s["start"], s["end"]) for s in trace[sid]["segments"]] for sid in ids}
configs = [("frozen bw0.2 vw0.5 k32", dict(bw=0.2, vw=0.5, k=32)),
           ("bw0.2 vw2", dict(bw=0.2, vw=2.0, k=32)), ("bw0.2 vw5", dict(bw=0.2, vw=5.0, k=32)),
           ("bw1 vw0.5", dict(bw=1.0, vw=0.5, k=32)), ("bw5 vw0.5", dict(bw=5.0, vw=0.5, k=32)), ("bw5 vw5", dict(bw=5.0, vw=5.0, k=32)),
           ("bw1 vw2 aw2", dict(bw=1.0, vw=2.0, aw=2.0, k=32)),
           ("direct post-blend jerk bw1 k32", dict(bw=1.0, direct=True, k=32)), ("direct post-blend jerk bw5 k32", dict(bw=5.0, direct=True, k=32)),
           ("bw0.2 vw0.5 k256", dict(bw=0.2, vw=0.5, k=256)), ("bw1 vw2 k256", dict(bw=1.0, vw=2.0, k=256)),
           ("direct post-blend jerk bw5 k256", dict(bw=5.0, direct=True, k=256)),
           ("direct post-blend jerk bw5 k=all", dict(bw=5.0, direct=True, k=10**9))]
table = []
for name, cfg in configs:
    t1 = time.time(); preds = {}; changed = 0; total = 0
    for sid in ids:
        segs, picks = assemble(sid, cfg); preds[sid], _ = frozen_blend(segs)
        total += len(picks); changed += sum(1 for a, b in zip(picks, frozen_picks[sid]) if a != b) + abs(len(picks) - len(frozen_picks[sid]))
    r = ratios(preds, gt, ids)
    row = dict(name=name, **cfg, changed_frac=changed / total, **r, secs=time.time() - t1); table.append(row)
    print(f"{name:34s} jerk {r['hand_jerk']:.3f} speed {r['hand_speed']:.3f} posestd {r['hand_posestd']:.3f} obj {r['joint_obj']:.3f} | xy-lead jerk {r['xy_jerk_lead_scale']:.3f} speed {r['xy_speed_lead_scale']:.3f} | changed {changed/total:.3f} ({changed}/{total}) {time.time()-t1:.0f}s", flush=True)
    if cfg == dict(bw=0.2, vw=0.5, k=32):
        same = all(np.array_equal(preds[s], local[s].reshape(local[s].shape[0], -1).numpy()) for s in ids)
        print("   faithful re-implementation reproduces frozen .pt exactly:", same)
json.dump(table, open(OUT / "priority1_selection_sweep_test641.json", "w"), indent=1)
