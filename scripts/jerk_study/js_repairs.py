import numpy as np
from js_common import frozen_blend

def plain_concat(segs):
    return np.concatenate([s.copy() for s in segs], 0).astype(np.float32)

def eff_h(h, prev, cur, need=2):
    return int(max(0, min(h, prev.shape[0] // 2, cur.shape[0] // 2, prev.shape[0] - need, cur.shape[0] - need)))

def ramp_exclusive(segs, w=4, target="shifted"):
    """Frozen blend but the ramp never reaches 1.0 (k+1)/(w+1). target: 'shifted' pairs prev[-w+k] with cur[k]
    as the frozen code does; 'hold' pairs every tail frame with cur[0]."""
    segs = [s.copy() for s in segs]; out = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]; ww = min(w, prev.shape[0] // 2, cur.shape[0] // 2)
        if ww >= 1:
            k = np.arange(ww); r = (0.5 * (1 - np.cos(np.pi * (k + 1) / (ww + 1)))).astype(np.float32)[:, None]
            tgt = cur[:ww] if target == "shifted" else np.repeat(cur[:1], ww, 0)
            prev[-ww:] = (1 - r) * prev[-ww:] + r * tgt
        out.append(cur)
    return np.concatenate(out, 0).astype(np.float32)

def xfade_extrap(segs, h=2):
    """Velocity-consistent two-sided crossfade: each side is faded toward the OTHER side's linear
    extrapolation at the same time index. Touches h frames on each side of the join."""
    segs = [s.copy() for s in segs]; out = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]; hh = eff_h(h, prev, cur)
        if hh >= 1:
            vA = prev[-1] - prev[-2]; vB = cur[1] - cur[0]
            for m in range(2 * hh):
                wgt = 0.5 * (1 - np.cos(np.pi * (m + 1) / (2 * hh + 1)))
                if m < hh:
                    k = -hh + m; tgt = cur[0] + k * vB          # cur extrapolated backward to this time index
                    prev[k] = (1 - wgt) * prev[k] + wgt * tgt
                else:
                    k = m - hh; src = prev[-1] + (k + 1) * vA   # prev extrapolated forward (uses the ORIGINAL prev tail)
                    cur[k] = (1 - wgt) * src + wgt * cur[k]
        out.append(cur)
    return np.concatenate(out, 0).astype(np.float32)

def _hermite5(A, vA, aA, B, vB, aB, N):
    """Quintic Hermite from A (s=0) to B (s=1); derivatives given per frame; N frame steps A->B."""
    s = np.arange(1, N)[:, None] / N
    H0 = 1 - 10*s**3 + 15*s**4 - 6*s**5; H1 = s - 6*s**3 + 8*s**4 - 3*s**5; H2 = 0.5*s**2 - 1.5*s**3 + 1.5*s**4 - 0.5*s**5
    H3 = 10*s**3 - 15*s**4 + 6*s**5;    H4 = -4*s**3 + 7*s**4 - 3*s**5;      H5 = 0.5*s**3 - s**4 + 0.5*s**5
    return A*H0 + N*vA*H1 + N*N*aA*H2 + B*H3 + N*vB*H4 + N*N*aB*H5

def _hermite3(A, vA, B, vB, N):
    s = np.arange(1, N)[:, None] / N
    return A*(2*s**3 - 3*s**2 + 1) + N*vA*(s**3 - 2*s**2 + s) + B*(-2*s**3 + 3*s**2) + N*vB*(s**3 - s**2)

def hermite_bridge(segs, h=2, order=5, base="concat", accel=True):
    """Replace the 2h frames around each join (last h of prev, first h of cur) by a minimum-jerk
    polynomial matching position, velocity (and acceleration if order 5 and accel) at the anchor frames
    prev[-h-1] and cur[h]. base='concat' bridges the raw concatenation; base='frozen' bridges the frozen blend."""
    if base == "frozen":
        flat, _ = frozen_blend(segs); segs_b = []; t = 0
        for s in segs: segs_b.append(flat[t:t + s.shape[0]].copy()); t += s.shape[0]
        segs = segs_b
    else:
        segs = [s.copy() for s in segs]
    out = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]; hh = eff_h(h, prev, cur, need=2)
        if hh >= 1:
            A = prev[-hh-1]; B = cur[hh]; N = 2 * hh + 1
            vA = prev[-hh-1] - prev[-hh-2]; vB = cur[hh+1] - cur[hh]
            aA = (prev[-hh-1] - 2*prev[-hh-2] + prev[-hh-3]) if (accel and prev.shape[0] >= hh + 3) else np.zeros_like(A)
            aB = (cur[hh+2] - 2*cur[hh+1] + cur[hh]) if (accel and cur.shape[0] >= hh + 3) else np.zeros_like(B)
            br = _hermite5(A, vA, aA, B, vB, aB, N) if order == 5 else _hermite3(A, vA, B, vB, N)
            prev[-hh:] = br[:hh]; cur[:hh] = br[hh:]
        out.append(cur)
    return np.concatenate(out, 0).astype(np.float32)

def binomial_local(segs, radius=3, lam=1.0):
    """Lead's family: 5-tap binomial smoothing within +-radius of each join on the frozen blend output,
    mixed back with strength lam under a raised-cosine taper that is zero at the window edges."""
    x, _ = frozen_blend(segs); T = x.shape[0]
    xp = np.pad(x, ((2, 2), (0, 0)), mode="edge"); k = np.array([1, 4, 6, 4, 1], np.float32) / 16
    sm = sum(k[i] * xp[i:i + T] for i in range(5))
    tap = np.zeros(T, np.float32); t = 0
    for s in segs[:-1]:
        t += s.shape[0]; c = t - 0.5
        for u in range(max(0, t - radius), min(T, t + radius)):
            tap[u] = max(tap[u], 0.5 * (1 + np.cos(np.pi * (u - c) / (radius + 0.5))))
    return (x + lam * tap[:, None] * (sm - x)).astype(np.float32)

def rcx_bridge(segs, h=2, p=None, coeff_sink=None):
    """RCX (retimed convex crossfade). Same footprint as hermite_bridge: the 2h frames strictly
    between the anchors A = prev[-h-1] and B = cur[h] are overwritten in place (length preserved,
    N = 2h+1). Source a is read from A forward with decelerating rate tau_a'(s) ~ (1-s)**p and
    source b is read up to B with accelerating rate tau_b'(s) ~ s**p, p = (h+1)/h; each map
    integrates to exactly h source frames, which makes the bridge leave A at source a's unit speed
    and arrive at B at source b's unit speed. The two retimed reads are mixed with the raised-cosine
    alpha(s) = 0.5*(1-cos(pi*s)) (alpha' = 0 at both ends). Each read is a linear interpolation of
    two ADJACENT archive rows, so every emitted row is a convex combination of four archive rows;
    the per-source weights are (1-alpha, alpha), so the ledger transfer is sum(alpha) = h per join,
    identical to the frozen cosine blend and to the Hermite bridge."""
    segs = [s.copy() for s in segs]; out = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]; hh = eff_h(h, prev, cur, need=2)
        if hh >= 1:
            q = ((hh + 1.0) / hh if p is None else float(p)) + 1.0   # q = p+1
            na = prev.shape[0]; N = 2 * hh + 1
            a_src = prev[na - hh - 1:].copy()      # h+1 rows: A .. prev[-1], read before overwrite
            b_src = cur[:hh + 1].copy()            # h+1 rows: cur[0] .. B
            for k in range(1, N):
                s = k / N
                ta = hh * (1.0 - (1.0 - s) ** q)   # frames travelled into source a from A
                tb = hh * s ** q                   # frames travelled into source b from cur[0]
                al = 0.5 * (1.0 - np.cos(np.pi * s))
                ia = min(int(ta), hh - 1); fa = ta - ia
                ib = min(int(tb), hh - 1); fb = tb - ib
                c = ((1 - al) * (1 - fa), (1 - al) * fa, al * (1 - fb), al * fb)
                assert min(c) >= 0.0 and abs(sum(c) - 1.0) <= 1e-6, f"RCX not convex: {c}"
                if coeff_sink is not None: coeff_sink.append(c)
                row = c[0] * a_src[ia] + c[1] * a_src[ia + 1] + c[2] * b_src[ib] + c[3] * b_src[ib + 1]
                if k <= hh: prev[na - hh - 1 + k] = row
                else:       cur[k - hh - 1] = row
        out.append(cur)
    return np.concatenate(out, 0).astype(np.float32)
