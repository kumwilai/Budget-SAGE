"""Counterfactual Provenance Flow utilities.

CPF-SAGE treats source reuse as an intervention variable.  The utilities here
operate on route-level alternatives that already exist in the pipeline, for
example a high-scoring source-assisted output and a generated replacement.  A
counterfactual policy then asks: what changes if a deployer bans a source,
caps a source, or requires a minimum generated-frame share?

The module is deliberately model-agnostic.  It can drive fast simulations from
cached route banks today and can later wrap a BudgetFM checkpoint that produces
the generated replacement on demand.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RouteLedger:
    """Frame-mass and source-contribution ledger for one output route."""

    sid: str
    total_frames: int
    whole_clip_frames: int = 0
    local_unit_frames: int = 0
    generated_frames: int = 0
    unknown_or_derived_frames: int = 0
    source_frames: Mapping[str, int] = field(default_factory=dict)

    def charged_frames(self) -> int:
        """Conservative archived-motion charge.

        Unknown/derived frames are charged because a strict deployment cannot
        prove they are source-free.
        """

        return (
            int(self.whole_clip_frames)
            + int(self.local_unit_frames)
            + int(self.unknown_or_derived_frames)
        )

    def has_any_source(self, sources: set[str]) -> bool:
        return any(src in sources and n > 0 for src, n in self.source_frames.items())

    def contribution_to(self, sources: set[str]) -> int:
        return int(sum(n for src, n in self.source_frames.items() if src in sources))


@dataclass(frozen=True)
class CandidateSwap:
    """One counterfactual replacement option."""

    sid: str
    base: RouteLedger
    replacement: RouteLedger
    lexical_loss_proxy: float = 0.0
    priority_bonus: float = 0.0

    def generated_gain(self) -> int:
        return max(0, self.replacement.generated_frames - self.base.generated_frames)

    def charged_reduction(self) -> int:
        return max(0, self.base.charged_frames() - self.replacement.charged_frames())

    def source_reduction(self, sources: set[str]) -> int:
        return max(0, self.base.contribution_to(sources) - self.replacement.contribution_to(sources))


def segment_len(seg: Mapping[str, object]) -> int:
    if "out_len" in seg:
        return max(0, int(seg["out_len"]))
    return max(0, int(seg.get("end", 0)) - int(seg.get("start", 0)))


def fallback_ledger_from_trace(sid: str, total_frames: int,
                               trace_record: Mapping[str, object] | None) -> RouteLedger:
    """Build a traced-local ledger for a PG-RAST-style fallback output."""

    counts: Counter[str] = Counter()
    for seg in (trace_record or {}).get("segments", []):
        src = str(seg.get("sid", seg.get("source", "")))
        if not src:
            continue
        counts[src] += segment_len(seg)
    traced = min(int(total_frames), int(sum(counts.values())))
    unknown = max(0, int(total_frames) - traced)
    if sum(counts.values()) > traced and traced > 0:
        # Rescale rare over-traced cases so frame mass remains conserved.
        scale = traced / float(sum(counts.values()))
        counts = Counter({src: int(round(n * scale)) for src, n in counts.items()})
        delta = traced - sum(counts.values())
        if counts and delta:
            src0 = sorted(counts)[0]
            counts[src0] += delta
    return RouteLedger(
        sid=sid,
        total_frames=int(total_frames),
        local_unit_frames=traced,
        unknown_or_derived_frames=unknown,
        source_frames=dict(counts),
    )


def whole_clip_ledger(sid: str, total_frames: int, source_sid: str | None) -> RouteLedger:
    counts = {str(source_sid): int(total_frames)} if source_sid else {}
    return RouteLedger(
        sid=sid,
        total_frames=int(total_frames),
        whole_clip_frames=int(total_frames),
        source_frames=counts,
    )


def generated_ledger(sid: str, total_frames: int) -> RouteLedger:
    return RouteLedger(
        sid=sid,
        total_frames=int(total_frames),
        generated_frames=int(total_frames),
        source_frames={},
    )


def aggregate_ledgers(ledgers: Iterable[RouteLedger]) -> dict[str, object]:
    rows = list(ledgers)
    total = sum(r.total_frames for r in rows)
    whole = sum(r.whole_clip_frames for r in rows)
    local = sum(r.local_unit_frames for r in rows)
    gen = sum(r.generated_frames for r in rows)
    unknown = sum(r.unknown_or_derived_frames for r in rows)
    source_counts: Counter[str] = Counter()
    for r in rows:
        source_counts.update(r.source_frames)
    max_source = max(source_counts.values()) if source_counts else 0

    def pct(n: int) -> float:
        return 0.0 if total <= 0 else 100.0 * float(n) / float(total)

    return {
        "n_outputs": len(rows),
        "total_frames": int(total),
        "whole_clip_frames": int(whole),
        "local_unit_frames": int(local),
        "generated_frames": int(gen),
        "unknown_or_derived_frames": int(unknown),
        "charged_source_frames": int(whole + local + unknown),
        "whole_clip_frame_pct": pct(whole),
        "local_unit_frame_pct": pct(local),
        "generated_frame_pct": pct(gen),
        "unknown_or_derived_frame_pct": pct(unknown),
        "charged_source_frame_pct": pct(whole + local + unknown),
        "max_source_frames": int(max_source),
        "max_source_frame_pct": pct(max_source),
        "unique_sources": len(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "top_sources": source_counts.most_common(20),
    }


def choose_for_generated_target(candidates: list[CandidateSwap],
                                target_generated_frames: int) -> set[str]:
    """Greedy counterfactual swaps for a requested generated-frame target."""

    chosen: set[str] = set()
    generated = 0
    order = sorted(
        candidates,
        key=lambda c: (
            -(
                (c.generated_gain() + c.charged_reduction())
                * (1.0 + max(0.0, float(c.priority_bonus)))
            )
            / (1.0 + c.lexical_loss_proxy),
            c.sid,
        ),
    )
    for cand in order:
        if generated >= target_generated_frames:
            break
        if cand.generated_gain() <= 0:
            continue
        chosen.add(cand.sid)
        generated += cand.generated_gain()
    return chosen


def choose_for_banned_sources(candidates: list[CandidateSwap],
                              banned_sources: set[str]) -> set[str]:
    """Rewrite every output whose base route uses a banned source."""

    return {c.sid for c in candidates if c.base.has_any_source(banned_sources)}


def choose_for_source_cap(candidates: list[CandidateSwap],
                          base_ledgers: Mapping[str, RouteLedger],
                          cap_frames: int) -> set[str]:
    """Greedily rewrite outputs until no traced source exceeds cap_frames."""

    chosen: set[str] = set()
    cand_by_sid = {c.sid: c for c in candidates}

    while True:
        active = [
            cand_by_sid[sid].replacement if sid in chosen else ledger
            for sid, ledger in base_ledgers.items()
        ]
        agg = aggregate_ledgers(active)
        over_sources = {str(src) for src, n in agg["source_counts"].items() if int(n) > int(cap_frames)}
        if not over_sources:
            break

        best_sid = None
        best_score = 0.0
        for cand in candidates:
            if cand.sid in chosen:
                continue
            reduction = cand.source_reduction(over_sources)
            if reduction <= 0:
                continue
            score = (
                reduction * (1.0 + max(0.0, float(cand.priority_bonus)))
            ) / (1.0 + cand.lexical_loss_proxy)
            if score > best_score or (score == best_score and (best_sid is None or cand.sid < best_sid)):
                best_sid = cand.sid
                best_score = score
        if best_sid is None:
            break
        chosen.add(best_sid)

    return chosen
