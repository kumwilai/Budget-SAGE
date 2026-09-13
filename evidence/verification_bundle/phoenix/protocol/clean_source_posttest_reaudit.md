# Clean-source post-test re-audit — FAIL (scoped)

PASS: `sha256sum outputs/revision/clean_source_test_protocol.json` reproduced `412f5b29…3cb7f3f`; its 17:15:58 freeze precedes all test route/materialization/evaluator/bootstrap mtimes. Official `test.pt` has 641 IDs. Independent comparison found query/trace/output IDs, order, and text exactly equal official 641 (ID SHA256 `c8702327…979ec0a`). Local manifest has 642 rows; only extra is `13December_2010_Monday_tagesschau-6940` and was not used.

PASS: in-memory reconstruction from the archive exactly matched both saved route tensors (641/641, max error 0), all tensors finite; route masks/traces/IDs align. Integer constraints reproduce: primary 19,693/49,254 = 39.982539% (used 85,227 <= 85,270); rerank 18,898/47,254 = 39.992382% (85,252 <= 85,270). Ledger recheck: primary/rerank unknown mass 0, source mass equals 49,254/47,254, every clip and blend conserved, no out-of-bounds intervals. The stale local length 91 for `14June_2010_Monday_tagesschau-3705` is safely overridden by archive/official tensor length 82; observed interval is [57,81).

PASS: route metadata has no evaluator outputs; baseline `.npz` selects only `rerank_score`. In-memory evaluator metrics match saved JSON. Seed-30373 paired bootstrap reproduces all requested point estimates/CIs and orientation.

FAIL for any blanket “all donors are nonexact” claim: 9 query outputs in each route use normalized exact-caption training sources through fallback (e.g. `guten abend liebe zuschauer` → `01August_2011_Monday_heute-4848`); canonical planned donors are nonexact and excluded. Fallback also omits 17/3,968 planned gloss tokens (3,951 emitted), with 26 single-source assemblies. Sources remain train-only and ledger-traced.

Follow-up: either globally exclude every normalized duplicate-caption source before final claims, or explicitly limit claims to canonical-donor exclusion and disclose these 9 overlaps/17 omissions. Do not call this pristine leakage-free evaluation until resolved.
