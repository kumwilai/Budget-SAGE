# Independent refresh of the two certificate controls — 2026-09-12

## Scope boundary

The authorized `BUDGET_SAGE_MANIFEST_KEY` was absent, so the eight-arm
authorized-key reproduction remains **0/8**. These controls use fresh ephemeral
keys generated inside launcher processes and never printed or persisted. They
re-derive data-path behavior and equality invariants; they do not reproduce the
original authentication tags.

## Control 1: `gate_empty` versus frozen primary

Command:

```text
/usr/bin/time -v /home/kumwilai/research/coopns-slr/.venv/bin/python -B outputs/takedown_certificates_reverify_2026-09-12/rederive_controls.py --out outputs/takedown_certificates_reverify_2026-09-12/gate_empty_control_rederived.json
```

| Cell | Recorded | Re-derived | Match |
| --- | ---: | ---: | --- |
| Certificate IDs | 641 identical | 641 identical | yes |
| Authenticated pose descriptors | 641/641 | 641/641 | yes |
| Authenticated ledgers | 641/641 | 641/641 | yes |
| Emitted pose tensors | 641/641 bit-identical | 641/641 bit-identical | yes |
| Archive commitment | identical | identical | yes |
| Archive ID | identical | identical | yes |
| Detector configuration | identical | identical | yes |
| Manifest key ID | identical | identical | yes |
| Base policy | identical | identical | yes |
| Added policy field | empty withdrawal set only | empty withdrawal set only | yes |
| Schema transition | v1 to v2 | v1 to v2 | yes |

Runtime: 1.41 s; peak RSS 858,716 kB; no swap.

## Control 2: `top1231`

The command used the same sequential issue/`verify-probe` launcher recorded in
`task_checkpoint.md`, substituting the frozen `top1231` input paths and writing
only to `synthetic_key_top1231_control/`. Its ephemeral key was generated in
memory and was neither printed nor persisted.

| Cell | Recorded | Re-derived | Match |
| --- | ---: | ---: | --- |
| Verdict | PASS | PASS | yes |
| Entries | 641 | 641 | yes |
| Exact reconstruction | 641/641 | verification PASS over all 641 | yes |
| Unknown/untraced frames | 0 | 0 | yes |
| Budget check | `98730 <= 98748` | `98730 <= 98748` | yes |
| Withdrawal-set size | 1,231 | 1,231 | yes |
| Withdrawn sources in archive | 1,231 | 1,231 | yes |
| Withdrawn set equals all sources used by frozen primary | yes | yes (set equality, 1,231/1,231) | yes |
| Emitted positions checked | 49,374 | 49,374 | yes |
| Positions drawing on withdrawn source | 0 | 0 | yes |
| Withdrawn mass | 0.0 | 0.0 | yes |
| Tamper drills rejected | 11/11 | 11/11 | yes |
| Issuer routing records opened by verifier | none | none in audit-hook log | yes |

Runtime: 54.92 s; peak RSS 9,424,996 kB; no swap.

## Hashes

- comparison script: `2c26e9114280439b49ff1a49ec18ca1faa5e1b889f5fdde5e4e8a2f9569e77ae`
- `gate_empty_control_rederived.json`: `24eb1eae8ed53fa8eebde8c644de01df3fad131fab514402c1f400b7e7b933ec`
- `top1231/manifest.json`: `7761fc7aefb363ef440eee4f37fcc2a04ffd54def24cfab8494c26399e111b3a`
- `top1231/verification.json`: `3ebc95bdb2046c5222a534382716bbc93ad8ba34aab52e6c037989a582e5b4f1`
- `top1231/verifier_file_access.json`: `5736ee9af47aac5e1a8db1969305c6b075e29bc651adf81a8387f2f4078691e1`

## Verdict

Both named controls reproduce at the data-path and verifier-behavior level.
The full requested evidence package does **not** yet reproduce independently:
the authorized-key eight-arm run remains 0/8 because its key is unavailable.

A distinct read-only reviewer was dispatched to audit these two refreshed
controls but did not return evidence within the bounded window and was stopped
to avoid open-ended token use. Therefore this report records a fresh primary
re-derivation, not a completed second-reviewer sign-off.
