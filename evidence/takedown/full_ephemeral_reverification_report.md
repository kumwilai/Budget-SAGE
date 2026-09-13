# Eight-arm certificate re-verification — 2026-09-12

## Verdict

**PASS for scientific values and verifier behavior: 8 of 8 arms reproduced.**
Every one of the 16 recorded cells per arm matches. Authenticated manifest
bodies match after excluding only key-dependent MAC bytes. Full verification
reports match after excluding only the expected new-manifest hash and the
versioned verifier-script hash.

**Authentication-byte limitation:** the original authorized environment key
was absent. This run used one fresh ephemeral key, generated inside the launcher
process and never printed, persisted, or passed in argv. It re-issued valid
certificates but cannot and does not claim byte-identical original HMAC tags.

## Cell-by-cell result

Every value below is identical to the recorded value.

| Arm | Verdict | Entries | Exact | Untraced | Issuer records read | Budget | Withdrawn | In archive | Positions checked | Exposed | Mass | Drills |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_empty | PASS | 641 | 641/641 | 0 | false | 98465 <= 98508 | 0 | 0 | 49254 | 0 | 0.0 | 11/11 |
| top10 | PASS | 641 | 641/641 | 0 | false | 96930 <= 96956 | 10 | 10 | 48478 | 0 | 0.0 | 11/11 |
| top50 | PASS | 641 | 641/641 | 0 | false | 96980 <= 97014 | 50 | 50 | 48507 | 0 | 0.0 | 11/11 |
| top250 | PASS | 641 | 641/641 | 0 | false | 96595 <= 96642 | 250 | 250 | 48321 | 0 | 0.0 | 11/11 |
| top1231 | PASS | 641 | 641/641 | 0 | false | 98730 <= 98748 | 1231 | 1231 | 49374 | 0 | 0.0 | 11/11 |
| Signer01 | PASS | 641 | 641/641 | 0 | false | 99240 <= 99252 | 1862 | 1852 | 49626 | 0 | 0.0 | 11/11 |
| Signer05 | PASS | 641 | 641/641 | 0 | false | 97195 <= 97210 | 1629 | 1623 | 48605 | 0 | 0.0 | 11/11 |
| rand250_s1 | PASS | 641 | 641/641 | 0 | false | 98830 <= 98846 | 250 | 250 | 49423 | 0 | 0.0 | 11/11 |

## Two controls

- `gate_empty`: 641/641 identical certificate ID ordering, authenticated pose
  descriptors, authenticated ledgers, and bit-identical emitted tensors against
  the frozen primary release. All four envelope fields and the base policy are
  identical; the only policy addition is the authenticated empty withdrawal set.
- `top1231`: the withdrawal set is exactly equal to the 1,231 sources used by
  the frozen primary release. All 641 outputs still emit and reconstruct with
  zero positions and zero mass drawing on those withdrawn sources.

## Verifier input boundary

The audit hook was re-run for every arm. Each verifier opened its emitted pose
bank, newly issued manifest, and committed archive. None opened an issuer mask,
retrieval trace, fallback trace, standalone ledger, ban set, or arm audit. This
conclusion comes from the recorded open-event lists rather than the report's
`router_records_read_by_verifier` flag.

## Tamper-drill reason audit and correction

The first fresh run reproduced 11/11 Boolean drill results, but reason inspection
found one defect: `release_membership_edit_caught` altered membership without
re-signing the release, so it was rejected early as `release HMAC failed` rather
than by the entry/membership guard. The same synthetic mutation, after re-signing,
was rejected as `release entries are malformed`; on a nonempty real release the
corresponding guard is `manifest membership or pose order differs`.

The verifier was version-corrected to re-sign the edited envelope and count this
drill only when one of those entry/membership errors is reached. No stored
certificate, split, configuration, or prior result was modified. The focused
reason tests passed 5/5; the complete certificate test file passed 92/92. All
eight real arms were then re-issued and re-verified under the corrected code;
each again rejected 11/11 drills.

## Commands

```text
/home/kumwilai/research/coopns-slr/.venv/bin/python -B -m pytest -q tests/test_clean_route_certificate.py
/usr/bin/time -v /home/kumwilai/research/coopns-slr/.venv/bin/python -B outputs/takedown_certificates_reverify_2026-09-12/run_ephemeral_full_reverify.py
/home/kumwilai/research/coopns-slr/.venv/bin/python -B outputs/takedown_certificates_reverify_2026-09-12/compare_ephemeral_all_arms.py --fresh-root outputs/takedown_certificates_reverify_2026-09-12/ephemeral_key_all_arms_v2_reason_corrected --out outputs/takedown_certificates_reverify_2026-09-12/ephemeral_all_arms_v2_comparison.json
```

Corrected run: exit 0; wall time 6:38.26; peak RSS 9,470,788 kB; swap 0.

## Hashes

- comparison: `5086b5d2fb6c77d2e9277ba50ec2fca7f0a58bbbc61bdb2fc6eef4f3ac2b1545`
- fresh provenance: `b08504c320ca9fb47bbc08e4a3a04456a5591258c3fb1eb142b7ad4e964950db`
- comparison script: `0f8bc7b8866f72f36f864bb2ba35a227d18140b7c00514a805589b6e6da7459d`
- corrected verifier: `bbab96fcd7b0e25a70a6b6dd738d078e302f921f5b98e07dff9a3464adfb1a11`
- certificate tests: `dff79a3a420ea39539291b89de40208155123468b3e7b63b992bf62d92b7326b`

## Plain conclusion

The certificate and source-withdrawal scientific evidence reproduces under
fresh independent re-issuance: **yes, 8/8 arms**, including both named controls.
Original authentication-tag byte identity is untested because the original
authorized key was unavailable.
