# Independent bounded review — certificate re-verification v2

Verdict: **PASS**.

- The membership-drill correction re-signs the edited envelope before
  verification and counts success only for entry/membership errors.
- `ephemeral_all_arms_v2_comparison.json` reports 8/8 arms passing and 16/16
  recorded cells matching per arm. Authenticated manifest bodies match after
  excluding MACs; reports match after excluding the expected manifest and
  implementation hashes.
- A direct recorded/fresh `top1231` inspection confirms PASS, 641 entries,
  11/11 drills, 49,374 checked positions, zero withdrawn exposure/mass, and no
  unknown frames.
- The inspected verifier access log contains no issuer routing file.
- The key limitation is explicit: this is fresh ephemeral-key re-issuance, not
  original-HMAC byte reproduction.

Residual risk: the automated forbidden-filename marker set is not exhaustive,
although direct inspection of the selected access log found no issuer record.

Reviewer: distinct Luna tester, read-only bounded follow-up. Runtime model
identity was not independently observable.
