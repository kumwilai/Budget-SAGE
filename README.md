# Budget-SAGE

Repository: <https://github.com/kumwilai/-Budget-SAGE>

Official code and reproducibility record for **Verifiable Source Routing for
Provenance-Governed Text-to-Sign Generation** by Wuttipong Kumwilaisak and
C.-C. Jay Kuo.

Budget-SAGE combines provenance-aware archive routing with a Sign-JEPA
generated-motion escape. The evaluated within-sequence hybrid uses generated
motion at recorded archive joins, applies a fixed quintic bridge, and records
fractional archive/generated source mass. The repository also contains the
cooperative HMAC certificate verifier, source-withdrawal checks, and the keyed
generated-motion watermark implementation.

## What this release supports

- deterministic routing and exact frame-budget admission;
- archive-source ledgers and cooperative reconstruction certificates;
- source-withdrawal re-issuance and tamper rejection;
- the frozen Budget-SAGE/Sign-JEPA join-hybrid implementation;
- a generated-motion watermark and its frozen detector records;
- compact PHOENIX-2014T, CSL-Daily, recent-method and raw-RGB audit records.

The command below verifies every released byte against `MANIFEST.sha256` and
checks the headline machine-readable evidence:

```bash
python -B verify_release.py
```

Expected final line:

```text
PASS: release integrity and declared evidence checks
```

For the focused source-only regression gate:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-verification.txt
python -m pytest -q \
  tests/test_budget_sage_signjepa_hybrid.py \
  tests/test_clean_route_certificate.py \
  tests/test_route_certificate_manifest.py
```

## Layout

- `budget_sage/`: reusable routing, generation and certificate modules.
- `scripts/`: exact research scripts retained for the reported routes and
  audits.
- `tests/`: focused deterministic and failure-case tests.
- `evidence/hybrid/`: preregistration, materialization, seam audit, score and
  independent review for the frozen join hybrid.
- `evidence/takedown/`: independent eight-arm withdrawal re-verification.
- `evidence/watermark/`: frozen watermark and detector records.
- `evidence/signbase/`: controlled Sign-Base scoring audit.
- `evidence/verification_bundle/`: compact archived verification records.

## Data and checkpoints

PHOENIX-2014T, CSL-Daily, the SLRTP evaluator, pretrained recognizers, pose
archives and model checkpoints are not redistributed. Obtain them from their
respective custodians and follow their licenses. Released manifests preserve
hashes and input contracts so authorized holders can check local copies.

## Human evaluation instrument

The web instrument used for the human evaluation is published separately, so
that the study materials stay distinct from this code release:

- instrument source: <https://github.com/kumwilai/sign-eval>
- live instrument: <https://kumwilai.github.io/sign-eval/>

That repository contains the participant-facing evaluation page only. No
participant response, identifier, access token, or collected result is
published there or here. Human-evaluation results are not reported in this
release; automatic recognition and trajectory metrics in this repository are
not evidence of human sign-language intelligibility.

## Claim boundaries

This release reproduces the current artifact-level routing, reconstruction,
verification and reported-evidence checks. It does not recreate missing
historical pose producers, continuous historical posterior arrays, or old
recognizer training. HMAC certificates provide authorized cooperative
integrity, not public nonrepudiation. Automatic recognition and trajectory
metrics are not evidence of human sign-language intelligibility.

## Version and citation

Prepared release: `v1.0.0` (2026-09-13). See `CITATION.cff`.

Released under the MIT License; see `LICENSE`. Dataset annotations, checkpoints,
third-party evaluators and third-party code remain governed by their original
licenses and are not relicensed here.
