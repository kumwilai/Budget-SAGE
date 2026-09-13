Budget-SAGE compact verification records

This directory contains summary-level protocols, hashes, result records and
audit reports. Per-item certificate manifests and corpus-derived annotation
tables are deliberately excluded from the public tree; authorized dataset
holders can regenerate or verify them with the source under `scripts/` and
their licensed local assets.

The records cover PHOENIX routing and reconstruction, the reduced CSL-Daily
port, recent-method artifact admission, and the bounded raw-RGB decision-
equivalence check. They do not redistribute PHOENIX-2014T, CSL-Daily, SLRTP,
model weights, pose banks, private keys, or participant data.

Run `python -B verify_release.py` from the repository root for the public,
dataset-free integrity and evidence gate.
