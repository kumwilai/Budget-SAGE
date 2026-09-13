# Public release checklist

- [ ] Confirm the GitHub repository owner and final URL.
- [ ] Select and add the software `LICENSE`; do not infer it.
- [ ] Run `python -B verify_release.py` and retain the PASS output.
- [ ] Run the focused pytest command from `README.md`.
- [ ] Confirm the secret scan reports no credential-like material.
- [ ] Confirm no human-evaluation response, token, protocol, or participant file is present.
- [ ] Confirm no dataset, checkpoint, pose bank, private key, `.env`, or credential file is present.
- [ ] Commit exactly the prepared release tree.
- [ ] Push the default branch.
- [ ] Create signed or annotated tag `v1.0.0`.
- [ ] Create GitHub release `v1.0.0` and attach `MANIFEST.sha256`.
- [ ] Connect the repository to Zenodo and mint a DOI for `v1.0.0`.
- [ ] Add the final GitHub URL and DOI to `README.md`, `CITATION.cff`, and the manuscript availability statement.
