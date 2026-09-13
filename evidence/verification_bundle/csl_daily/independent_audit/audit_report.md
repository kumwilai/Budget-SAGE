# Independent CSL audit: PASS

- `sealed_release()` passed. Freeze SHA256 `3031ebfec31aaebe6cb9b145cdfecb9e2040e44250ab7a8d91b8d8d30b17e608`; seal SHA256 `1d2f6fa96b4e8983b84e4c1140ada6f77cbbab5c11ba59a6463e1e66460fd7b3`.
- Seal precedes all four test starts: seal `1788706687388819749`; starts local `1788706765410716851`, whole `1788706765411447979`, fixed-first20 `1788715467672433282`, learned-first20 `1788715589797668199`.
- Independent `reduced_learned_csl_route.py analyze` completed into this directory. Prediction JSONs and `first20_identity.json` are byte-identical to v3; all non-time analysis fields, all-head metrics, identity, evidence, and bootstrap point/CI/group/replicate hashes match. Only `completed_unix_ns` differs. Peak RSS: 1,366,792 kB.
- Focused command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kumwilai/research/coopns-slr/.venv/bin/python -m pytest -q tests/test_csl_mska_eval_adapter.py tests/test_reduced_learned_csl_route.py` -> `161 passed in 12.47s`; adversarial subset -> `4 passed in 1.84s`.
- Amendment SHA256 `a64589c5d454ce757687828070e7d6ad1cc8361217fe52ade2b4f34e15acb8a2`; every declared file hash matched. Live route vs freeze snapshot diff is limited to declared test-evaluator relocation/API handling.

Residual risk/claim ceiling: runtime third-party environment is not hashed; this supports sealed-evidence compatibility and the stated reduced CSL result only, not new efficacy, untouched-test, full-method-transfer, or human-intelligibility claims. Follow-up: retain this audit and do not modify sealed artifacts.
