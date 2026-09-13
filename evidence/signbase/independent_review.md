# Independent Sign-Base result review — 2026-09-12

Verdict: **PASS for the bounded claim below.**

Admissible claim: **hashed locally trained Sign-Base stored bank, inference seed
11, train-only text-predicted duration**. The evidence does not establish an
HFUT-LMC author-system reproduction, multi-seed robustness, or independently
reproducible generation execution.

The reviewed audit verifies the checkpoint hash, all 641 official test keys,
the non-ground-truth duration contract, precise motion definitions, the
organizers' ground-truth row to published precision, and a byte-identical fresh
CPU re-score of the retained result JSON. It separately records training seed
27 and inference seed 11. The processor device shim and full original producer
shell command were not preserved, so the generated bank is admitted only as a
stored result with audited checkpoint, inputs, coverage, and scoring.

Residual scope note: field-by-field stored-versus-rescore comparison selects
the manuscript-retained metrics rather than BLEU-2, BLEU-3, chrF, and ROUGE.
This is non-blocking because the complete result JSONs are byte-identical and
the manuscript reports only the retained fields.
