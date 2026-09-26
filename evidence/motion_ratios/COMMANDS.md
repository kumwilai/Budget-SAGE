# Motion ratios (definition K), test split, 641 clips, 2026-09-26

Each file was produced by `scripts/exp_revision_motion_ratios.py`. The reference is
`external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt`.

    python scripts/exp_revision_motion_ratios.py --pred_pt outputs/reviewer_closure_takeover_20260912/hybrid_signjepa/budget_sage_signjepa_hybrid_test.pt --gt_pt <reference> --out_json hybrid_fixed_test_kinematics.json
    python scripts/exp_revision_motion_ratios.py --pred_pt outputs/hybrid_learned_route_20260915/learned/budget_sage_signjepa_hybrid_learned_test.pt --gt_pt <reference> --out_json hybrid_learned_test_kinematics.json
    python scripts/exp_revision_motion_ratios.py --pred_pt outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt --gt_pt <reference> --out_json all_generated_test_kinematics.json

The archive routes use the same script, with outputs in `outputs/revision/{revision_clean_phrase_lattice,clean_rerank_frame40,clean_strict_cac_frame40,revision_clean_topk}_test_kinematics.json`.

Duration ratios in Table IV are the evaluator's `avg_duration`, read from each route's `evaluator_workspace/results/*.json`.

These values were reproduced exactly from the values recorded on 2026-09-26. Fable ruling: use K only (native length, median over clips, hand joints 8:50).
