# Sign-Base PHOENIX SLRTP test audit

Date: 2026-09-12. This report audits the completed processor-only prediction
bank. It does not claim a completed GPU run.

## Verdict

The hashed 641-item bank is admissible as **our locally trained Sign-Base
instance, training seed 27, inference seed 11, training-only text-predicted
duration**. It is not ground-truth-duration conditioned. It is not an
authenticated reproduction of HFUT-LMC's challenge submission, a multi-seed
robustness estimate, or a fully reproducible generation execution: the CPU
runtime device shim and full producer shell command were not preserved.

The stored bank and its scoring are independently checkable. The independent
CPU evaluator re-score is byte-identical to the original result JSON.

## Results on 641 test items

| result | BLEU-1 | BLEU-4 | WER | DTW-MJE |
|---|---:|---:|---:|---:|
| organiser ground-truth row | 34.396394 | 12.777149 | 85.774702 | 0.000000 |
| reproduced ground-truth control | 34.396394 | 12.777149 | 85.774702 | 0.000000 |
| locally trained Sign-Base, seed 11 | 28.023406 | 8.114024 | 91.490452 | 0.038313 |
| independent CPU re-score | 28.023406 | 8.114024 | 91.490452 | 0.038313 |

The complete ground-truth row matches the organisers' retained values except
for BLEU-3 floating roundoff of `1.4210854715202004e-14`; the maximum absolute
difference is below `1e-12`. The two Sign-Base result JSON files have the same
SHA-256, `35b3ea17ceb3fe9d568cede38806e52fdded25e05b3aee6a0e20226b7834c0df`.

## Motion ratios relative to official test ground truth

| ratio | value |
|---|---:|
| duration | 1.091552 |
| hand speed | 1.212093 |
| body speed | 1.227280 |
| hand pose standard deviation | 0.598103 |
| hand jerk | 4.738737 |

Duration is the organiser evaluator's mean per-clip length ratio after common
frame-rate handling. Other rows are prediction-median divided by GT-median over
641 clips, using full xyz: hand joints 8:50 and body joints 0:8; speed is mean
first-difference L2 magnitude, jerk is mean third-difference L2 magnitude, and
pose standard deviation is temporal standard deviation averaged over hand
joints and channels.

## Model and inference contract

- Source commit: `a82e124a4ffdd9c73de423e75df31a5509fd9b9d`.
- Training used the official SLRTP training release, upstream filter and
  hyperparameters, 7,037 of 7,060 clips, seed 27.
- The only best checkpoint on disk is `44000_best.ckpt`, SHA-256
  `9544e272bd1b15eae34db555ff09db93627674290292bf811307dcf5709cd26c`.
  It achieved validation DTW 11.061; validations through step 58,000 did not
  improve it. Strict loading was verified by tensor equality and a parameter
  norm of 54.2457847595 versus about 40.46 for fresh initialisations.
- Generation consumed the official German test text verbatim, the
  training-built vocabulary, a zero target scaffold whose length came from the
  frozen text-only duration regressor, and diffusion noise under inference
  seed 11. It consumed no test pose coordinates, face rows, keyframes,
  retrieval result, or evaluator output.
- The duration policy was fitted on training data only and reused without
  refitting, SHA-256 `38a192cbb297572c86a7c1e1590773c4fbe7f966bd77e31c7065f97681effc82`.
  Its 641 supplied lengths equal all generated lengths but equal only 15 test
  GT lengths; versus GT their MAE is 15.7785 frames and Pearson correlation is
  0.8615. Ground truth was opened only later for diagnostics and scoring.
- The bank has exactly the official 641 keys, no extras or omissions, every
  tensor has shape `[T,178,3]`, and the harvested bank is tensor-exact to the
  processor shadow bank. Prediction-bank SHA-256:
  `fdbe08061c27fa7b85dd6c6c4ad5079fe3848c16c36c2231dab367457ebc243a`.
- The earlier GPU attempt produced no prediction bank. There is therefore no
  GPU/CPU result pair to compare; the only completed bank is processor-only.

The upstream warm-start line `Can't find checkpoint in directory None` occurs
after strict loading, when the test-only training manager checks the empty
shadow output directory. It does not replace the loaded checkpoint.

## Commands

Mandatory GPU access gate, repeated in the active Codex session:

```text
/home/kumwilai/research/coopns-slr/.venv/bin/python -B -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

Result: `False 0`; `nvidia-smi` was blocked by the operating system. Per the
protocol, no new GPU inference was attempted.

The surviving production log proves the explicit checkpoint argument but does
not retain its complete shell command or CPU device-shim source:

```text
[driver] invoking test() with explicit --ckpt=/home/kumwilai/research/signgen-t2m/outputs/signbase_slrtp_2026-09-11/model/44000_best.ckpt
```

Independent CPU re-score of the immutable bank:

```text
CUDA_VISIBLE_DEVICES='' /home/kumwilai/research/coopns-slr/.venv/bin/python -B /home/kumwilai/research/signgen-t2m/external/SLRTP-Sign-Production-Evaluation/main.py /home/kumwilai/research/signgen-t2m/outputs/signbase_slrtp_2026-09-11/eval_20260912/preds/signbase_test_sampleseed11.pt /home/kumwilai/research/signgen-t2m/external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt /home/kumwilai/research/signgen-t2m/external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model --tag signbase_seed11_cpu_rescore --fps 25
```

Audit and motion-ratio computation:

```text
CUDA_VISIBLE_DEVICES='' /home/kumwilai/research/coopns-slr/.venv/bin/python -B scripts/audit_signbase_eval.py --repo /home/kumwilai/research/signgen-t2m --out /home/kumwilai/research/signgen-t2m/outputs/signbase_slrtp_2026-09-11/eval_gpu_2026-09-12/audit_existing_cpu/audit.json
```

The first re-score launch failed before process creation because the declared
working directory did not yet exist; it wrote no result. The directory was
created and the absolute-path command above succeeded. No workaround changed
the bank or evaluator.
