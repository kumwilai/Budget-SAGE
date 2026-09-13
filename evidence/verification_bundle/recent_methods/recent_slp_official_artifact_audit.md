# Recent-SLP common-protocol artifact update

Checked 2026-09-07 from official project repositories, project pages, model
cards, and released local checkouts. This record distinguishes runnable source
code from a validated model/output release. It does not infer that an unlinked
private artifact is absent.

## Admission rule

A numerical row can enter the common protocol only when all of the following
are fixed: producing checkpoint or author-indexed outputs; exact request text
or gloss and output-duration policy; item-to-output mapping; native pose
definition and frame rate; deterministic conversion to the common 178-joint
SLRTP representation; and the same frozen evaluator used for Budget-SAGE.
Conversions that use reference duration, target poses, target face rows, or
test-tuned calibration are inadmissible as deployable comparisons.

## Official-source status

| Method | Official material confirmed | Decisive requirement not yet satisfied |
| --- | --- | --- |
| G2P-DDM | [Official code](https://github.com/Anynoumsiccv9970/G2P-DDM) documents separate Pose-VQVAE and discrete-diffusion training and an inference command. | A trained VQ-VAE/generator or indexed PHOENIX bank was not identified or accessible in the inspected official sources; no validated 178-joint adapter was identified. |
| Sign-IDD | [Official code](https://github.com/NaVi-start/Sign-IDD) gives PHOENIX train/test commands and points to a separate back-translator. | A generator checkpoint/indexed output bank and exact duration/item contract were not identified or accessible in the inspected official sources. The local historical conversion cannot substitute because its producing checkpoint is not present locally and it used target-derived duration/face/calibration inputs. |
| Sign-D2C | The [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.pdf) defines transition inpainting from observed boundary poses. | It is not a full text/gloss-to-utterance generator; weights and an official 641-request boundary policy were not identified or accessible in the inspected official sources. It belongs in a transition-component track, not the full-generator ranking. |
| SOKE | [Official code](https://github.com/2000ZRL/SOKE) releases the SMPL-X pipeline, normalization files, and tokenizer checkpoint. | An autoregressive-generator checkpoint/indexed outputs and a validated SMPL-X-to-SLRTP-178 conversion were not identified or accessible in the inspected official sources. The tokenizer alone is not the generator. |
| FGDM | [Official code](https://github.com/yuyiheng-eu/FGDM-main) now provides CVPR 2026 training/testing and PHOENIX preparation; it documents an OpenPose-to-3D lift and a Sign-IDD back-translator bundle. | No released trained FGDM checkpoint or indexed generated bank is identified by the official README; preprocessing/lift/item mapping therefore cannot yet be frozen for a common row. |
| SignSparK | The [official repository](https://github.com/JianHe0628/SignSparK), [model card](https://huggingface.co/LionelLow/SignSparK), and [data card](https://huggingface.co/datasets/LionelLow/SignSparK_data) release three approximately 1.4B-parameter stream checkpoints and 12.7 GB of CSL-Daily/How2Sign SMPL-X LMDB data. The official sampling configuration includes `eval.keyframes_mask_all=True`, so a no-keyframe mode is documented. | The repository still marks back-translation evaluation weights as coming soon, and no validated SMPL-X-to-common-evaluator conversion is available here. The present restricted runtime cannot retrieve the external binary bundles, so their hashes, coverage, and executable behavior cannot be checked locally. |
| MSKA-SLT evaluator | The [official MSKA README](https://github.com/sutwangyan/MSKA/blob/main/README.md) links a CSL-Daily SLT checkpoint, and the [official configuration](https://github.com/sutwangyan/MSKA/blob/main/configs/csl-daily_s2t.yaml) declares the translation and pruned-vocabulary assets. | The linked Google Drive checkpoint is not present locally and returned an authorization failure to the current browser. The local 4.29 GB `step_1000.ckpt` is a G2T initializer, not the final SLT row. The exact official `old2new_vocab.pkl`/complete final bundle must reproduce the published ground-truth-input BLEU before pose-to-Chinese scores are admissible. |

## Executed common-protocol evidence

In this audit, the organizer-released Progressive Transformer bank is the only
external pose bank admitted for a stored-release-bank re-score. All 641 items
were evaluated with the declared `--fps 12` setting in the frozen SLRTP evaluator:
BLEU-1 16.9147, BLEU-4 1.5942, WER 101.6788, and DTW-MJE 0.0449293. A ground-
truth identity anchor and the deliberately wrong 25-fps negative control are
retained in `outputs/baseline_protocol/provenance.json`. Producer execution and
native-rate provenance are not authenticated by the stored bank.

## Quarantined local controls

The stored CSL-Daily AR-pose, continuous-diffusion, VQ-AR, and masked-token
controls are not reproductions of Progressive Transformer, Sign-IDD, SOKE, or
G2P-DDM. An independent code-and-artifact audit found that they omit defining
mechanisms of the named systems. It also measured batch-horizon dependence in
the AR and VQ exports, unseeded continuous-diffusion inference, out-of-range
confidence channels, training on 18,401 rather than the admitted 18,400
sources, and per-result best-head selection in the historical evaluator. Their
aggregate WER values therefore remain historical failure evidence and are not
admitted as recent-method comparison rows. Repairing those controls would make
them study-specific mechanism controls, not faithful named-method reruns.

## Closure boundary

The comparison *protocol, admission criteria, one complete external release-
bank row, rejected-control evidence, and method-specific blockers* are
reproducible. A claim that recent generators were rerun is not currently
supported. Closing that empirical request requires either faithful training
under a sealed protocol or the named public/author artifacts and a common
conversion; it cannot be manufactured from literature tables, a tokenizer, or
generic architectural analogues. Author contact is an external communication
and has not been made.
