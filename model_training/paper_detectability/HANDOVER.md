# Handover — ICCCIT-2027 detectability paper

**For:** the next agent picking this up cold.
**Written:** 2026-09-23. **Status:** three hypotheses tested, two falsified, one decisive
experiment still unrun.

Read this before `METHODOLOGY.md` — that document was written *before* the experiments ran
and its central hypothesis has since been overtaken by data. Sections 1-2 of it (positioning
vs. prior art) are still correct and still load-bearing. Section 2's H1/H2 are not.

---

## 1. What the goal is

Find something in this project that is **genuinely publishable** and **inside ICCCIT-2027's
scope** (Track-1: Signal and Image Processing / Machine Learning). It does not have to be part
of the original project plan — being applicable to the rover is enough. It does **not** have to
be running on the robot today; "publishable" was explicitly decoupled from "deployed".

The user leads on direction and has final say. They have repeatedly and correctly rejected
proposals that were not real research questions. Do not pitch anything until you can state its
research question, its falsifiable hypothesis, and the measurement that would kill it.

## 2. The asset everything is measured against

| item | value |
|---|---|
| Detector | YOLO26-n, **13 classes**, mAP50 **0.5508**, mAP50-95 **0.3970** |
| Weights | `model_training/models/circe_yolo26n_13cls_100ep/best.pt` (Git LFS) |
| Training | 100/100 epochs, complete, not resumed mid-flight |
| Test split | `model_training/merged_v2/test/` — **5,565 images**, split membership pinned by manifest |
| Classes | concrete_crack, corrosion, fluid_patch, fire, smoke, gauge_face, efflorescence, exposed_rebar, spalling, missing_bolt, bolt_ok, bolt_defective, bolt_corroded |

The detector is **frozen**. Every experiment below characterises a given detector; none of them
retrain one. Keep it that way unless the paper direction changes — retraining invalidates
cross-experiment comparison.

## 3. What has actually been measured

### Experiment 1 — resolution hypothesis → **FALSIFIED**

The obvious story was "diffuse defects fail because they're under-resolved." Measured the
median bounding-box footprint per class against per-class mAP50. It inverts:

| class | median footprint | mAP50 |
|---|---|---|
| efflorescence | **660 px** | 0.209 |
| missing_bolt | **31 px** | 0.455 |

The *largest* objects score worst and the *smallest* score well. Resolution is not the binding
constraint. **Do not resurrect this without new evidence** — it is the first thing that sounds
right and it is wrong.

### Experiment 2 — recognition vs. localisation → ran, then **inverted by its own control**

`iou_sweep.py` (`CONF = 0.25`, the operating threshold) measures per-class recall as the IoU
matching threshold sweeps 0.1 → 0.7. Logic: high recall at IoU 0.1 collapsing by 0.5 means the
model *finds* the defect but cannot delineate it; already-low recall at 0.1 means it genuinely
never detects it.

Results in `iou_sweep_results.txt`. Recall at IoU 0.1 was low for the hard classes
(corrosion 0.159, efflorescence 0.175, smoke 0.337, concrete_crack 0.350), which reads as a
**recognition failure**.

**Then the control killed that reading.** `iou_sweep_lowconf.py` is the identical sweep at
`CONF = 0.001`. If recall stayed low, genuine recognition failure. It did not:

| class | r@IoU0.1, conf 0.25 | r@IoU0.1, conf 0.001 | factor |
|---|---|---|---|
| corrosion | 0.159 | **0.865** | **5.4×** |
| bolt_defective | 0.154 | **0.769** | **5.0×** |
| efflorescence | 0.175 | **0.841** | **4.8×** |
| smoke | 0.337 | **0.974** | 2.9× |
| concrete_crack | 0.350 | **0.931** | 2.7× |
| gauge_face | 0.999 | 1.000 | 1.0× |

**The model finds these defects. It just assigns them confidence below 0.25.** That is a
class-dependent **confidence-calibration** problem, not a capability problem. A single global
threshold silently discards most true detections for exactly the hardest classes.

### What survived the control: morphology ordering

The collapse ratio (recall@0.5 / recall@0.1) still orders by pre-registered morphology at
`conf=0.001`, just compressed:

```
texture/diffuse   efflorescence 0.792  exposed_rebar 0.850  concrete_crack 0.852
                  corrosion 0.857      spalling 0.873
compact/bounded   bolt_corroded 0.956  bolt_ok 0.963        gauge_face 1.000
```

So boundary ambiguity is **real but secondary**. At `conf=0.25` corrosion's ratio was 0.407;
at 0.001 it is 0.857. Most of the apparent delineation collapse was the confidence threshold,
not geometry. The morphology grouping in `METHODOLOGY.md` §6 is pre-registered — it was fixed
before results were inspected. Keep it that way.

## 4. The next experiment, and why it decides the paper

**Per-class precision-recall across confidence thresholds.**

Recall at `conf=0.001` proves the detections *exist*. It does **not** prove a usable operating
point exists — precision down there could be catastrophic. The question is whether a per-class
threshold genuinely beats the global one:

> For each class, does there exist a confidence threshold τ_c such that the per-class operating
> point (τ_c) dominates the global τ = 0.25 in recall at equal or better precision?

- If **yes** → the paper is *"class-dependent confidence miscalibration in multi-class defect
  detection, and per-class thresholds as the fix"*, with a directly actionable result for the
  rover: stop using one global threshold.
- If **no** (precision collapses as fast as recall rises) → the low-confidence detections are
  worthless, the finding degrades to an observation about score distributions, and the paper
  needs a different spine.

This is one inference sweep over 5,565 images, no retraining. **Run it before proposing
anything.** Model after `iou_sweep_lowconf.py` — same frozen weights, same test split, same
batching (`B = 8`, `device=0`).

## 5. Prior art already found — do not re-claim these

- **Zwick, Gerdts & Stütz**, *Sensors* 23(2):664, 2023, DOI `10.3390/s23020664` — builds a
  detector "sensor performance model" (AP vs ground sample distance and elevation angle) and
  optimises UAV trajectories with it. This is the performance-surface idea. **Ours is not the
  first.** Cite it.
- **Dollár et al.**, *IEEE TPAMI* 34(4):743-761, 2012 — miss-rate vs. object pixel height.
- **COCO small/medium/large AP** (Lin et al., ECCV 2014).
- **MIL-HDBK-1823A** — probability-of-detection (POD) methodology from non-destructive
  evaluation. A per-class detection-vs-size threshold *is* a vision POD curve; say so.
- **Scalix**, arXiv:2608.17553 — scale posteriors.
- **Paull et al.**, ICRA 2014 — probabilistic coverage.

What is *not* covered by the above: per-class calibration behaviour across a multi-morphology
defect taxonomy. That gap is where the current finding lives.

## 6. Things that will waste your time — all verified dead ends

- **Do not pitch reverse-engineering** (the DM002HW protocol work). Not research, and out of
  the current project scope.
- **Do not pitch streaming inference, remote offload, or video compression.** Explicitly ruled
  out by the user. The compression question in particular ("what bitrate before detection
  degrades?") was rejected as a bandwidth-engineering question, not a research question.
- **Do not assume `circe_v3` is design-only.** It is the current version and ~11 ROS 2 packages
  (~2,000 LOC: `circe_coverage`, `circe_mapping`, `circe_explore`, `circe_localization`,
  `circe_vggt_client`) implement it. A doc label saying "design stage" was stale.
- **Do not propose anything needing per-track measurements on the rover.** A proposal was built
  on that and collapsed — the rover-side tracking files do not exist.
- **Gemini's proposal** (`circe_v3/GEMINI_RESEARCH_PROPOSAL.md`, `GEMINI_METHODOLOGY.md`) has a
  **factual error at its core**: it claims VGGT-SLAM submaps relate by **Sim(3)** and builds a
  scale-invariance proof on that. VGGT-SLAM 2.0 aligns submaps by **SL(4) homographies** —
  15-DOF projective, not similarity. See `my_slam_vggt_omega/RESEARCH.md`. The scale factor does
  not simply cancel as claimed. Both files are saved verbatim with provenance headers for
  evaluation; neither is validated.

## 7. Known gap, unaddressed

Three bolt classes (`bolt_ok`, `bolt_defective`, `bolt_corroded`) carry **LLM-generated labels**
(Gemini 3-pass consensus + SAM3 box tightening), not human annotation. Measured inter-pass class
disagreement is **14-18%**. A 30-image human-review folder was built at
`dataset_annotation/circe_datasets/_label_review/` and **was never reviewed**. Any claim resting
on those three classes needs this closed or explicitly disclosed. `METHODOLOGY.md` §8 discloses
it; keep that disclosure.

## 8. Lost work — re-create if needed

Two completed ablations existed and **their code is gone** (not on `D:`, not in WSL, not in the
scratchpad — `dataset_annotation/paper_analysis/` is now an empty directory). Only their results
survive, in conversation history, and are therefore **unverified from source**:

- `ablation_consensus.py` — found the LLM annotator is geometrically stable (~90% cross-pass box
  agreement) but **semantically unstable (14-18% class disagreement)**; 289 single-pass boxes
  rejected by majority vote.
- `ablation_sam3.py` — SAM3 tightening shrank 30-34% of boxes, grew <5%, IoU(pre,post) 0.93-0.94.

Treat those numbers as **claims, not evidence**, until re-derived. If the paper leans on the
annotation pipeline, rewrite both scripts and re-run.

## 9. Files that matter

```
model_training/paper_detectability/
  METHODOLOGY.md                 formal methodology; §1-2 valid, §2 H1/H2 overtaken by data
  HANDOVER.md                    this file
  iou_sweep.py                   Exp-2 at CONF=0.25 (operating threshold)
  iou_sweep_results.txt          its results
  iou_sweep_lowconf.py           the control, CONF=0.001 - the experiment that changed everything
  iou_sweep_lowconf_results.txt  its results
model_training/models/circe_yolo26n_13cls_100ep/   frozen detector + README with per-class results
model_training/merged_v2/                          13-class dataset, gitignored, split pinned
circe_v3/GEMINI_*.md                               Gemini's proposal - see §6, technically wrong
my_slam_vggt_omega/RESEARCH.md                     SL(4) evidence contradicting it
```

## 10. Hard constraints — non-negotiable

From `CLAUDE.md` (user-level and project-level) and standing instructions:

1. **Verify against official sources; attach evidence.** Never assert from memory. Cite a URL, a
   command and its output, or a file and line. An unverified guess dressed as a "hypothesis" is
   a hallucination. Say "I have not verified this" and go verify it.
2. **Surface blockers up front**, before implementing — never mid-execution.
3. **Never abandon a task list midway.** Only stop for a genuine blocking question.
4. **Do exactly what was asked.** No unrequested fallbacks, safety nets or extra scope. If you
   see a better way, *ask first*.
5. **The user commits. You do not.** Never run `git commit`.
6. **No subagents without explicit approval.**
7. **Never compromise either laptop's security.** State LAN/firewall exposure plainly and get
   explicit confirmation first.
8. **Do not waste Gemini API quota.** Annotation reruns cost real money.
9. **Never delete NPU-BOLT data.**
