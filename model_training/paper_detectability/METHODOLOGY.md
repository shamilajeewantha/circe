# Methodology — Morphology-Dependent Resolution Limits of Visual Defect Detection

**Target venue:** ICCCIT-2027 (Track-1: Signal and Image Processing / Machine Learning)
**Status:** methodology draft, 2026-09-22. Experiments not yet run.

---

## 0. Positioning (read this first)

This work is **not** the first to calibrate camera placement from measured detector
performance. That is:

> Zwick, Gerdts & Stütz, *"Sensor-Model-Based Trajectory Optimization for UAVs to Enhance
> Detection Performance,"* **Sensors 23(2):664, 2023**, DOI 10.3390/s23020664.

Zwick et al. build a "sensor performance model" — average precision as a joint function of
ground sample distance and elevation angle, measured empirically over ~3,300 annotated aerial
images — and use that surface inside trajectory optimisation. **We cite them as the origin of
the performance-surface idea and do not claim it.**

Related established results we also do not claim:
- Detector accuracy degrades with apparent object size: Dollár et al., *IEEE TPAMI* 34(4):743-761,
  2012 (miss-rate vs pedestrian pixel height); COCO small/medium/large AP (Lin et al., ECCV 2014).
- Thresholding detection probability against target size is the **probability-of-detection (POD)**
  methodology from non-destructive evaluation (MIL-HDBK-1823A). Our per-class threshold is a
  vision POD curve and is described as such.

**What this work adds.** Zwick's surface is single-class and aerial. Existing GSD rules in
inspection practice ("sample distance < half the smallest defect", e.g. DOI 10.3390/s26031031)
are uniform across defect types. We ask whether that uniformity is justified:

> **Do different defect morphologies impose different resolution requirements, and can a single
> standoff rule serve them all?**

To our knowledge no published work derives **per-class** resolution requirements from **measured
detector recall** across a multi-morphology defect taxonomy.

---

## 1. Research questions

- **RQ1.** How does per-class detection recall vary with the apparent pixel footprint of the
  defect instance?
- **RQ2.** Does the degradation profile depend on **defect morphology** — specifically, do
  texture-defined defects (corrosion, efflorescence) differ systematically from compact,
  boundary-defined objects (bolts, gauges)?
- **RQ3.** Does a single resolution threshold satisfy all classes, or do the per-class
  requirements conflict?

## 2. Hypotheses

- **H1.** Per-class recall declines monotonically with decreasing footprint, with a class-specific
  knee (the POD-style threshold).
- **H2 (principal).** Compact boundary-defined classes retain recall to **smaller** footprints than
  texture-defined classes.
  *Mechanism:* a compact object remains identifiable from surviving low-frequency shape evidence;
  a texture-defined defect's discriminative signal is high-spatial-frequency and is destroyed by
  downsampling before its extent is.
- **H3.** The spread of per-class knee footprints is large enough (ratio > 1.5) that a uniform
  standoff rule is either wasteful for some classes or unsafe for others.

Note H2 is **directional and falsifiable**. The opposite outcome — texture defects being *more*
robust because they are spatially extended — is a legitimate result and is reported as such.

## 3. Materials

| item | detail |
|---|---|
| Detector | YOLO26-n, 13 classes, mAP50 0.5508, mAP50-95 0.3970 (frozen; no retraining) |
| Evaluation set | held-out test split, 5,565 images, 13 classes, split membership pinned by manifest |
| Classes | concrete_crack, corrosion, fluid_patch, fire, smoke, gauge_face, efflorescence, exposed_rebar, spalling, missing_bolt, bolt_ok, bolt_defective, bolt_corroded |

The detector is frozen throughout. This study characterises a *given* detector; it does not
tune one.

## 4. Independent variable — apparent footprint

Apparent footprint of an instance is defined as

```
F = sqrt(w_px * h_px)
```

the geometric mean side length of its bounding box in pixels — a scale-equivariant summary of
apparent size.

**Manipulation.** For each target footprint level, the **full image** is resampled by factor
`s = F_target / F_original` so the instance attains the target footprint, then inference is run.
Full-image resampling (rather than crop-and-rescale) is used deliberately so that object/context
ratio and background clutter are preserved; cropping would confound resolution with context loss.

**Levels.** `F_target` ∈ {8, 12, 16, 24, 32, 48, 64, 96} px, plus the native-resolution control.

Because instances within an image have differing native footprints, each image is processed once
per target level with the scale factor set from a designated reference instance; all instances in
that image are re-measured post-resampling and binned by their **achieved** footprint rather than
the nominal target. This avoids assuming uniform object size.

## 5. Dependent variables

Per class, per footprint bin:
- **Recall** (primary — a missed defect is the costly error in inspection)
- Precision, AP50, AP50-95
- Count of instances in bin (for confidence intervals)

## 6. Morphology grouping — pre-registered

Assigned **before** results are inspected, to prevent post-hoc rationalisation:

| group | classes |
|---|---|
| **Compact / boundary-defined** | bolt_ok, bolt_defective, bolt_corroded, missing_bolt, gauge_face |
| **Texture / diffuse** | corrosion, efflorescence, concrete_crack, spalling, exposed_rebar |
| **Region / plume** | fire, smoke, fluid_patch |

**Robustness check.** Because this grouping is judgement-based, a quantitative proxy is reported
alongside it: mean normalised high-frequency gradient energy within ground-truth boxes per class.
If the proxy orders classes consistently with the manual grouping, the grouping is empirically
supported; if not, the discrepancy is reported.

## 7. Analysis

1. Fit a per-class recall-vs-footprint curve (monotone logistic).
2. Define **F_90** and **F_80** per class: the footprint at which recall reaches 90% / 80% of its
   native-resolution value. This is the POD-style operating threshold.
3. **Test H2:** compare F_80 distributions between morphology groups (Mann-Whitney U;
   non-parametric, small group sizes, no normality assumption).
4. **Test H3:** report max/min ratio of F_80 across classes.
5. Report bootstrap confidence intervals per bin; classes with < 30 instances in a bin are marked
   indicative and excluded from significance testing.

## 8. Threats to validity

- **Resampling is not distance (principal threat).** Downsampling an image is not equivalent to
  imaging the same scene from further away: it omits defocus, atmospheric effects, motion blur and
  sensor-noise re-scaling. Our results therefore characterise **resolution sensitivity**, not full
  standoff-distance behaviour, and this is stated as a scope boundary rather than buried.
  *Mitigation:* validate on a small purpose-captured set imaged at several true distances, and
  report agreement between resampled and true-distance curves for the classes it covers.
- **Incidence angle not varied.** Zwick varied GSD *and* elevation angle; our source images carry
  no pose metadata, so only footprint is manipulated. This is a genuine reduction in scope relative
  to prior work and is declared.
- **Single detector.** Findings may be architecture-specific. *Mitigation:* replicate on one
  additional detector if time permits; otherwise declared as a limitation.
- **Class imbalance.** Rare classes (bolt_defective, n=24 in the val split) yield wide intervals.
  Treated as indicative, excluded from hypothesis tests.
- **Label provenance.** Three bolt classes carry LLM-generated labels (Gemini 3-pass consensus +
  SAM3 refinement) rather than human annotation. Measured inter-pass class disagreement is 14-18%.
  This is disclosed, and those classes are reported separately from dataset-native classes.

## 9. Expected contributions

1. **Empirical:** per-class resolution thresholds (F_80 / F_90) across a 13-class,
   multi-morphology defect taxonomy.
2. **Explanatory:** evidence for or against morphology as a predictor of resolution sensitivity —
   transferable beyond this dataset and detector.
3. **Practical:** evidence on whether uniform GSD rules in inspection practice are adequate, and if
   not, per-class replacements.

## 10. Feasibility

| requirement | status |
|---|---|
| Trained detector | ✅ exists |
| Test split | ✅ 5,565 images, pinned |
| New training | none |
| New annotation | none |
| Robot hardware | not required |
| Compute | 9 levels × 5,565 images inference — hours |

## 11. Downstream use (not part of this paper)

The per-class F_80 thresholds are the calibration input for a subsequent coverage-planning paper,
where they replace hand-tuned standoff bounds in an inspection planner. That work is out of scope
here and targets a robotics venue.
