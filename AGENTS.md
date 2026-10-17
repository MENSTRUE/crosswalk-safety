# AGENTS.md

## Purpose

This file defines the engineering and claim rules for AI coding agents working on **Crosswalk Safety v4**.

The repository is a portfolio MVP for weakly supervised temporal pedestrian–vehicle conflict screening.

All future changes must preserve technical honesty, reproducibility, and consistency between training and inference.

---

## Project Identity

**Project:** Crosswalk Safety v4  
**Domain:** Computer Vision / Temporal Deep Learning  
**Primary task:** next-sample binary conflict-proxy prediction  
**Detector:** YOLO11n pretrained on COCO  
**Custom model:** Temporal Conv1D + BiLSTM  
**Application:** Streamlit

---

## Non-Negotiable Claim Rules

Never describe this project as:

- a validated traffic-violation detector;
- a production traffic enforcement system;
- an automatic zebra-crossing detector;
- a custom-trained YOLO model;
- a system with validated real-world violation accuracy.

The correct framing is:

> Weakly supervised temporal safety-conflict screening using YOLO11n bounding-box preprocessing and a custom Temporal Conv1D + BiLSTM classifier.

The held-out metrics measure agreement with the project's weak/pseudo target definition.

---

## Target Definition

```text
conflict_proxy =
    pedestrian_in_ROI > 0
    AND
    vehicle_in_ROI > 0
```

The temporal model predicts the proxy on the **next sampled frame**.

Do not silently change this target definition without updating the training notebook, inference code, README, reports, and saved checkpoint metadata.

---

## Canonical v4 Configuration

```text
sampling interval      = 2.0 seconds
sequence length        = 8 sampled frames
history duration       = 16 seconds
prediction horizon     = 1 sampled step / 2 seconds
YOLO confidence        = 0.25
YOLO IoU               = 0.50
YOLO input size        = 640
validation threshold   = 0.405
feature count          = 21
```

---

## Canonical Held-Out Metrics

```text
Accuracy   : 72.25%
Precision  : 57.61%
Recall     : 80.84%
F1         : 67.28%
ROC-AUC    : 82.02%
PR-AUC     : 71.88%
```

Best validation F1:

```text
68.43%
```

Do not replace these values with local demo diagnostics.

A Streamlit upload may show agreement-vs-proxy for one clip. That is **not** benchmark accuracy.

---

## Dataset / Split Rules

```text
120 videos
18,000 sampled frames
91 train videos
19 validation videos
10 test videos
12,922 train sequences
2,698 validation sequences
1,420 test sequences
```

Train / validation / test separation is **video-level**.

Never randomly split neighboring frames from one video across different evaluation partitions.

---

## Detector Role

YOLO11n is only the preprocessing detector.

Target COCO classes:

```text
0 = person
2 = car
3 = motorcycle
5 = bus
7 = truck
```

Do not claim YOLO11n was trained by this repository.

The custom-trained component is:

```text
crosswalk_temporal_cnn_bilstm_v4.pt
```

---

## Feature Contract

Inference must generate the exact feature order used during training.

The canonical feature list is controlled by:

```text
src/feature_extractor.py
```

The current model expects **21 features**.

Never reorder, rename, add, or remove features without retraining the temporal checkpoint.

Normalization must use:

```text
models/feature_mean.npy
models/feature_scale.npy
```

Do not fit a new scaler during application inference.

---

## Temporal Alignment Rule

Training and inference must preserve:

```text
8 previous sampled feature vectors
        ↓
predict proxy label on the next sampled frame
```

The current frame must not be inserted into the history buffer before the prediction intended for that frame is produced.

This alignment is critical.

---

## Repository Responsibilities

### `app/`
Contains user-facing Streamlit code only.

### `src/detector.py`
Owns YOLO loading and object detection.

### `src/feature_extractor.py`
Owns the canonical 21-feature schema and temporal delta calculation.

### `src/temporal_model.py`
Owns the Conv1D + BiLSTM architecture, checkpoint loading, scaling, probability inference, and classification threshold.

### `src/pipeline.py`
Owns end-to-end video sampling and temporal inference orchestration.

### `src/roi.py`
Owns ROI coordinate conversion and geometry helpers.

### `src/visualize.py`
Owns frame overlays and rendering.

### `notebooks/`
Contains experimentation, preprocessing, training, evaluation, and visual previews.

### `reports/`
Contains generated experiment evidence.

### `assets/previews/`
Contains figures intended for README / portfolio presentation.

---

## Engineering Rules

1. Keep training and inference preprocessing identical.
2. Keep feature order deterministic.
3. Never fit normalization statistics on validation or test data.
4. Preserve video-level split boundaries.
5. Prefer explicit code over hidden global state.
6. Validate model-artifact paths before inference.
7. Fail clearly when required model files are missing.
8. Keep model claims separate from demo diagnostics.
9. Do not silently change the configured ROI.
10. Do not silently change sample timing.
11. Do not commit raw large traffic videos.
12. Do not commit temporary runtime files.
13. Keep public documentation synchronized with actual code.

---

## Video Rendering

Temporal inference is sampled every 2 seconds.

If a smooth full-FPS preview is later implemented, keep model inference sampled unless a new benchmark justifies denser inference.

Do not imply that the current temporal network evaluates every original video frame.

For browser-compatible annotated output, H.264 MP4 with `yuv420p` is preferred.

---

## Evaluation Rules

Any future experiment should report at minimum:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix
- validation-selected threshold
- split definition

Because this is a safety-screening context, recall should be discussed alongside precision rather than in isolation.

Never cherry-pick only the strongest metric.

---

## README Evidence

README may reference existing repository assets:

```text
assets/previews/bbox_preprocessing_preview.png
assets/previews/frame_proxy_distribution.png
assets/previews/raw_video_preview.png
assets/previews/test_confusion_matrix.png
assets/previews/test_pr_curve.png
assets/previews/test_probability_distribution.png
assets/previews/test_roc_curve.png
assets/previews/training_loss.png
assets/previews/validation_metrics.png
```

README may link to:

```text
reports/final_summary.json
reports/frame_features.csv
reports/frame_features_with_motion.csv
reports/test_metrics.json
reports/test_predictions.csv
reports/training_history.csv
reports/video_audit.csv
reports/video_split.csv
```

Do not invent missing plots or reports.

---

## Code Change Checklist

```text
[ ] imports resolve
[ ] checkpoint loads
[ ] feature count = 21 unless retrained
[ ] feature order matches checkpoint
[ ] scaler statistics load
[ ] Streamlit starts
[ ] sample video completes inference
[ ] temporal predictions appear after warm-up
[ ] CSV export works
[ ] limitation text remains visible
[ ] README remains technically accurate
```

---

## Git Rules

Preferred examples:

```text
feat: integrate trained temporal conflict model
feat: add browser-compatible annotated video output
docs: document v4 architecture and held-out metrics
fix: preserve temporal inference alignment
refactor: separate bbox preprocessing from temporal inference
```

Avoid meaningless streak commits or empty commits.

Do not force-push unless explicitly required and approved.

---

## Scientific Honesty

Do not:

- call confidence a severity score;
- call agreement-vs-proxy real-world accuracy;
- call a pseudo label human ground truth;
- call conflict prediction a legal violation judgment;
- report unmeasured latency or FPS;
- fabricate external validation.

If a requested claim is not supported by repository evidence, state that it is unsupported.

---

## Future Extension Rules

Potential future work may include manual event annotation, crosswalk segmentation, tracking, trajectory modeling, time-to-collision, adaptive sampling, stronger temporal baselines, probability calibration, and external validation.

Any such addition must be described as future work until implemented and evaluated.

---

## Final Agent Principle

**Evidence first. Claims second.**

The repository should make it easy for a reviewer to trace:

```text
raw video
→ preprocessing
→ engineered features
→ temporal sequence
→ trained model
→ test report
→ Streamlit inference
```

without overstating what the system has actually validated.
