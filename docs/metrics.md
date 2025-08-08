# Metrics

SuperSegmentaion evaluates local features and semantic masks with a shared set of metrics. This reference outlines what each metric measures, provides the governing formula, notes when the metric is computed in the training/validation/inference cycle, and lists production thresholds or service level objectives (SLOs). The goal is to give engineers a concise yet comprehensive guide for interpreting evaluation reports.

The metrics fall into three families:

1. **Detector metrics** – judge how consistently the interest point detector fires across viewpoint and illumination changes.
2. **Descriptor metrics** – quantify the discriminative power of feature descriptors and the reliability of matching.
3. **Segmentation metrics** – score the alignment between predicted semantic labels and ground truth masks.

Beyond summarizing performance, these metrics influence model selection, hyperparameter tuning, and operational alerts. During experimentation, lightweight metrics provide rapid feedback, while full evaluations run on scheduled validation jobs. In production, a subset of metrics is recomputed on shadow traffic to detect drift before it affects downstream tasks.


Each section below describes the metric, states the formula, clarifies when it is computed, and summarizes the production target. All formulas use variables common in computer vision: $K$ for keypoints, $H$ for homographies, $TP$/$FP$/$FN$ for confusion‑matrix counts, etc.

## 1. Detector Metrics

Detector metrics capture the raw ability of the model to find stable interest points. Without reliable detections, even the best descriptors cannot recover accurate matches, so these metrics form the foundation of the feature pipeline.

### 1.1 Repeatability

**Definition:** Fraction of detector responses that reappear after mapping keypoints from a reference image to a target image using the ground truth transformation.

**Formula:** With keypoints $K_1$ and $K_2$ and transformation $H$, let $R$ be the set of matched keypoints within a tolerance $\epsilon$:

$$\text{Repeatability} = \frac{|R|}{\min(|K_1|, |K_2|)}.$$

**Lifecycle:** Calculated during **validation** and sampled in **inference** monitoring. Training loops usually skip it because pairing images and computing homographies is costly.

**SLO:** ≥ **0.65** on the standard validation pairs.

### 1.2 Localization Error

**Definition:** Average Euclidean distance, in pixels, between a reference keypoint warped by $H$ and its match in the target image.

**Formula:**

$$\text{LocErr} = \frac{1}{|R|} \sum_{(p,q) \in R} \|Hp - q\|_2.$$

**Lifecycle:** Used in **validation**; occasionally checked offline after deployment. Not computed during routine training.

**SLO:** Mean error < **1.5 px** at 1‑MP resolution.

### 1.3 Homography Accuracy

**Definition:** Error of a homography estimated from matched keypoints, measured as average corner error of the image.

**Formula:** For corners $c_i$ and homographies $H_{gt}$ and $H_{est}$:

$$\text{ACE} = \frac{1}{4}\sum_{i=1}^4 \|H_{gt}c_i - H_{est}c_i\|_2.$$

**Lifecycle:** Reported during **validation**; used in **inference** regression tests to detect drift.

**SLO:** Corner error < **3 px** on evaluation scenes.

### 1.4 Supporting Statistics

- **Keypoints per image:** ensures the detector produces enough but not too many points. Computed in **training** summaries and **validation** reports. Target **500–1200** keypoints for a megapixel frame.
- **Detection latency:** average runtime per image. Profiled in **training** and **inference**; SLO **< 15 ms** on reference hardware.

## 2. Descriptor Metrics

Descriptors transform image patches into vectors that can be compared across frames. Metrics here assess both individual descriptor quality and the behavior of the matching system built on top of them.

### 2.1 Precision and Recall

**Definition:** Evaluate correctness of descriptor matches produced by nearest‑neighbor search and optional ratio test.

**Formula:** For putative matches $M$, true matches $T$, and ground truth correspondences $G$:

$$\text{Precision} = \frac{|T|}{|M|}, \qquad \text{Recall} = \frac{|T|}{|G|}.$$

The harmonic mean yields the F1 score: $2PR/(P+R)$.

**Lifecycle:** Precision‑recall curves generated in **validation**; diagnostics may also run during **training** when tuning descriptor losses. Not usually computed in production due to ground truth requirements.

**SLO:** At least **0.85 precision** at **0.60 recall**.

### 2.2 Matching Score

**Definition:** Ratio of correct matches to the smaller number of detected keypoints; reflects joint detector and descriptor quality.

**Formula:**

$$\text{Matching Score} = \frac{|T|}{\min(|K_1|,|K_2|)}.$$

**Lifecycle:** Logged in **validation** and occasionally during **training**. In production, small sample checks catch gross regressions.

**SLO:** ≥ **0.55**.

### 2.3 Mean Average Precision (mAP)

**Definition:** Treats matching as a ranking task. Matches are sorted by similarity score and the area under the precision‑recall curve is computed.

**Formula:**

$$\text{mAP} = \int_0^1 P(r) dr \approx \sum_i P(r_i)\Delta r_i.$$  
The integral is approximated by summing precision at discrete recall levels.

**Lifecycle:** Calculated on **validation** sets and for **inference** regression tests. Rarely used during training unless a ranking loss is optimized.

**SLO:** **0.80** or higher.

### 2.4 Ratio Test Calibration

**Definition:** Determines the best threshold for the Lowe ratio test by analyzing how precision and recall vary with $r=d_1/d_2$.

**Lifecycle:** Performed offline during **validation** sweeps. Not part of routine training or production pipelines but informs the default threshold.

**Guideline:** Default ratio **0.8** maximizes F1 on validation data; deviation requires explicit justification.

## 3. Segmentation Metrics

### 3.1 Pixel Accuracy

**Definition:** Portion of pixels correctly labeled.

**Formula:**

$$\text{PixelAcc} = \frac{N_{correct}}{N}.$$

**Lifecycle:** Quick feedback metric during **training** and **validation**. Rarely used in production dashboards because it may hide class imbalance.

**SLO:** ≥ **0.90** on balanced datasets.

### 3.2 Mean Intersection over Union (mIoU)

**Definition:** Average IoU across classes; the principal metric for semantic segmentation.

**Formula:**

For class $c$ with true positives $TP_c$, false positives $FP_c$, and false negatives $FN_c$:

$$\text{IoU}_c = \frac{TP_c}{TP_c+FP_c+FN_c}, \qquad \text{mIoU} = \frac{1}{C}\sum_{c=1}^C \text{IoU}_c.$$

**Lifecycle:** Tracked during **training** for convergence, used in **validation** for model selection, and checked in **inference** regression tests.

**SLO:** mIoU **> 0.75** on the target dataset; additionally, no single class should drop below **0.60**.

### 3.3 Dice Coefficient / F1

**Definition:** Harmonic mean of pixel precision and recall, emphasizing overlap for small structures.

**Formula:**

$$\text{Dice}_c = \frac{2TP_c}{2TP_c + FP_c + FN_c}, \qquad \text{Dice} = \frac{1}{C}\sum_{c=1}^C \text{Dice}_c.$$

**Lifecycle:** Used as an auxiliary metric in **training** when Dice loss is employed, and in **validation** for tasks requiring sharp boundaries. In production, monitored when small objects are critical.

**SLO:** Dice **> 0.80** for key foreground classes.

### 3.4 Boundary IoU

**Definition:** IoU computed over a narrow band around object edges, capturing boundary accuracy.

**Formula:** With $B_{gt}$ and $B_{pred}$ denoting dilated boundary masks:

$$\text{Boundary IoU} = \frac{|B_{gt} \cap B_{pred}|}{|B_{gt} \cup B_{pred}|}.$$

**Lifecycle:** Evaluated in **validation** and in specialized **inference** tests for edge‑critical applications. Optional during training when boundary‑aware losses are used.

**SLO:** > **0.70** for classes with strict boundary requirements.

### 3.5 Panoptic Quality (PQ)

**Definition:** Unified score for panoptic segmentation combining detection and mask quality.

**Formula:** For matched segments $TP$:

$$\text{PQ} = \frac{\sum_{(p,g)\in TP} \text{IoU}(p,g)}{|TP| + 0.5|FP| + 0.5|FN|}.$$

**Lifecycle:** Used in **validation** for panoptic models and in offline **inference** checks. Rare in standard training loops unless panoptic loss is optimized.

**SLO:** PQ **> 0.55** before release.

### 3.6 Confusion Analysis

Per-class IoU, accuracy, and Dice scores are accompanied by full confusion matrices during **validation**. The matrices reveal which classes are commonly confused, highlight systematic false positives or false negatives, and guide data collection efforts. Production gating requires every monitored class to satisfy its SLO; a single degraded class can trigger retraining or targeted annotation drives.

### 3.7 Inference-Time Monitoring

Segmentation deployments maintain rolling windows of mIoU and pixel accuracy on manually annotated samples collected from production streams. This delayed ground truth enables alerting when distribution shifts occur. Though less frequent than validation, the process provides a final safeguard against silent regressions.

## 4. Metric Timing Overview


| Metric | Training | Validation | Inference |
| --- | --- | --- | --- |
| Repeatability | – | ✓ | ✓ (offline) |
| Localization Error | – | ✓ | ✓ (offline) |
| Homography Accuracy | – | ✓ | ✓ (offline) |
| Keypoints per Image | ✓ | ✓ | ✓ |
| Detection Latency | ✓ | ✓ | ✓ |
| Precision/Recall | ✓ (diagnostics) | ✓ | – |
| Matching Score | ✓ (optional) | ✓ | ✓ (spot checks) |
| mAP | – | ✓ | ✓ (offline) |
| Pixel Accuracy | ✓ | ✓ | – |
| mIoU | ✓ | ✓ | ✓ (regression) |
| Dice | ✓ (if loss) | ✓ | ✓ (task specific) |
| Boundary IoU | ✓ (boundary loss) | ✓ | ✓ (edge critical) |
| Panoptic Quality | ✓ (panoptic models) | ✓ | ✓ |

Check marks denote phases where the metric is routinely computed. A dash indicates the metric is typically skipped unless a special experiment demands it.

## 5. Production Threshold Reference

| Metric | Target |
| --- | --- |
| Repeatability | ≥ 0.65 |
| Localization Error | < 1.5 px |
| Homography Accuracy | < 3 px |
| Precision @ 0.60 Recall | ≥ 0.85 |
| Matching Score | ≥ 0.55 |
| Descriptor mAP | ≥ 0.80 |
| Keypoints per Image | 500–1200 |
| Detection Latency | < 15 ms |
| Pixel Accuracy | ≥ 0.90 |
| mIoU | > 0.75 |
| Per‑class IoU | > 0.60 |
| Dice Coefficient | > 0.80 |
| Boundary IoU | > 0.70 |
| Panoptic Quality | > 0.55 |

Meeting these SLOs is required before a model is promoted to production. Continuous monitoring helps detect degradation as new data or hardware is introduced.

## 6. Key Takeaways

- Detector metrics focus on consistency and geometric precision. Repeatability ≥ 0.65 and homography corner error < 3 px are the main production gates.
- Descriptor metrics judge the reliability of feature matching. Precision 0.85@0.60 recall, matching score ≥ 0.55, and mAP ≥ 0.80 are typical acceptance levels.
- Segmentation metrics evaluate mask quality. Production targets include mIoU > 0.75, per‑class IoU > 0.60, Dice > 0.80 for critical classes, and boundary IoU > 0.70 when edges matter.
- Tables summarize when each metric is computed and list the associated SLOs, giving teams a single source of truth for evaluation and deployment standards.
