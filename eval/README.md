# Shared NIDS evaluation module

## Purpose and current scope

All detection methods use this module to evaluate binary attack detection.

Normal traffic is 0 and attack traffic is 1. Higher scores indicate greater

attack suspicion.

The synthetic demo checks interfaces and result formatting. Its numbers

are not experimental findings. Passing unit tests does not establish

acceptance of the actual P2 datasets.

Run all commands from the repository root.

## Installation

Use a separate Python environment.

```bat

python -m pip install -r eval/requirements-eval.txt

```

Development was performed on Windows with Python 3.13.10.

Dependency versions are recorded in requirements and generated results.

## Quick start

Generate the synthetic example:

```bat

python -m eval.demo

```

Validate the generated record:

```bat

python -m eval.validate eval/examples/result_demo.json

```

Run tests:

```bat

python -m unittest discover -s tests -p "test_*.py" -v

```

Generated example files:

- eval/examples/synthetic_input.json

- eval/examples/result_demo.json

The demo calls all five core metric interfaces and the supplementary

classification summary. It measures a toy NumPy scorer, not a trained model.

## Input conventions

- y_true: one-dimensional binary ground truth, normal=0 and attack=1.

- y_score: one-dimensional finite attack scores, aligned with y_true.

- y_pred: binary predictions at one externally determined threshold.

- attack_family: true family metadata, aligned with predictions.

- row IDs: unique nonempty strings identifying the evaluated samples.

- Threshold comparison: score >= threshold means attack.

- Scores need not be probabilities; reconstruction errors are valid when

  larger values mean greater anomaly.

- Evaluate complete test-set scores. Do not average batch AP or ROC-AUC.

- Do not rebalance or apply training class weights to test-set metrics.

## Core function interfaces

Import functions from eval.metrics.

### compute_curves(y_true, y_score)

Returns status, roc_auc, average_precision, and ROC/PR curve arrays.

PR summary uses scikit-learn average_precision_score (AP), not trapezoidal

integration of the PR curve.

For single-class ground truth, this interface returns status=single_class

and null summary values and curves. Returning null AP is an explicit

interface policy for this situation.

The first ROC threshold is serialized as null when it represents the

above-maximum-score, never-alarm endpoint. PR precision and recall arrays

have one more element than the PR threshold array.

### tpr_at_fpr(y_true, y_score, targets=(0.001, 0.01, 0.05))

Returns empirical ROC operating points: target FPR, actual FPR, TPR,

threshold and threshold kind.

Selects the maximum observed TPR subject to actual FPR <= target.

Ties favor the lowest FPR, then the first corresponding curve point.

Equal scores are not split, and no interpolation is used.

This is a descriptive test-set ROC summary, not validation-calibrated

deployment performance. A threshold selected on validation data must

instead be applied unchanged to test scores, and its actual test FPR/TPR

must be reported.

### per_class_recall(y_true, y_pred, attack_family, families=None)

Returns attack-family support, detected count and binary detection recall.

It does not require predicting the attack family itself.

Specify the complete expected attack-family list to include absent groups.

Support=0 produces recall=null, not zero.

Under the current project label mapping, benign corresponds to y_true=0.

All observed attack families must be included if families is supplied.

Normal traffic is excluded from attack-family recall denominators.

### measure_inference(predict_fn, X, batch_size=512,

                      warmup=5, repeats=30, threads=1)

Measures synchronous CPU prediction on a prepared fixed batch.

predict_fn must return a materialized NumPy array of shape (batch_size,),

containing one finite attack score per input row.

Returns individual measured batch times, median batch milliseconds,

amortized milliseconds per flow, throughput and measurement settings.

Input slicing, file reading, model loading, training and metric evaluation

are excluded. Operations inside predict_fn are included.

Defaults: batch=512, warmup=5, repeats=30, supported thread-pool limit=1.

The demo uses smaller settings solely for interface checks.

threadpoolctl limits supported BLAS/OpenMP pools. Model owners must

separately control model/framework workers and record those settings.

Neural networks must use evaluation mode and disable gradient recording.

All compared methods must use the same hardware and measurement settings.

Batch time divided by batch size is amortized time, not single-request

response latency. Measure batch=1 separately if needed.

This implementation does not benchmark GPU inference.

### paired_bootstrap(y_true, scores_a, scores_b, *,

                     row_ids_a, row_ids_b, metric="ap",

                     target_fpr=0.01, n_bootstrap=2000,

                     confidence=0.95, seed=42)

Compares AP or empirical TPR-at-FPR. Difference is always A minus B.

Returns original metric values, their difference, percentile interval,

class counts and resampling settings.

Both binary classes must be present. Normal and attack rows are sampled

with replacement separately, preserving their observed counts.

The same sampled positions are used for A and B.

Row IDs must be unique and identical in order. Before calling, also verify

that both prediction sets have the same dataset version, protocol,

variant, test-file identity and ground-truth labels. Do not manufacture

matching IDs independently.

This row-level interval conditions on the observed class proportions and

the supplied fitted models. It does not include model-training variation

or account for within-group traffic dependence.

For TPR-at-FPR, each resample recomputes the descriptive ROC operating

point. This is not uncertainty for a fixed deployment threshold.

The demo uses 50 repetitions for speed. Predefine the formal repetition

count and report it. Full-dataset resampling may be computationally costly.

## Supplementary classification summary

classification_metrics(y_true, y_pred, *, is_attempted=None)

Returns accuracy, macro-F1, confusion counts, TPR, FPR, precision,

sample counts and optional attempted-subgroup FPR.

Macro-F1 uses fixed labels \[0,1] and zero_division=0.

Undefined TPR, FPR or precision returns null.

is_attempted must be a matching boolean array. Under the disclosed P2

policy, attempted samples are negative-class traffic. Missing metadata

is distinguished from a provided subgroup with zero samples.

Use the same y_pred for classification metrics and per-family recall.

## Real-data integration

Read P2 data using src.model_loader.ModelDataset.

Each iteration yields X, y and metadata. Only X enters the model;

labels, row IDs, attack family, dates and attempted flags are metadata.

Use the delivered train/val/test model files and corresponding

model_*_indices.parquet. Do not refit preprocessing or regenerate splits.

Main input has 140 features; the no_destination_port variant has 75.

Follow the processor feature whitelist.

Final counts and distributions are documented in docs/preprocess_report.md.

The original stratified split is approximately 71.20%/14.81%/13.99% after

float32 duplicate removal; it is not exactly 70/15/15.

Original and corrected versions are independent evaluation populations.

Do not pair their rows merely because row numbers or labels match.

## Result records

eval/schema.json defines one model, one seed, one test-dataset record.

Preserve per-seed records; five-seed mean and sample standard deviation

are downstream summaries, not replacements for those records.

Required sections include method/configuration, data identity and hash,

seed, decision rule, metrics, training/inference cost, optional daily

alert scenario, pairwise comparisons and provenance.

Each comparison identifies both methods and their seeds.

Comparisons may be empty before another method's aligned scores exist.

Use is_demo=true for synthetic examples. Formal records require real

dataset metadata, measured training time and a non-demo threshold source.

Serialize with json.dumps(..., allow_nan=False).

Do not manually edit computed metric values.

The Git commit may exclude uncommitted files. Save the implementation

version and relevant file hashes; demo provenance records this distinction.

## Daily alert scenarios

With explicit assumed daily normal/attack flow counts:

expected_false_alerts = assumed_benign_flows * FPR

expected_true_alerts = assumed_attack_flows * TPR

expected_total_alerts = expected_false_alerts + expected_true_alerts

Use FPR and TPR from the same classification operating point.

These are scenario estimates assuming the measured conditional rates

transfer to the stated traffic mix, not measured deployment alert counts.

If no scenario is specified, daily_alerts is null.

## Validation limits

eval.validate checks schema, finite JSON values and selected cross-field

relations: counts, classification metrics, family recalls, timing formulas,

daily alert formulas and comparison differences/interval order.

It does not establish provenance authenticity, verify actual file hashes,

recompute results from raw predictions or prove threshold-selection

independence. Those checks remain part of experiment review and handover.

## Handover

P3 must separately complete actual P2 data acceptance and Methods 3.2.

P4 reruns the synthetic demo and validation in its own environment.

Record actual test results, implementation version and P4 acceptance

in the shared handover table. Do not prefill acceptance signatures.
