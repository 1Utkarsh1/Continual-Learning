# Benchmark Card

## Scope

This project evaluates continual-learning strategies on task streams where a model
sees one subset of classes at a time and is re-evaluated on all previously seen
tasks after every training step.

## Implemented Methods

- Baseline sequential fine-tuning.
- Elastic Weight Consolidation with empirical Fisher estimates.
- Reservoir replay with a bounded memory budget.
- Learning without Forgetting with temperature-scaled distillation.
- DER++ with online replay-logit storage.
- A-GEM with replay-memory gradient projection.
- ER-ACE with asymmetric current-task cross-entropy plus replay.
- GDumb with class-balanced memory and from-scratch memory training.
- Calibrated Anchor Replay with balanced exemplars, logit anchors, feature
  anchors, per-class prototypes, and a post-task calibration head.
- `bic`, `icarl`, and `x_der_lite` are lightweight protocol baselines built from
  the CAR components for ablation and comparison; they are not drop-in
  reproductions of the original papers.

## Datasets

- `synthetic`: deterministic image-like tensors for fast CI and unit tests.
- `split_mnist_quick`: real MNIST images with bounded per-task subsets so a full
  four-method comparison runs on CPU.
- `split_mnist`: full five-task MNIST stream for longer local experiments.
- `split_cifar10_headline`: real CIFAR-10 images, five class-incremental tasks,
  two verification seeds, and a compact residual ConvNet.
- `paper/split_cifar10_full`: full Split CIFAR-10 protocol for paper runs.
- `paper/split_cifar100_full`: full Split CIFAR-100 protocol for paper runs.
- `paper/split_tinyimagenet`: Split TinyImageNet protocol; requires the dataset
  to be downloaded into the configured `data_dir`.

## Metrics

- Average final accuracy: mean accuracy over seen tasks after the final task.
- Average learning accuracy: mean diagonal accuracy after each task is learned.
- Average forgetting: best previous accuracy minus final accuracy for prior tasks.
- Backward transfer: final accuracy minus first-learned accuracy on prior tasks.

## Reproducibility

Every run writes `config.yaml`, `run_metadata.json`, event-level `metrics.jsonl`,
CSV/JSON matrices, final `metrics.json`, and optionally MLflow params, metrics,
tags, and artifacts. The suite command can aggregate multiple methods and seeds
into a leaderboard plus report plots.

## Limitations

The Split CIFAR-10 reported result is designed for local reproducibility and
engineering verification. It is not a leaderboard claim. Serious research
comparisons should increase seeds, epochs, memory budgets, and dataset coverage.
The GDumb comparison uses a larger memory budget than the main 5,000-example
suite and should be read as a high-memory result rather than a same-budget claim.
Paper claims must be made only from matched memory budgets, matched model
families, full protocol configs, and multi-seed confidence intervals.
