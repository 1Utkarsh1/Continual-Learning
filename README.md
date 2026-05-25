# Continual Learning Benchmark

[![CI](https://github.com/1Utkarsh1/Continual-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/1Utkarsh1/Continual-Learning/actions/workflows/ci.yml)

A PyTorch benchmark framework for comparing continual-learning strategies under
the same task stream, metric suite, artifact pipeline, local MLflow tracker, and
report generator.

Experiments are config-driven, methods share a stable lifecycle interface, real
and synthetic benchmarks use the same runner, and every run writes
reproducibility artifacts that can be aggregated into leaderboard CSV/JSON files
and plots. The primary reported result is a verified Split CIFAR-10 suite.

![Split CIFAR-10 headline benchmark leaderboard](docs/assets/split_cifar10_headline/leaderboard.png)

## Project Scope

- Config-driven benchmark runner for single runs and multi-method suites.
- Implemented baseline fine-tuning, EWC, reservoir replay, LwF, DER++, and A-GEM.
- Deterministic synthetic CI benchmark plus real MNIST and CIFAR-10 task streams.
- Artifact tracking for config snapshots, metadata, JSONL events, CSV matrices,
  checkpoints, MLflow runs, aggregate reports, and plots.
- Python package, CLI, Dockerfile, Makefile, Ruff, pytest coverage, and GitHub
  Actions matrix across Python 3.10, 3.11, and 3.12.

## Quickstart

```bash
git clone https://github.com/1Utkarsh1/Continual-Learning.git
cd Continual-Learning

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,experiment,report]"

ruff check .
ruff format --check .
pytest --cov
cl-bench run --config-name smoke --method baseline --epochs 1 --device cpu
```

Or use the project automation:

```bash
make setup
make verify
make benchmark
```

## CLI

Run one benchmark:

```bash
cl-bench run --config-name smoke --method replay --epochs 1 --device cpu
```

Run the headline Split CIFAR-10 suite with local MLflow tracking and plots:

```bash
cl-bench suite \
  --config-name split_cifar10_headline \
  --methods baseline ewc replay lwf derpp agem \
  --seeds 13 21 \
  --tracking both \
  --report-dir docs/assets/split_cifar10_headline \
  --title "Split CIFAR-10 Headline Benchmark"
```

Use Hydra/OmegaConf-style overrides for quick experiments:

```bash
cl-bench suite \
  --config-name split_cifar10_headline \
  --methods baseline derpp \
  --seeds 13 \
  --tracking json \
  training.epochs=1 strategy.replay_buffer_size=500
```

Inspect local experiment runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Aggregate existing run directories or MLflow artifact exports:

```bash
cl-bench report \
  --runs runs \
  --output-dir reports/local \
  --title "Local continual-learning report"
```

## Verified Headline Benchmark

Local verification on 2026-05-25 used Python 3.11.15, PyTorch 2.12.0,
torchvision 0.27.0, NumPy 2.4.6, Hydra 1.3.2, MLflow 3.12.0, Ruff 0.15.14,
pytest 9.0.3, and Matplotlib 3.10.9.

Command:

```bash
cl-bench suite --config-name split_cifar10_headline --methods baseline ewc replay lwf derpp agem --seeds 13 21 --tracking both --report-dir docs/assets/split_cifar10_headline --title "Split CIFAR-10 Headline Benchmark"
```

The headline benchmark uses real CIFAR-10 images, five class-incremental tasks,
2,500 training examples per task, 1,000 test examples per task, two seeds,
5 epochs per task, a compact residual CIFAR ConvNet, and a 5,000-example replay
memory budget where applicable. It is a reproducible benchmark, not a paper
leaderboard claim.

| Method | Average final accuracy | Average forgetting | Mean runtime |
| --- | ---: | ---: | ---: |
| DER++ | 51.15% +- 3.95% | 34.06% +- 4.74% | 578.7s |
| replay | 41.99% +- 0.27% | 45.27% +- 1.73% | 547.4s |
| LwF | 16.53% +- 0.13% | 76.71% +- 0.09% | 224.3s |
| A-GEM | 14.37% +- 0.39% | 79.34% +- 0.96% | 516.3s |
| baseline | 14.06% +- 0.10% | 79.14% +- 1.39% | 181.0s |
| EWC | 12.12% +- 0.74% | 69.20% +- 3.02% | 223.1s |

Generated report artifacts live in
[`docs/assets/split_cifar10_headline`](docs/assets/split_cifar10_headline/README.md).

## Architecture

```text
src/cl_bench/
  cli.py             # run, suite, report, config discovery, and overrides
  config.py          # TaskSpec, ExperimentConfig, BenchmarkResult
  datasets.py        # synthetic, MNIST-family, and CIFAR-10 task construction
  experiments.py     # seeded run orchestration and evaluation loop
  metrics.py         # accuracy, forgetting, transfer, and summary metrics
  models.py          # linear, MLP, small CNN, and CIFAR residual ConvNet factory
  reporting.py       # run aggregation, leaderboard CSV/JSON, and plots
  tracking.py        # JSON/JSONL/CSV artifacts and optional MLflow logging
  strategies/        # baseline, EWC, replay, LwF, DER++, and A-GEM
configs/
  smoke.yaml              # fast deterministic CPU benchmark
  split_mnist_quick.yaml  # bounded real MNIST suite for local CPU runs
  split_mnist.yaml        # full five-task MNIST stream
  split_cifar10_headline.yaml # verified CIFAR-10 benchmark used in the README
docs/
  BENCHMARK_CARD.md       # scope, metrics, limitations, reproducibility
tests/                    # unit and integration coverage
```

## Run Artifacts

Each run is written to `runs/<benchmark>_<method>_<timestamp>/` and contains:

- `config.yaml`: exact config snapshot.
- `run_metadata.json`: seed, device, and git commit when available.
- `metrics.jsonl`: event-level training and evaluation metrics.
- `metrics.json`: final run summary.
- `accuracy_matrix.{json,csv}` and `forgetting_matrix.{json,csv}`.
- Optional `checkpoints/final_model.pt`.
- Optional MLflow run entries with params, metrics, tags, and artifacts.

The report command writes:

- `leaderboard.csv`
- `summary.json`
- `README.md`
- `leaderboard.png`
- `retention_curves.png`
- `accuracy_matrices.png`

Generated `data/`, `runs/`, `results/`, logs, checkpoints, and NumPy arrays are
ignored by git. Curated README assets under `docs/assets/` are intentionally kept.

## Engineering Notes

- EWC estimates the empirical Fisher from per-sample log-likelihood gradients and
  normalizes by the actual number of samples used.
- Replay uses reservoir sampling so a bounded buffer represents the full observed
  stream instead of only the newest examples.
- LwF stores a frozen teacher after each task and combines supervised loss with
  temperature-scaled KL distillation.
- DER++ stores replay logits online and combines current CE, replay CE, and
  logit-matching losses.
- A-GEM projects conflicting gradients against replay-memory reference gradients.
- Best validation checkpoints are deep-copied before restoration to avoid mutable
  `state_dict` aliasing bugs.
- The suite/report layer separates expensive benchmark execution from cheap,
  repeatable analysis over saved metrics.
