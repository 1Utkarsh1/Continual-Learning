#!/usr/bin/env bash
set -euo pipefail

# Full matched-memory paper matrix. This is intentionally long-running.
# Run from the repository root. TinyImageNet is prepared idempotently before the
# long-running matched-memory matrix starts.

python scripts/prepare_tinyimagenet.py --data-dir data

python scripts/run_paper_suite.py \
  --stage cifar10 \
  --methods joint replay derpp er_ace gdumb car bic icarl x_der_lite \
  --memory-budgets 200 500 1000 2000 5000 \
  --tracking both

python scripts/run_paper_suite.py \
  --stage cifar100 \
  --methods joint replay derpp er_ace gdumb car bic icarl x_der_lite \
  --memory-budgets 200 500 1000 2000 5000 \
  --tracking both

python scripts/run_paper_suite.py \
  --stage tinyimagenet \
  --methods joint replay derpp er_ace gdumb car bic icarl x_der_lite \
  --memory-budgets 200 500 1000 2000 5000 \
  --tracking both
