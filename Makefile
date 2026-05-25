PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

.PHONY: setup lint format test smoke suite benchmark build verify clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev,experiment,report]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

test:
	$(BIN)/pytest --cov

smoke:
	$(BIN)/cl-bench run --config configs/smoke.yaml --method baseline --epochs 1 --device cpu

suite:
	$(BIN)/cl-bench suite --config configs/smoke.yaml --methods baseline ewc replay lwf derpp agem --seeds 7 --epochs 1 --device cpu --report-dir reports/smoke

benchmark:
	$(BIN)/cl-bench suite --config-name split_cifar10_headline --methods baseline ewc replay lwf derpp agem --seeds 13 21 --tracking both --report-dir docs/assets/split_cifar10_headline --title "Split CIFAR-10 Headline Benchmark"

build:
	$(BIN)/python -m build

verify: lint test smoke build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
