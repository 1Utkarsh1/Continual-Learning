from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ExperimentTracker:
    """Writes reproducibility artifacts for one benchmark run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "metrics.jsonl"

    def log_event(self, event: dict[str, Any]) -> None:
        payload = {"time_utc": utc_now(), **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_json(self, name: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def write_matrix_csv(self, name: str, matrix: np.ndarray) -> Path:
        path = self.run_dir / name
        np.savetxt(path, matrix, delimiter=",", fmt="%.6f")
        return path


def create_run_dir(output_dir: str | Path, benchmark_name: str, method: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = _safe_slug(benchmark_name)
    safe_method = _safe_slug(method)
    return Path(output_dir) / f"{safe_name}_{safe_method}_{timestamp}"


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip(
        "-"
    )


class MLflowRunLogger:
    """Optional local MLflow logger layered on top of JSON artifacts."""

    def __init__(
        self,
        tracking_uri: str | Path,
        experiment_name: str,
        run_name: str,
        enabled: bool,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled
        self._mlflow: Any | None = None
        self._active_run: Any | None = None

    def __enter__(self) -> MLflowRunLogger:
        if not self.enabled:
            return self
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking was requested but mlflow is not installed. "
                'Install with: python -m pip install -e ".[experiment]"'
            ) from exc

        self._mlflow = mlflow
        mlflow.set_tracking_uri(_normalize_mlflow_uri(self.tracking_uri))
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._mlflow is not None and self._active_run is not None:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        for key, value in _flatten_params(params).items():
            self._mlflow.log_param(key, value)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        for key, value in tags.items():
            if value is not None:
                self._mlflow.set_tag(key, str(value))

    def log_environment(self) -> None:
        if self._mlflow is None:
            return
        self.set_tags(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available(),
            }
        )

    def log_metrics(
        self, metrics: dict[str, float | int | str | None], step: int | None = None
    ) -> None:
        if self._mlflow is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(key, float(value), step=step)

    def log_artifacts(self, run_dir: str | Path) -> None:
        if self._mlflow is None:
            return
        self._mlflow.log_artifacts(str(run_dir))


def _normalize_mlflow_uri(uri: str | Path) -> str:
    value = str(uri)
    if value.startswith("sqlite:///"):
        db_path = Path(value.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return value
    if "://" in value:
        return value
    return Path(value).resolve().as_uri()


def _flatten_params(
    params: dict[str, Any], prefix: str = ""
) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, name))
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = json.dumps(value, sort_keys=True, default=str)
    return flattened
