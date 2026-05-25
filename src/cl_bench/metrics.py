from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy as a percentage."""

    if targets.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item() * 100.0)


def compute_forgetting(accuracy_matrix: np.ndarray) -> np.ndarray:
    """Compute best-so-far forgetting for every evaluated task."""

    matrix = np.asarray(accuracy_matrix, dtype=float)
    forgetting = np.full(matrix.shape, np.nan, dtype=float)

    for train_step in range(matrix.shape[0]):
        for task_id in range(matrix.shape[1]):
            current = matrix[train_step, task_id]
            if np.isnan(current) or task_id > train_step:
                continue
            if train_step == task_id:
                forgetting[train_step, task_id] = 0.0
                continue

            previous = matrix[:train_step, task_id]
            if np.isnan(previous).all():
                forgetting[train_step, task_id] = 0.0
                continue
            best_previous = float(np.nanmax(previous))
            forgetting[train_step, task_id] = max(0.0, best_previous - current)

    return forgetting


def summarize_accuracy(
    accuracy_matrix: np.ndarray,
    initial_accuracy: np.ndarray | None = None,
    pre_task_accuracy: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize continual-learning accuracy and forgetting metrics."""

    matrix = np.asarray(accuracy_matrix, dtype=float)
    final_row = matrix[-1]
    final_seen = final_row[~np.isnan(final_row)]
    average_final_accuracy = float(np.mean(final_seen)) if final_seen.size else 0.0

    diagonal = np.diag(matrix)
    learned = diagonal[~np.isnan(diagonal)]
    average_learning_accuracy = float(np.mean(learned)) if learned.size else 0.0

    forgetting = compute_forgetting(matrix)
    if matrix.shape[0] > 1:
        final_forgetting = forgetting[-1, : matrix.shape[0] - 1]
        final_forgetting = final_forgetting[~np.isnan(final_forgetting)]
        average_forgetting = float(np.mean(final_forgetting)) if final_forgetting.size else 0.0
    else:
        average_forgetting = 0.0

    if matrix.shape[0] > 1:
        previous_task_ids = range(matrix.shape[0] - 1)
        transfers = [
            matrix[-1, task_id] - matrix[task_id, task_id]
            for task_id in previous_task_ids
            if not np.isnan(matrix[-1, task_id]) and not np.isnan(matrix[task_id, task_id])
        ]
        backward_transfer = float(np.mean(transfers)) if transfers else 0.0
    else:
        backward_transfer = 0.0

    forward_transfer = compute_forward_transfer(initial_accuracy, pre_task_accuracy)

    return {
        "average_final_accuracy": average_final_accuracy,
        "average_learning_accuracy": average_learning_accuracy,
        "average_forgetting": average_forgetting,
        "backward_transfer": backward_transfer,
        "forward_transfer": forward_transfer,
    }


def compute_forward_transfer(
    initial_accuracy: np.ndarray | None,
    pre_task_accuracy: np.ndarray | None,
) -> float:
    """Compute mean pre-training improvement on future tasks before learning them."""

    if initial_accuracy is None or pre_task_accuracy is None:
        return 0.0
    initial = np.asarray(initial_accuracy, dtype=float)
    pre_task = np.asarray(pre_task_accuracy, dtype=float)
    if initial.shape != pre_task.shape or initial.size <= 1:
        return 0.0
    transfers = [
        pre_task[task_id] - initial[task_id]
        for task_id in range(1, initial.size)
        if not np.isnan(initial[task_id]) and not np.isnan(pre_task[task_id])
    ]
    return float(np.mean(transfers)) if transfers else 0.0


def matrix_to_jsonable(matrix: np.ndarray) -> list[list[float | None]]:
    """Convert a NumPy matrix to JSON-safe nested lists."""

    result: list[list[float | None]] = []
    for row in np.asarray(matrix, dtype=float):
        result.append([None if np.isnan(value) else float(value) for value in row])
    return result


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0
