from __future__ import annotations

import numpy as np

from cl_bench.metrics import compute_forgetting, matrix_to_jsonable, summarize_accuracy


def test_forgetting_uses_best_previous_accuracy() -> None:
    matrix = np.array(
        [
            [80.0, np.nan, np.nan],
            [75.0, 70.0, np.nan],
            [78.0, 65.0, 90.0],
        ]
    )

    forgetting = compute_forgetting(matrix)

    assert forgetting[0, 0] == 0.0
    assert forgetting[1, 0] == 5.0
    assert forgetting[2, 0] == 2.0
    assert forgetting[2, 1] == 5.0
    assert np.isnan(forgetting[0, 1])


def test_summary_and_json_matrix_are_nan_safe() -> None:
    matrix = np.array([[50.0, np.nan], [40.0, 75.0]])

    summary = summarize_accuracy(matrix)
    json_matrix = matrix_to_jsonable(matrix)

    assert summary["average_final_accuracy"] == 57.5
    assert summary["average_forgetting"] == 10.0
    assert json_matrix == [[50.0, None], [40.0, 75.0]]
