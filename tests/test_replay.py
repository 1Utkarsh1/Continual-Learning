from __future__ import annotations

import torch

from cl_bench.strategies.replay import ReservoirReplayBuffer


def test_reservoir_replay_buffer_is_bounded_and_not_last_n() -> None:
    buffer = ReservoirReplayBuffer(capacity=3, seed=3)

    for value in range(10):
        inputs = torch.full((1, 2), float(value))
        targets = torch.tensor([value])
        buffer.add_batch(inputs, targets)

    stored_targets = [sample.target for sample in buffer.samples]

    assert len(buffer) == 3
    assert buffer.seen_count == 10
    assert stored_targets != [7, 8, 9]


def test_replay_sampling_returns_tensors() -> None:
    buffer = ReservoirReplayBuffer(capacity=5, seed=0)
    buffer.add_batch(torch.randn(4, 1, 8, 8), torch.tensor([0, 1, 2, 3]))

    inputs, targets = buffer.sample(batch_size=2)

    assert inputs.shape == (2, 1, 8, 8)
    assert targets.shape == (2,)
    assert targets.dtype == torch.long
