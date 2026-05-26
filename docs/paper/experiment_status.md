# Full Paper Experiment Status

Status: **not complete**.

## Requested Matrix

| Dataset | Seeds | Methods | Memory budgets | Planned runs |
| --- | --- | --- | --- | ---: |
| Split CIFAR-10 | 13, 21, 34, 55, 89 | joint, replay, DER++, ER-ACE, GDumb, CAR, BiC, iCaRL, X-DER-lite | 200, 500, 1000, 2000, 5000 | 225 |
| Split CIFAR-100 | 13, 21, 34, 55, 89 | joint, replay, DER++, ER-ACE, GDumb, CAR, BiC, iCaRL, X-DER-lite | 200, 500, 1000, 2000, 5000 | 225 |
| TinyImageNet | 13, 21, 34 | joint, replay, DER++, ER-ACE, GDumb, CAR, BiC, iCaRL, X-DER-lite | 200, 500, 1000, 2000, 5000 | 135 |

Total planned full-data runs: **585**.

## Local Audit Result

- CUDA is not available on the current machine.
- MPS is available.
- CIFAR-10 is present under `data/cifar-10-batches-py`.
- CIFAR-100 is present under `data/cifar-100-python`.
- TinyImageNet is present under `data/tiny-imagenet-200`.
- TinyImageNet layout was prepared with `scripts/prepare_tinyimagenet.py`; train
  and validation each expose 200 matching `ImageFolder` classes.
- Paper configs were loaded against the full local datasets:
  - CIFAR-10: 5 tasks, first task split 9,000 train / 1,000 val / 2,000 test.
  - CIFAR-100: 10 tasks, first task split 4,500 train / 500 val / 1,000 test.
  - TinyImageNet: 20 tasks, first task split 4,500 train / 500 val / 500 test.
- The full matrix has not been completed, so CAR is **not** proven to improve the
  memory-accuracy Pareto frontier.

## Required Completion Command

```bash
bash docs/paper/full_suite_commands.sh
```

After the suite finishes, run:

```bash
cl-bench report --runs runs/paper --output-dir docs/paper/assets/all --title "Full Paper Suite" --paper
```
