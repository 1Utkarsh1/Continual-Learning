# Calibrated Anchor Replay Paper Scaffold

This directory contains the manuscript scaffold and reproducibility checklist for
turning the benchmark into a research paper.

## Research Claim Gate

The paper may claim a memory-accuracy Pareto improvement only if CAR beats DER++,
ER-ACE, and replay at matched memory budgets on at least two full-data protocols.
Rows that differ in memory budget, backbone, epochs, task order, or seed count
must be described as protocol variants rather than direct wins.

## Primary Experiment Commands

```bash
cl-bench suite --config-name paper/split_cifar10_full --methods replay derpp er_ace gdumb car bic icarl x_der_lite --seeds 13 21 34 55 89 --memory-budgets 200 500 1000 2000 5000 --tracking both --paper --report-dir docs/paper/assets/split_cifar10_full --title "Split CIFAR-10 Full-Data Paper Protocol"
cl-bench suite --config-name paper/split_cifar100_full --methods replay derpp er_ace gdumb car bic icarl x_der_lite --seeds 13 21 34 55 89 --memory-budgets 200 500 1000 2000 5000 --tracking both --paper --report-dir docs/paper/assets/split_cifar100_full --title "Split CIFAR-100 Full-Data Paper Protocol"
cl-bench suite --config-name paper/split_tinyimagenet --methods replay derpp er_ace gdumb car bic icarl x_der_lite --seeds 13 21 34 --memory-budgets 500 1000 2000 5000 10000 --tracking both --paper --report-dir docs/paper/assets/split_tinyimagenet --title "Split TinyImageNet Paper Protocol"
```

## Manuscript Status

- `manuscript.tex`: outline and claims skeleton.
- `reproducibility_checklist.md`: run and reporting checklist.
- `claims_table.md`: public claim discipline before any SOTA wording.
