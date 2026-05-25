# Reproducibility Checklist

- Use only full-data paper configs for paper claims.
- Run matched memory budgets before comparing methods.
- Use at least five seeds for CIFAR-10 and CIFAR-100.
- Use at least three seeds for TinyImageNet if compute is tight.
- Log JSON artifacts and MLflow artifacts for every run.
- Keep raw `runs/`, `mlruns/`, datasets, and checkpoints out of git.
- Commit only curated paper assets, tables, and figures.
- Include per-seed CSVs and confidence intervals in every paper report.
- Separate classic from-scratch results from frozen-backbone results.
- Mark external Mammoth, Avalanche, and ContinualAI numbers with protocol caveats.
