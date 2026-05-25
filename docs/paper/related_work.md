# Related Work And Positioning

This project is positioned as a reproducible class-incremental continual-learning
benchmark and method scaffold, not as a claim that established ideas are new.

## Method Families

- Regularization methods: EWC penalizes movement away from parameters important
  to earlier tasks through an empirical Fisher estimate.
- Distillation methods: LwF keeps a frozen teacher to preserve prior responses.
- Replay methods: ER, DER++, ER-ACE, iCaRL-style exemplar replay, and X-DER-style
  dark-logit anchoring use stored examples or predictions to reduce forgetting.
- Gradient projection methods: A-GEM projects conflicting gradients against a
  replay-memory reference gradient.
- Memory/oracle baselines: GDumb and joint cumulative-data training test whether
  simple memory-heavy baselines explain apparent gains.

## External Frameworks

Avalanche, Mammoth, and ContinualAI baselines are the comparison anchors for
future protocol matching. Any external number must be labeled with its dataset,
backbone, task order, memory budget, epochs, and seed count before it is compared
with this repository's results.

## CAR Positioning

Calibrated Anchor Replay combines known replay ingredients: balanced exemplars,
stored logits, feature anchors, class prototypes, ER-ACE-style current-task
masking, and post-task calibration. Its research value is not claiming that each
ingredient is new; the hypothesis is that this combination can improve the
memory-accuracy Pareto frontier under matched protocols.
