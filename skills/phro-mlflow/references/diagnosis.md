# Diagnosis: comprehensive analysis of a run or sweep

Goal: explain WHY the metrics moved, down to a mechanism you can test, so the next experiment attacks the cause instead of a symptom. The output feeds the run's `result`, the insight ledger, and the next sweep's `intro`.

## When
- Every finished run or sweep, before writing `result` and the insight entry.
- At interim evals when a pattern shows across runs (same step, same set).
- Whenever the user asks why or what is wrong ("왜", "문제가 뭐야", "원인").

## Data rules
Numbers only: `phro.py report`, `compare`, `ci`, eval lines in run logs, probe outputs. Diagnostic scripts may read train, dev and prediction files on the server but print only counts and ratios, never data rows.

## Steps
1. **Blocking criterion.** Which gate criterion fails, by how much, relative to the gain. Example: recall up 2 points but false positives up 6 points, so false positives are the problem, not recall.
2. **Ablation ladder.** List every run that differs from this one in exactly one component (compare tags, sweeps, frozen runs). Table: component | runs with / without | effect on the blocking metric. Same step and same eval set only; never mix an interim step with final, or the select set with the heldout set.
3. **Rule out.** For each sweep, state what it eliminated, with the number ("learning rate is not the cause: the error stays 3 points above target at every rate tried"). A bundled change is split into its parts; parts not yet isolated are the remaining suspects.
4. **Ceiling and input bound.** Compare the best-achievable-from-input metric, the baseline input metric and the output metric. Is the model selecting from its input or generating beyond it? Probe numbers bound what any training can reach.
5. **Mechanism: data -> objective -> behaviour.** Ask what the training signal rewards. Usual traps:
   - targets holding information the input cannot support, so the model learns to guess;
   - sampling that shifts the prior (for example mostly positive examples);
   - loss weights that reward emitting one token class;
   - eval blind spots (a metric without blockers, tiny sets).
   Pick the simplest story consistent with ALL moves. If two fit, name both.
6. **Evidence status per claim.** Mark each one 측정 (a number in hand) or 추정 (inference). Never present an inference as measured.
7. **Discriminating diagnostics.** For each open hypothesis, one cheap check (CPU, counts on train or dev files) with the value expected under each hypothesis. Run read-only checks directly; propose anything needing a GPU or training (launching needs the user's OK).
8. **Implication.** Look in `.phro/ledger/idea-bag.md` for ideas targeting this limit; if none fits, or this is the second closed sweep on it, follow `references/breakthrough.md`. State what to do next and what NOT to do (for example "HPO cannot fix a data or objective problem", "more blockers treat a symptom"). A diagnostic worth reusing goes into `probe.md` through the lessons escalation.

## Answer format (in `project.json language`)
1. One line: the problem, i.e. the blocking metric key with numbers.
2. Narrowing list: each item = claim + number + 측정/추정.
3. Mechanism hypothesis and the evidence that fits it (including the ceiling comparison).
4. Diagnostics proposed or run, with expected outcomes.
5. What it changes in the plan (next axis, or what to stop).

## After
Write `result` from steps 1-6. Record the mechanism as an insight (`references/insights.md`) with its status and update the open-limits table. Put the diagnostics and their expected values in the next sweep's `intro`, so the next result can say which hypothesis held.
