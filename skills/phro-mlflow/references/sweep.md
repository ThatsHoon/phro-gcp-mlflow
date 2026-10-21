# Sweeps

A sweep = runs sharing a `sweep` name (comma-separate if a run belongs to two). Write each run's manifest entry BEFORE launching, with `phro.py manifest set <run> key=value ...` (the manifest lives on the server at `server.manifest`; no scp, no race). It becomes MLflow tags at sync. Fields:

| field | content |
|---|---|
| `variation` | short name of this arm |
| `compare` | run(s) it is judged against |
| `sweep` | sweep name |
| `axis` | the ONE thing that changes |
| `fixed` | data version, eval sets, steps, seed, upstream model |
| `pick` | rule decided now, naming the metric key (`<set>/<name>`, min or max) |
| `hypothesis` | expected direction and why, citing insight IDs |
| `budget` | `short` or `full` (also a tag `decide` reads; no tag counts as full), plus GPU and hours |
| `idea` | `IDEA-NNN` if the run tests one |
| `intro` | 실험 서론, written before launch (pattern below) |
| `result` | 실험 결과, written after the run (pattern below) |

Example: `phro.py manifest set <run> "axis=<one change>" "compare=<run>" "sweep=<sweep>" "intro=<text>"`. `audit` checks only `intro`, `result`, `review` and the `기대:`/`Expected:` marker; the other fields are your own checklist.

Every run gets two write-ups (in `project.json language`). Both appear as tags; `audit` flags finished runs without them and an `intro` without `기대:` (or `Expected:`).

### `intro` = 실험 서론 (before launch)
Order, with real numbers and insight IDs:
1. Evidence: which earlier run, which metric key, which value (set name).
2. Inference: what you concluded from it, and why.
3. Change: what changed, from what to what (must equal `axis`).
4. Purpose: which question this run answers.
5. Expected result with a threshold, and what it means if it does not happen.

Pattern: "이전 <run>에서 <지표 키>가 <값>(<평가셋>)이어서 <판단>이라고 판단하여 <무엇>을 <A>에서 <B>로 바꾼 버전이며, <무엇>이 <어떤지> 확인하기 위함. 기대: <지표 기준>. 아니면 <의미>."
Why: months later the tag alone must explain why the run exists, and the stated expectation is what the result is checked against.

### `result` = 실험 결과 (after the run finishes)
Write it after `report` or `compare` and the diagnosis (`references/diagnosis.md`), with numbers from `phro.py`, never from memory, in this order:
1. Outcome: key numbers on the select set with 95% CI, vs the compare run, and the `decide` verdict once the family has `stats.seeds_min` seeds (before that: "seeds missing", no verdict).
2. Against the intro: was the expected result met? Say plainly when the hypothesis was wrong.
3. Cause: why the metrics moved (per-tier or bucket or probe evidence), with the insight ID added or updated.
4. Meaning: what this changes in the design (keep, drop, freeze with `reopen_if`).
5. Next: the experiment this points to, or "none".

Pattern: "<평가셋>에서 <지표 키> <값> [CI], <지표 키> <값> [CI] → <판정>. 비교 런 <run> 대비 <차이>. 기대(<서론의 기대>)는 <맞음/틀림>. 원인: <근거>(I-NNN). 따라서 <설계 결정>. 다음: <실험>."
Wrong expectations are information: record them in the insight ledger too.

## Rules (each from a past miss)
1. One axis per sweep. Pairs (with/without) are sweeps too. Hypothesis first; `compare --sweep` shows whether it held.
2. Adopt a change only through `phro.py decide <compare family> <arm family>`. Runs are named `<family>_s<N>`; seeds pair by `N`, finished runs (`status=done`) only. Deciding the same arm again with more seeds replaces its earlier verdict. It needs at least `stats.seeds_min` pairs (default 4) and `stats.min_units` common units (default 30). For every metric of the stage on the select set, it computes the mean of the per-seed deltas with a t interval over seed and unit variance. accept = every metric on the pass side of `gate.min_delta`; reject = any metric on the fail side; undecided = otherwise (more seeds or units, not a failure). Every decision goes to `.phro/ledger/decisions.jsonl`. A single-run `ci` never decides.
3. An arm with `budget=short` runs can only reach `accept-short`. Rerun it at `budget=full` as a new family, then `decide <compare family> <new family> --reverify <short family>`. Only that `accept` adopts it (tag `accepted=yes`). `audit` lists short passes not yet re-verified.
4. When 3 or more accepted changes on one stage and data version have not been tested by leave-one-out, `phro.py loo --stage S --data V` lists one arm per change (the stack without that change). Run them as a sweep named `loo_<...>` through plan.md. Close each with `decide <arm> <stack> --loo <change family>`. reject drops the change; undecided means no contribution is shown: add seeds, or drop it for the simpler stack. Do this before calling the stack a baseline. `audit` reports the backlog.
5. Choose on the select set, judge on the heldout set within `look_budget`, use the final set once. `look_budget` counts run families (all seeds of a family = one look) per data version. `compare` defaults to the select set. Check `phro.py splitcheck <run>` once per data version: the eval sets must share no unit ids.

## Launch checklist
0. `intro` written for every run in the sweep. Legacy review done and recorded (`references/legacy-review.md`); after launch, `diff` vs the compare run.
1. Script header comments (purpose, what changed, compare target) become the run description.
2. Output goes to `<server.root>/runs/<stage dir>/<run name>`; run name = folder name, unique. Write the standard folder (`references/contract.md`) so sync reads it.
3. Wait and finish conditions use result files (`DONE`), never `pgrep`. One start condition per run (its GPU and the file that frees it), never one for a whole batch.
4. One training per GPU unless memory allows more (`status` shows who uses a GPU).
5. Training noise or augmentation uses training-only sources and a seed other than the eval seed.
6. User approves the launch unless the item is in `.phro/plan.md`. Do this before step 7.
7. Launch, or queue (queueing is launching; not allowed when `server.read_only` is true, see SKILL.md): `phro.py queue add <GPU> <script>` (launched within 5 minutes when that GPU has no compute process). Send launches in the background and confirm with `status` in a separate call; arm a `DONE` watcher at once (`references/lessons.md`).

## Two layers: method screening, then HPO
1. **Method screening** (one-axis sweeps above): find a method with signal (target metric up without a blocking metric exploding, or an undecided result leaning positive). Hyperparameters cannot fix a structural failure, so do not HPO a method that fails structurally.
2. **HPO** only for the chosen method, with Optuna TPE and a pruner (for example a percentile pruner that drops the bottom third at early rungs and never prunes the first few trials).
   - Space: ranges for learning rate (log), loss weights, steps and other numeric knobs; pin known-good values.
   - Objective: the gate written as one number on the select set (gain minus penalties for blocking criteria), maximize. Same direction as the gate.
   - Pruning compares rungs, so pin the step count (schedules that decay to the end look worse at early rungs). The first trial is the screening baseline as a same-objective reference.
   - Never look at the heldout or final set during HPO. Afterwards rerun the best params as a family of `stats.seeds_min` seeds and `decide` it against the screening baseline family; only after `accept`, the heldout set once for that finalist.
   - Dry-run the flow without GPUs first. Record the study (what mattered, best params) as one insight entry.

After a sweep finishes: `report --run <name>`, `compare --sweep <name>`, `decide <compare family> <arm family>` for each arm with enough seeds, then for each run `manifest set <run> gate_<set>=<verdict>` (the per-metric verdict `decide` printed, on a pack that is not read_only), its `result` and its insight entry.
