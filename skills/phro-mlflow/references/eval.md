# Evaluation rules

Gate: each metric in `project.json metrics[]` has `dir`, `target` and `gate.min_delta`. A change is decided between two run families (`<family>_s<N>`, same seeds) by `phro.py decide` (one metric: `ci --pairs`), from `units.jsonl`:
- per seed pair (finished runs only), the metric delta (arm minus base) on the units every run shares; the mean delta gets a 95% t interval (df = pairs - 1) whose variance is the seed variance of the deltas plus the unit-resampling variance of the mean delta (`stats.bootstrap` resamples of the `unit` ids in `units.jsonl`, one id per `resample_unit`).
- whole interval on the pass side of `min_delta` -> `통과`; whole interval on the fail side -> `미달`; straddles -> `보류` (not decided: more seeds or units, not a verdict). Fewer than `stats.seeds_min` pairs or `stats.min_units` units -> `보류` with the reason.
- over the metrics of the stage on the select set: any `미달` -> reject; all `통과` -> accept (`accept-short` if the arm ran `budget=short`); otherwise undecided. `.phro/ledger/decisions.jsonl` is the record; `gate_<set>` tags copy the per-metric verdict and win over point estimates in run descriptions or training logs.
- Single-run `ci` intervals cover unit sampling only. Seed noise is often larger, so they never decide.

Eval sets: choose on `eval_sets.select`, judge finalists on `eval_sets.heldout` within `look_budget` (one look = one run family with metrics on that set, counted per data version; `audit` reports excess looks), `eval_sets.final` once at the end. `splitcheck <run>` confirms the sets share no unit ids.

Known validity limits (add pack-specific ones to `domain.md`)
| Limit | Handling |
|---|---|
| Proxy metric of an intermediate stage | confirm a pass end to end (output -> next stage -> final KPI) before calling it progress |
| Heldout set reused many times | finalists only; watch `look_budget`; final set once |
| Few examples per tier or bucket | tiers with a reference count below 30 are reported as "표본 부족" (too few), not judged |
| Small gold set | judge on the large split; the small set is a path check |
| Seed variance larger than sampling noise | compare data versions or formats with at least `stats.seeds_min` seeds per arm: seed mean [min, max] (`baseline`) plus `decide` on the same-seed pairs. Flag a seed whose training loss did not converge instead of averaging it silently |
| Synthetic eval data | also check the real-condition set that `domain.md` names; reject a change that worsens it beyond the tolerance written there |
| Point estimates in logs vs gate | logs may lack blockers or resampling; small differences are expected |
