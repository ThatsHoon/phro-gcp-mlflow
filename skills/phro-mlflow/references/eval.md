# Evaluation rules

Gate: each metric in `project.json metrics[]` has `dir`, `target` and `gate.min_delta`. A criterion is decided on a 95% bootstrap CI over `resample_unit` (resample count from `stats.bootstrap`), computed from `units.jsonl` by `phro.py ci`:
- whole CI on the pass side -> pass; whole CI on the fail side -> fail; straddles -> undecided.
- any fail -> `미달`; all pass -> `통과`; otherwise `보류` (not decided: more data or a seed rerun, not a verdict).
- Gate results are the `gate_<set>` tags. The tag is authoritative over point estimates in run descriptions or training logs.

Eval sets: choose on `eval_sets.select`, judge finalists on `eval_sets.heldout` within `look_budget` (`audit` reports excess looks), `eval_sets.final` once at the end.

Known validity limits (add pack-specific ones to `domain.md`)
| Limit | Handling |
|---|---|
| Proxy metric of an intermediate stage | confirm a pass end to end (output -> next stage -> final KPI) before calling it progress |
| Heldout set reused many times | finalists only; watch `look_budget`; final set once |
| Few examples per tier or bucket | tiers with a reference count below 30 are reported as "표본 부족" (too few), not judged |
| Small gold set | judge on the large split; the small set is a path check |
| Seed variance larger than sampling noise | compare data versions or formats with at least `stats.seeds_min` seeds per arm: seed mean [min, max] plus same-seed paired deltas (`ci`). Flag a seed whose training loss did not converge instead of averaging it silently |
| Synthetic eval data | also check the real-condition set that `domain.md` names; reject a change that worsens it beyond the tolerance written there |
| Point estimates in logs vs gate | logs may lack blockers or resampling; small differences are expected |
