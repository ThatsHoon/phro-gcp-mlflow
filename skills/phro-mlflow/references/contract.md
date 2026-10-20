# Contract: pack keys, MLflow contract, standard result folder

The engine reads MLflow, not result files. Anything that follows the MLflow table below works, with or without the standard folder.

## Pack keys (`.phro/project.json`)
Only values some script reads are in it. Validate with `phro.py check-pack`.
| key | meaning |
|---|---|
| `name`, `language` | project name; language of reports to the user |
| `goal.kpis[]` | `key`, `target`, `dir` (max/min), `source` (where the target comes from) |
| `stages[]` | `name`, `experiment` (MLflow experiment), `depends_on` (stage names; defines `cascade`) |
| `metrics[]` | `key` (`<set>/<name>`), `stage`, `dir`, `target`, `resample_unit`, `agg`, `gate.min_delta` |
| `metrics[].agg` | `{"type": "mean", "field": F}` mean of unit values; `{"type": "f1", "tp", "fp", "fn"}` sums unit counts, then micro F1 (not the mean of unit F1s) |
| `eval_sets` | `select`, `heldout`, `final`, `look_budget` ({set: max runs that look at it}) |
| `server` | `host`, `allow_hosts`, `root`, `mlflow_port`, `gpus`, `venv`; optional `manifest` (default `<root>/runs.json`), `queue` (default `<root>/gpu_queue.txt`), `read_only` (default false), `done_globs` (default `["runs/**/DONE"]`), `run_name_rules` (default `[["(?:.*/)?([^/]+)/DONE$", "\\1"]]` (JSON escape); empty replacement = script end marker) |
| `egress` | `forbidden_terms` (never in searches, URLs or subagent prompts), `note` |
| `stats` | `seeds_min` (default 4), `bootstrap` (default 1000) |
| `tag_alias`, `derived_params` | contract tag name -> project tag name; result-like params that `diff` ignores. Defaults `{}` and `[]` |
The data version is a run tag (`data_version`), not a pack key.

## MLflow contract
| item | rule |
|---|---|
| experiment | `stages[].experiment` |
| run name | unique, equal to the result folder name; seeds end in `_s<N>` |
| tags | `status` (running/done/smoke/frozen/aborted), `stage`, `sweep` (comma list), `variation`, `compare`, `axis`, `fixed`, `pick`, `hypothesis`, `idea`, `insight`, `review`, `intro`, `result`, `data_version`, `budget` (short/full), `accepted` (yes), `gate_<set>`, `reopen_if` (when frozen) |
| metrics | `<eval set>/<name>`; compare only identical keys. A run that records its own CI uses `<set>/ci_lo_<name>`, `<set>/ci_hi_<name>` |
| artifact `units.jsonl` | one line per unit: `{"set", "unit", <fields the agg reads>}`; `ci` computes CIs and paired deltas from it |
| leave-one-out | sweep name `loo_<...>` |
| finished | a `DONE` file in the result folder. Never `pgrep` |

## Standard result folder (read by `phro_sync.py`)
`run.json` (`name`, `stage`, `params`, `tags`), `metrics.jsonl` (`{"step", "set", "key", "value"}`; the last value uses `"step": "final"`), `units.jsonl`, `DONE` (written last). Manifest entries (`phro.py manifest set NAME key=value`) become tags at sync, so `intro`, `result`, `insight`, `review` can be written before or after the run.

Training script snippet (unit fields `tp`, `fp`, `fn` match an `f1` agg):
```python
import json, pathlib
d = pathlib.Path("runs/<stage dir>/<run>"); d.mkdir(parents=True, exist_ok=True)
(d / "run.json").write_text(json.dumps({"name": "<run>", "stage": "<stage>", "params": {"lr": 2e-4, "seed": 1},
    "tags": {"status": "done", "data_version": "<V>", "sweep": "<sweep>", "variation": "<variation>"}}))
(d / "metrics.jsonl").write_text(json.dumps({"step": "final", "set": "<set>", "key": "<name>", "value": 0.5}) + "\n")
(d / "units.jsonl").write_text("".join(json.dumps({"set": "<set>", "unit": u, "tp": 1, "fp": 0, "fn": 0}) + "\n" for u in ("u1", "u2")))
(d / "DONE").write_text("")  # last, after every other file
```
