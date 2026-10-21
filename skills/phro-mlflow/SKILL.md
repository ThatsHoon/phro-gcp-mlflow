---
name: phro-mlflow
description: Runs the experiment loop of a project that has a phro domain pack (.phro/) on one GPU VM with MLflow. Judges finished runs, compares runs and sweeps with paired bootstrap CIs, reports GPU and progress, diagnoses why metrics moved, plans probes and sweeps before new training, keeps the insight ledger and the open-limits table, fills idle GPUs from plan.md, and starts a breakthrough search when a limit stays stuck. Use it whenever the user asks about training results, progress, GPUs, comparisons, next experiments, ideas or MLflow, including Korean phrasings like "결과 어때", "끝났어?", "진행상황", "남는 gpu", "sweep", "다음 실험", "돌파구", "새 아이디어", "MLflow", and English phrasings like "results", "next experiment", "what is blocking us" - even if MLflow is not named. Query MLflow before quoting any training number.
---

# phro-mlflow

MLflow is the scoreboard. Read numbers from it through `phro.py`, judge with one rule set, and tie every decision to the goal chain in the pack. After a long chat or a context summary, re-query instead of recalling numbers.

Run commands as `python ${CLAUDE_PLUGIN_ROOT}/bin/phro.py <cmd>` from the project dir (the CLI finds `.phro/project.json` from the working dir). No pack found -> tell the user to run the phro-onboard skill first and stop. Reports to the user follow `project.json language`.

## 1. Goal chain (read first)
Read `.phro/domain.md`: the goal-chain table (stage, metric key, target, ceiling, what it hands to the next stage), hardware and model limits, data and privacy rules, terms, report format. The table holds no numbers on purpose. Refresh them with `phro.py baseline --data <data_version>`. Start every result report with one line: which cell of the chain moved, by how much, on which metric key.

## 2. Hard rules
- Numbers come only from `phro.py` output, never from memory or from reading result files. Name the metric key with every number and compare only the same key.
- The only server is `project.json server.host`. Never touch another host or instance; never stop jobs you did not start.
- Never stop, reset or delete VMs; never run gcloud (the guard blocks anything that is not read-only).
- Print and read numbers, counts and tags only, never data rows (transcripts, predictions, labels). Diagnostic scripts print counts and ratios.
- `server.read_only: true` means the VM is read-only: no launch, queue entry, process stop, file write or `manifest set` on the server. Result, insight and review text then go to the user and the local `.phro/ledger/`, and GPU-filling is reported as a proposal only.
- Launching training needs the user's OK, unless that exact item is already in `.phro/plan.md` (the user's approved queue). New methods, direction changes and deletions are always asked.
- Eval discipline: select on `eval_sets.select`, judge on `heldout` within `look_budget`, `final` once. See `references/eval.md`.
- Accept or reject a change only with `decide BASE ARM` (same-seed pairs, at least `stats.seeds_min`, on the select set). Never accept from a single run, a single-run `ci`, or a point estimate.
- A finished run missing from MLflow is a sync bug: run `audit` and fix it before reporting.
- Plain technical English in files this skill writes. No emoji, no em or en dashes, no AI-attribution lines in commits, docs, manifests or MLflow text.

## 3. Commands
| cmd | use |
|---|---|
| `status` | per GPU: running procs or FREE, mem, util; queue; waiting scripts |
| `list [--exp E]` | exact run, status, sweep and gate names; map the user's words to names |
| `report --run NAME` / `--since HOURS` | verdict lines, hypothesis, intro, result, compare-tag runs |
| `compare A B ... [--set S]` / `--sweep S` | same-key side by side, gates; sweep axis, fixed, pick, hypothesis. Default set is `eval_sets.select`; `--set <heldout>` counts as a look |
| `diff NEW --vs OLD` | effective parameter differences; beyond the declared axis = confound |
| `ci A B ... --metric KEY [--units DIR]` | single runs: per-run 95% bootstrap CI over units and the same-unit delta vs the first run (from `units.jsonl`). Unit noise only, so never a verdict |
| `ci BASE ARM --metric KEY --pairs [--units DIR]` | run families (`<family>_s<N>`): per-seed deltas, mean delta with a t interval over seed and unit variance, gate verdict (통과/미달/보류) |
| `decide BASE ARM [--set S] [--limit L-ID] [--idea IDEA-ID] [--note T] [--reverify SHORT-ARM \| --loo CHANGE-ARM]` | the accept decision: `ci --pairs` on every metric of the arm's stage on the select set; prints accept, accept-short (arm ran `budget=short`), reject or undecided and appends it to `.phro/ledger/decisions.jsonl` (local, so it works on read_only packs). `--reverify` marks a full-budget rerun of a short pass; `--loo` marks a leave-one-out (BASE = the stack without that change, ARM = the stack) |
| `loo --stage S --data V` | accepted changes not yet tested by leave-one-out, and the arm to build for each (stack minus that change) |
| `splitcheck RUN [--units DIR]` | unit ids shared between the select, heldout and final sets of one run (must be 0) |
| `baseline --data V` | per stage and metric: run families (seed suffix dropped), seeds, mean [min, max], gap to target |
| `cascade --stage S --data V` | finished downstream runs not on data version V = retrain list |
| `audit` | missing intro/result/review, finished but unsynced, GPU idle while scripts wait, look_budget exceeded (run families per data version), 3+ accepted changes not yet tested by leave-one-out, short-budget accepts not re-verified at full budget, frozen list |
| `manifest get NAME` / `manifest set NAME key=value ...` | run manifest on the server (intro, result, insight, review, axis, ...) |
| `queue add GPU SCRIPT` / `queue list` | append one job to the server GPU queue (launched by the queue cron only on a GPU with no compute process) / show it |
| `check-pack`, `leakcheck [dir]`, `setup` | pack validation, public-repo leak scan, server install (onboarding) |

`manifest set`, `queue add` and `setup` refuse when `server.read_only` is true (see the read-only rule above). Tunnel down -> the CLI prints the `ssh -N -L` command; run it, then retry. Contract and file formats: `references/contract.md`.

## 4. Workflows
- **Result question** -> `report --run` (or `--since 6`). Answer: chain line, table `run | stage | variation | set | metric (key) | CI | gate`, one line vs compare runs and vs the run's `intro` expectation (met or not), one line why (insight ID), one line next proposal (which free GPU, what). The gate cell comes from `decide` or `ci --pairs` on the run's family; a family with fewer than `stats.seeds_min` seeds has no gate yet (write `-`, and how many seeds are missing). Gate `보류` = CI straddles the threshold: say "not decided", not "failed".
- **Every finished run or sweep** -> diagnose (`references/diagnosis.md`), update the ledger (`references/insights.md`), then write the run's `result` and `insight` with `manifest set` (`references/sweep.md`). Once an arm family has `stats.seeds_min` seeds paired with its compare family, run `decide <compare family> <arm family> --limit <L-ID> --idea <IDEA-ID>`; only `accept` makes it the new compare run. If it tested an idea, update that idea in `.phro/ledger/idea-bag.md`. Read the ledger before designing the next sweep.
- **Why / what is wrong** -> `references/diagnosis.md`, in its answer format; run read-only diagnostics, propose GPU ones.
- **New experiment** -> stage section of `.phro/ledger/insights.md`, then `references/idea-bag.md` (ideas whose `targets` match the limit), then `references/probe.md` (input, labels, ceiling first), then `references/sweep.md` (define the sweep, write each run's `intro`), then `references/legacy-review.md` (independent review of inherited defaults), approval, launch, and `diff` vs the compare run. Check the `audit` frozen list so a dead end is not repeated.
- **Tuning numbers** -> only after method screening picked a method (`references/sweep.md` "Two layers").
- **Progress / free GPU** -> `status`; GPU idle while a script waits -> `audit`. Queueing counts as launching: queue every approved (or `plan.md`) ready script with `phro.py queue add <GPU> <script>` (launched by the queue cron only on a GPU with no compute process) so GPUs work when no session is open.
- **Stuck** (two sweeps on one limit closed with no gain, no idea targets it, or the user asks for new directions) -> `references/breakthrough.md`.
- **Upstream stage changed** -> `cascade --stage S --data V`, rerun its probes, retrain downstream in dependency order. Keep old versions' configs, evals and results; never overwrite them.
- **Evaluation doubts** -> `references/eval.md`.
- **Something failed or went idle** -> `references/lessons.md` (record; propose a rule when it repeats).
- **Foreman** (MCP `foreman_check`, `foreman_report`; Stop hook active while `.phro/foreman/ENABLED` exists; it blocks only the session id written in that file, an empty file or `claim` goes to the next session that stops, and an owner silent for 2 hours is taken over): when it blocks, judge each finished run against its `intro` expectation, fill idle GPUs from `.phro/plan.md`, then `foreman_report` (summary with metric keys and numbers, `next`, `blocked` when waiting on the user). Keep `plan.md` current.
- **Changing this skill** -> `references/lessons.md` steps 3-4.

## 5. References
`diagnosis.md`, `eval.md`, `sweep.md`, `insights.md`, `legacy-review.md`, `lessons.md`, `probe.md`, `idea-bag.md`, `breakthrough.md`, `contract.md` (all under `references/`).
