---
name: phro-onboard
description: Turns a project description into a phro domain pack (.phro/) so the phro-mlflow loop can run on one GPU VM with MLflow. Mode 1 takes a rough workflow and finds a model combination with source-backed research (imozuru) and a small demonstration; Mode 2 takes fixed models and structure and proposes per-stage probes, sweep axes, gates and a baseline table. Use it when the user starts a new project or asks to design a workflow, choose a model combination, or build a domain pack, including Korean phrasings like "새 프로젝트", "워크플로우 설계", "모델 조합", "도메인 팩", and English phrasings like "onboard", "set up AutoML for", "design the pipeline for".
---

# phro-onboard

Output: `<project>/.phro/` (project.json, domain.md, plan.md, ledger/, research/). Operating the loop afterwards is the phro-mlflow skill. Talk to the user in their language; write files in plain English except where a format below names Korean labels. Commands: `python ${CLAUDE_PLUGIN_ROOT}/bin/phro.py <cmd>` from the project dir.

## Hard rules
- GPU probes and `phro.py setup` need the user's OK each time. Run `setup` only on a VM the user names, never one you guessed or found.
- Web searches, WebFetch, imozuru questions and subagent prompts carry a generic problem statement only. Never put project data, project names, or any `egress.forbidden_terms` entry in them. Ask the user for the forbidden terms before the first search.
- `domain.md` holds prose; `project.json` holds only what a script reads (`references/contract.md` in phro-mlflow lists the keys). A value no script reads does not go in `project.json`.
- Do not invent numbers: targets are the user's stated values with their source in `goal.kpis[].source`; baselines stay empty until `phro.py baseline` fills them.
- No emoji, no em or en dashes, no AI-attribution lines in any file written.

## 1. Choose the mode
Ask once, then pick. Only a rough workflow, or models not decided -> Mode 1 (it ends by handing a fixed structure to Mode 2). Models and structure already fixed -> Mode 2. If `.phro/` already exists, read it and edit; never overwrite.

## 2. Mode 1: workflow -> model combination
1. Split the workflow into stages. For each: input, output, budget (model size, latency, data available). Show a table and confirm with the user.
2. One imozuru dig per stage, topic dir `.phro/research/<stage>-models`. State the question generically ("small models that do <task> within <size>"). Required claims, fixed: the model families that do this job at this size, the measured numbers, the failure conditions, the original source. Stop by imozuru's rules. imozuru is a separate skill; if it is not installed, tell the user and stop.
3. Offer 2 or 3 compositions (cascade, end to end, mixed) as a table: `option | stages | expected strength | expected weakness | evidence`, each evidence cell a path `sources/<file>:<line>`. No claim without a path.
4. Demonstration: propose which candidates run zero-shot or with a small fine-tune on a slice of the select set, to measure a per-stage ceiling. Propose first; run only after the user's OK (it uses a GPU).
5. When the user fixes one composition, continue with Mode 2 on it. Keep `research/` as the evidence record.

## 3. Mode 2: fixed structure -> proposals
For every stage produce, as files (section 4) and not as chat:
- Probe plan in the order of phro-mlflow `references/probe.md`: data check, ceiling, information, loss split (needs an upstream stage), smoke. Name the read-only script for each; scripts print counts only.
- Method-screening axes: one structural choice per axis, listed in `plan.md`. A numeric HPO space only after a method wins; mark it "after screening".
- Metric and gate: key `<set>/<name>`, direction, target, `resample_unit`, `agg` (`mean` or `f1`), `gate.min_delta`. Every metric must be computable from per-unit values; every gate needs a CI rule (paired seeds, `stats`, rules in phro-mlflow `references/sweep.md`). Declare each gated metric twice: on the select set (`<select>/<name>`, which `phro.py decide` uses to accept or reject a change) and on the heldout set (finalists and the KPI). Set `stats.seeds_min` (seed pairs per decision) and `stats.min_units` (fewest eval units a decision may rest on).
- Baseline table (data version x stage x variation x seed, chain cells included) in `domain.md`; numbers come later from `phro.py baseline`.
- Eval discipline: `eval_sets` select, heldout, final and a `look_budget`. Ask which data backs each set; one set cannot be both select and final.

## 4. Write the pack
1. `project.json` from `${CLAUDE_PLUGIN_ROOT}/skills/phro-onboard/templates/project.example.json`. Replace every placeholder; delete keys you do not need (defaults are in phro-mlflow `references/contract.md`). Required: name, goal, stages, metrics, eval_sets, server, egress; `server.host` must be in `server.allow_hosts`. Ask the user for host, root, MLflow port, GPU count; never guess them. Ask explicitly whether the VM is new, or an existing or shared VM that already runs other work; an existing or shared VM, or one the user wants only read, gets `server.read_only: true`. Ask whether runs already exist in MLflow under other tag names or with result-like params; map them with `tag_alias` and `derived_params`. Ask the user for `name` and `language` (report language). Create `.phro/` in the project root. Each stage's `experiment` is its MLflow experiment name. Each metric's `stage` must name a stage; each `goal.kpis[].key` should equal a `metrics[].key`. `look_budget` has one entry per non-select set. `agg.field` or `tp/fp/fn` name the fields the training code will write to `units.jsonl`; record them in `domain.md` Glossary.
2. `domain.md` sections: Goal chain (table `stage | metric key | target | baseline (read via phro.py baseline) | ceiling | hands to next stage`), Constraints (hardware, model size, latency; used to score idea fit), Data and privacy rules (what is never printed or searched, mirroring `egress`), Glossary, Report style (one example report), Inheritance traps (empty at first), Troubleshooting log path.
3. `plan.md`: the user's approval queue, top first, one line per item: `- [ ] <stage>: <probe|sweep>: <what>: <GPU estimate>`. Items you propose carry `(proposed)` until the user approves; only approved items run without asking.
4. `ledger/insights.md`: the open-limits table header, then an empty heading per stage. `ledger/idea-bag.md`: a one-line pointer to the format, then `## Not added`. `ledger/lessons.md`: a title only. Headers match phro-mlflow `references/insights.md`, `references/idea-bag.md`, `references/lessons.md`. Table header:
   `| L-ID | stage | blocking metric (value) | gap to target | ruled out (sweep) | bound evidence | IDEA | status |`
5. If the project is a git repo that may become public, add `.phro/` to `.gitignore`.

## 5. Validate
Run `python ${CLAUDE_PLUGIN_ROOT}/bin/phro.py check-pack` from the project dir. It must print `pack ok: <name>`; fix each listed error and rerun. Then check by hand what it cannot: every stage has at least one metric, every metric key's set is in `eval_sets`, every `depends_on` edge matches the real data flow.

## 6. Independent review
Spawn a fresh read-only subagent (no session context) with this prompt and fold in its findings:
```
Read-only review of a project pack. Goal as stated by the user: <goal in generic words>.
Read .phro/project.json, domain.md, plan.md, ledger/*.md. Do not edit anything. No web search or fetch.
List: (1) stages the goal needs that the pack lacks, or that are in the wrong dependency order;
(2) metrics that cannot be measured from per-unit values, or whose target has no source;
(3) gates with no CI rule (no seeds, no paired comparison, no min_delta rationale);
(4) egress gaps: private names, hosts or data terms that appear in domain.md or plan.md but not in
egress.forbidden_terms, and any step that would send project data to a search;
(5) eval-set leakage risks (select, heldout, final overlap; look_budget missing);
(6) plan.md items that need a GPU and are not marked proposed.
Cite file and line for each item.
```
Never put `egress.forbidden_terms` into this prompt.

## 7. Approve and install
Show the user the pack summary (stages, metrics, eval sets, first plan items, review findings and your response). Wait for approval. Only if the user names a VM to install on, run `phro.py setup` (it refuses when `server.read_only` is true), then report the result and the MLflow tunnel command. Otherwise stop; the pack alone is valid. Then hand over to phro-mlflow.
