# Legacy review (before every new experiment)

Every run inherits defaults, picks and code from earlier runs. Some still fit the new run's intent; some silently contradict it. Review them with fresh eyes before launch, then verify the effective arguments right after launch.

## When
- Required: first run of a new sweep, a new method, stage, data version or upstream model, a script copied from an older script, any code change. HPO: once per study (trials share the review).
- Skip: reruns that only change the seed of an already reviewed run.

## 1. Independent review (fresh read-only subagent)
Use a subagent with no session context so earlier decisions do not anchor it. Prompt:
```
Read-only review. New experiment intent: <sweep entry: variation, axis, fixed, hypothesis, compare run>.
Launch script: <path>. Code it runs: <paths>. Insight ledger: .phro/ledger/insights.md (stage section).
List every value the run inherits instead of choosing on purpose: argparse defaults, hard-coded paths and constants,
selection or pick files, lexicon or glossary versions, eval sets and the upstream model version they were built with,
data manifests, noise and seed settings, metric functions.
For each: still valid for THIS intent? must change? or a confound vs the compare run? Cite file:line.
Do not edit files; do not read prediction files or MLflow table artifacts.
```
Never put `egress.forbidden_terms` into the prompt. Then decide each item: **keep** (fits the intent), **change before launch**, or **keep as deliberate confound control** (same as the compare run so only the axis differs; say so).

Project-specific inheritance traps live in `.phro/domain.md`. Check them first, and add a new trap there when a review finds one.

## 2. Record
`phro.py manifest set <run> "review=<kept: ...; changed: ...; confound-controlled: ...>"` (one line). It becomes the `review` tag; `audit` flags finished runs without it.

## 3. Post-launch check (first minutes)
After the first log line with the effective arguments: `phro.py diff <new run> --vs <compare run>`. Every parameter difference must be the declared axis or an item the review chose to change. Anything else = stop the run (it is yours to stop; not on a `server.read_only` VM, report to the user instead), fix, relaunch, and add a lesson.
