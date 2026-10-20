# Idea bag (format)

The pack's bag is `.phro/ledger/idea-bag.md`; this file defines its format only. It holds candidate methods from outside (papers, other fields) that could break a limit in the open-limits table. Insights record what the project's runs showed; the bag records what might be tried next. An idea that has been run moves its result to the insight ledger and keeps only a pointer.

## How to use when building an experiment
1. Name the limit first (L-ID, blocking metric key, insight ID from `references/diagnosis.md`).
2. Look up ideas whose `targets` name that L-ID. Prefer `status: queued`, then `new` with fit H.
3. Put the idea ID in the sweep entry (`idea=IDEA-NNN` via `manifest set`) and cite it in the run's `intro`.
4. After the run is judged, update the idea: `status` (tried-win, tried-null, tried-worse, rejected) and `result` (one line with numbers and the insight ID); update the L-ID row.
5. If no idea fits the limit, or two sweeps on it closed with no gain, follow `references/breakthrough.md` before designing more variants of the same axis.

## Entry format
```
### IDEA-001 <short name>
- stage: <stage name from project.json>
- targets: L-001 (the limit row it should move) and the blocking metric key
- idea: what to build, in one or two sentences
- why it may work here: mechanism tied to the project's own evidence
- cost: code size, GPU hours, device or deployment impact
- fit: H / M / L (+ reason: constraints from domain.md, data, risk)
- first experiment: axis, compare run, expected result with a threshold
- evidence: research/<topic>/sources/<file>:<line> (논문|본인|정리)
- evidence: research/<topic>/sources/<file>:<line> (논문|본인|정리)
- status: new | queued | running | tried-win | tried-null | tried-worse | rejected
- result: (after a run) numbers, sweep, insight ID
```
Rules:
- One idea per entry; update instead of duplicating; each entry under about 14 lines.
- One `evidence:` line per claim (mechanism, measured effect, failure condition, cost at our scale). The path is relative to `.phro/`; the source kind is 논문 (paper), 본인 (author's own page or repo), or 정리 (secondary summary, treat as unverified).
- fit L ideas are not added; record them as one line under "Not added" so they are not re-scanned.
- No project data rows (transcripts, labels) in entries.
