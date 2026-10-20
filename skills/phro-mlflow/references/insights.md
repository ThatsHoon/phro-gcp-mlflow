# Insight ledger (why metrics moved -> next design)

Ledger: `.phro/ledger/insights.md` (language per `project.json language`), entries `I-NNN` grouped by stage, with the open-limits table on top. Process failures go to `lessons.md`, not here.

## When
- Every finished run or completed sweep, right after `report` or `compare`. Check by hand that every finished run has an `insight` tag (`audit` does not check it).
- Before designing any sweep: read the ledger section for that stage and cite the insight IDs the design relies on in the sweep's `hypothesis`.

## How to analyse a result
Follow `references/diagnosis.md`. Also check: CI clear of 0 (`ci`), per-tier and length-bucket metrics when the pack has them, and confounds vs the compare run (`diff`).
Status: 확정 (CI clear, or same direction twice), 시사 (direction only or confounded), 열림 (separating run or diagnostic pending), 반박 (a later run contradicts; link it).

## Entry format
`### I-NNN <one-line claim> (<status>, <runs/sweep>)` then bullets: 관찰 (numbers + CI), 원인 (mechanism), 근거·확신 / 교란, 설계 (do / avoid / next axis), 반증 (what result would overturn it).
Update existing entries instead of adding near-duplicates; flip status when new runs land. Attach `insight=I-NNN` (comma list ok) to each analysed run: `phro.py manifest set <run> insight=I-NNN`.

## Open limits table (top of the ledger)
One row per limit that blocks a goal-chain cell. The agent edits it after every diagnosis.

| L-ID | stage | blocking metric (value) | gap to target | ruled out (sweep) | bound evidence | IDEA | status |
|---|---|---|---|---|---|---|---|
| L-001 | `<stage>` | `<metric key>` (`<value>`) | `<gap>` | `<sweep>`: `<what it eliminated>` | `<probe or run>` | IDEA-001 | open |

Status: `open`, `broken` (broken through; link the run), `bound` (confirmed as an input or information ceiling). A limit with two sweeps closed without gain and no IDEA is the trigger for `references/breakthrough.md`. Idea entries name the L-ID they target.

## Escalation
An insight confirmed twice that changes how experiments should be run is proposed as a rule in `probe.md`, `sweep.md` or `eval.md`; apply after the user approves (same path as lessons).
