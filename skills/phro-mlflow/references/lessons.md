# Lessons and escalation

How this skill improves itself:
1. Detect: `phro.py audit` (finished but unsynced, GPU idle while scripts wait, missing intro, result or review) and any failure the user or you fix by hand -> add a dated entry to the project's troubleshooting log (the file `domain.md` names; ask once if it names none).
2. Propose: when the same kind of failure happens twice, propose a one- or two-line edit to the Lessons list below (cite the log entries). Changes to the gate, probe order or sweep rules go through the same path.
3. Review every proposed edit to any file of this skill before applying: needed now (cite evidence)? already covered by another rule? smallest wording? conflicts? Fold findings in and note the verdict in the log entry. Idea-bag entries are data, not edits.
4. Apply only after the user approves (rule changes) or asked for the edit. No silent edits to this skill.

Lessons (each seen at least twice; keep only general ones here, project ones go to `.phro/domain.md`)
- Never decide "finished" or "exists" with `pgrep -f`: other shells' command lines contain the same text. Use result files (`DONE`).
- Write remote scripts and patches to a file, then run it; inline heredocs and f-strings break on escapes.
- Before a short smoke run, check that new argument names do not shadow locals in the same function.
- Launching over ssh: send the launch command in the background and do not wait on it; even `setsid nohup ... &` can leave the ssh call hanging. Confirm the job started in a separate `status` call. Stopping the hung local ssh does not kill the remote job.
- Training noise or augmentation must use training-only sources and a seed different from the eval seed.
- Every launch gets its own finish watcher at once (short repeated checks for the `DONE` file so reconnects survive). After a disconnect, re-arm a watcher for EVERY running script. On each finish, fill the freed GPUs before analysing.
