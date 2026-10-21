# phro-gcp-mlflow

A generic AutoML loop for one GPU VM with MLflow, packaged as a Claude Code plugin.
The engine is project independent. Everything specific to a project lives in a
domain pack (`.phro/`) inside that project, never in this repo.

What you get:

- `bin/phro.py`: a standard-library CLI that reads MLflow over an ssh tunnel and the VM over `ssh`.
- `skills/phro-onboard`: interviews you and writes the domain pack.
- `skills/phro-mlflow`: runs the loop (judge runs, decide changes on same-seed pairs with a logged verdict, fill idle GPUs, plan sweeps, keep the insight ledger, start a breakthrough search when a limit stays stuck).
- `hooks/guard.py`: PreToolUse guard for Bash, PowerShell, Monitor, WebSearch and WebFetch.
- `foreman/`: an optional Stop hook plus MCP server that keeps the loop going.
- `server/`: files copied to the VM by `setup` (queue, sync, MLflow server).

## Requirements

- Python 3.10+ on the PC. The CLI needs no third-party packages.
- `python` on the PATH. The foreman MCP server starts `python` directly; on Linux or macOS without a `python` command, add one (for example the `python-is-python3` package or a symlink).
- bash for the hooks. On Windows install Git for Windows (Git Bash); without it the hook commands cannot run. The hooks run `python3` where it exists outside Windows and `python` otherwise, so the Windows Store `python3` stub is never started.
- An ssh alias for the GPU VM (key based, no password prompt).
- The `imozuru` skill, used by the breakthrough research step.
- MLflow is installed on the VM by `setup`; nothing is installed on the PC.

## Install

In Claude Code:

```
/plugin marketplace add ThatsHoon/phro-gcp-mlflow
/plugin install phro-gcp-mlflow@phro-gcp-mlflow
```

Or in one step: `/plugin install phro-gcp-mlflow --marketplace ThatsHoon/phro-gcp-mlflow`.
The plugin registers the two skills, the guard and foreman hooks, and the
`phro-foreman` MCP server from `.claude-plugin/plugin.json`, `hooks/hooks.json` and
`.mcp.json`. The manifest has no version field, so every commit is a new version;
`/plugin marketplace update phro-gcp-mlflow` pulls the latest.

## Quick start

In the commands below, `PHRO` stands for `python <plugin dir>/bin/phro.py`, where
`<plugin dir>` is where this repo is installed (inside Claude Code it is
`${CLAUDE_PLUGIN_ROOT}`). Run every command from your project directory; the CLI
finds `.phro/project.json` from the working directory upwards.

1. In your project, run the `phro-onboard` skill. It writes the pack under `.phro/`.
2. `PHRO check-pack` validates the pack.
3. `PHRO setup` copies `server/` to the VM named in `project.json` and installs MLflow there.
4. Use the `phro-mlflow` skill ("how are the runs", "next experiment", "free GPUs").

## Commands

`PHRO <command>`

| Command | Purpose |
|---|---|
| `check-pack` | validate the domain pack |
| `leakcheck [dir] [--no-history]` | scan git-tracked files in dir, and every commit in its history, for forbidden terms and hosts from the pack |
| `setup` | copy `server/` to the VM and install (refused when `read_only`) |
| `status` | per GPU: memory, utilization, running jobs or FREE; queued lines; waiting scripts |
| `list [--exp PREFIX]` | run, status, sweep and gate names |
| `report --run N` / `--since H` | verdict line, CI, compare-tagged runs |
| `compare A B ... [--set S]` / `--sweep S` | same-key side by side, CI, gates; default set is `eval_sets.select` |
| `diff NEW --vs OLD` | actual parameter differences (outside the declared axis = confound) |
| `ci A B ... --metric KEY [--units DIR]` | single runs: 95% bootstrap CI over units and the paired difference vs the first run; unit noise only, never a verdict |
| `ci BASE ARM --metric KEY --pairs` | run families (`<family>_s<N>`): per-seed deltas, mean delta with a t interval over seed and unit variance, gate verdict |
| `decide BASE ARM [--limit L] [--idea I] [--note T] [--reverify SHORT \| --loo CHANGE]` | the accept decision over every metric of the stage on the select set: accept, accept-short, reject or undecided, appended to `.phro/ledger/decisions.jsonl` |
| `loo --stage S --data V` | accepted changes not yet tested by leave-one-out, and the arm to build for each |
| `splitcheck RUN` | unit ids shared between the select, heldout and final sets (must be 0) |
| `baseline --data V` | per-stage baseline table for data version V |
| `cascade --stage S --data V` | finished downstream runs to retrain when stage S changes |
| `audit` | missing tags, unsynced runs, idle GPU while `run_*.sh` scripts wait (seen with `pgrep` on the VM), look budget (run families per data version), leave-one-out backlog, short-budget passes not re-verified, frozen list |
| `manifest get NAME` / `manifest set NAME key=value ...` | read or write the run manifest on the VM (`set` refused when `read_only`) |
| `queue add GPU SCRIPT` / `queue list` | append a job to the VM GPU queue (refused when `read_only`) or show it |
| `selftest` | built-in self check |

## Pack layout

```
.phro/
  project.json   stages, metrics, eval sets, server, egress terms (read by scripts)
  domain.md      goal chain, hardware limits, data rules, terms (read by the agent)
  plan.md        the next jobs you approved; only these start without asking
  ledger/        insights, open limits, idea bag, decisions.jsonl (written by decide)
  foreman/       optional; ENABLED turns the foreman on
```

Start from `skills/phro-onboard/templates/project.example.json`.

## Run folder contract

Every run on the VM follows one folder and tag contract so that sync, reports and
audits work for any project. See
[skills/phro-mlflow/references/contract.md](skills/phro-mlflow/references/contract.md).

## Safety and isolation

- The engine never calls `gcloud`. Remote work is only `ssh <server.host>` and `scp` to it.
- `hooks/guard.py` is a seatbelt against mistakes, not a sandbox. It blocks any `gcloud` command that is not read-only (allowed: `list`, `describe`, `info`, `version`, `get-*`, `list-*`, and `gcloud compute ssh|scp` to an allowed host); `ssh`, `scp` and `rsync` to hosts outside `server.allow_hosts`, including jump hosts, proxy commands, `sh -c` strings and the remote command of an allowed `ssh`; and searches or URLs containing the pack's `egress.forbidden_terms`, including terms in shell commands that reach the network (`curl`, `wget`, `gh`, `Invoke-WebRequest`, the imozuru fetch scripts; loopback URLs such as the MLflow tunnel are not egress) and in Monitor WebSocket URLs. PowerShell and Monitor commands get the same checks; for PowerShell this is best effort (`powershell -Command`, script blocks, `Start-Process`, `iex`), and encoded PowerShell (`-EncodedCommand`, `-e`) is blocked outright. Without a pack only the gcloud rules apply. A pack that fails `check-pack` still supplies its host and term lists; one that cannot be parsed blocks every `ssh`.
- On the VM, the queue starts jobs only on GPUs with no compute process (any user, any project) and never stops processes. Other sessions and projects are never touched.
- `setup` installs only under the pack's `server.root`, uses no sudo, starts MLflow with `nohup` and restarts it from an `@reboot` cron line, tags its cron lines `# phro:<project>` and leaves other lines alone, binds MLflow to `127.0.0.1`, and aborts if the port is taken by something other than its own running MLflow. Re-running `setup` keeps a running MLflow and refreshes the cron lines.
- `server.read_only: true` makes the VM read-only: no launches, queue entries, process stops, file writes or `manifest set`. Findings go to you and to the local `.phro/ledger/`.
- The foreman hook works only when `.phro/foreman/ENABLED` exists, and blocks only the session whose id the file holds. Create it empty (or with `claim`): the next session that stops claims it. A session that has not stopped for 2 hours loses it to the next one. Delete the file to turn it off.
- Scripts print counts and metrics, not data rows.

## Domain packs are never committed

`.phro/` and `domains/` are gitignored here. Packs hold project goals, hosts and
data rules, so add `.phro/` to your project's `.gitignore` too if the project may
become public. Run `PHRO leakcheck <repo>` from a project that has a pack before
any public push; it must print `leakcheck clean`. It also scans git history, so a
term that was committed once and removed later still fails until history is rewritten.

## License

MIT. See [LICENSE](LICENSE).
