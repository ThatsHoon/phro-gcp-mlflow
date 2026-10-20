# phro-gcp-mlflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the generic AutoML plugin (engine + domain pack contract) and prove it with a read-only acceptance test against the first private pack.

**Architecture:** `bin/phro.py` runs on the PC, reads the domain pack `<project>/.phro/project.json`, talks to MLflow over REST through an ssh tunnel and to the GPU VM only through `ssh <server.host>`. The foreman and the guard hook use only the stdlib and `bin/phro.py`. The VM side is two files copied to `~/phro/<project>/`: `phro_server.py` (stdlib) and `phro_sync.py` (mlflow client in the VM venv).

**Tech Stack:** Python 3.10+ stdlib (PC side, foreman, hooks, `phro_server.py`), mlflow 2.17.2 (VM venv only), Claude Code plugin (skills, hooks.json, .mcp.json), imozuru skill (already installed, not vendored).

**Spec:** `docs/superpowers/specs/2026-10-19-phro-gcp-mlflow-design.md`

## Global Constraints

- PC-side code (`bin/`, `foreman/`, `hooks/`) and `server/phro_server.py`: Python >= 3.10 stdlib only. Only `server/phro_sync.py` may import `mlflow`.
- No `gcloud` call anywhere in the code. Remote work only through `ssh <pack.server.host>` / `scp` to it.
- Every script entry point calls `sys.stdout.reconfigure(encoding="utf-8")` (a cp949 Windows console broke Korean output in this environment).
- Tests: each script has `selftest` (stdlib `assert`), run as `python -X utf8 <file> selftest`, printing `<name> selftest ok`. No pytest.
- Output prints numbers, names and tags only, never data rows (transcripts, predictions, labels).
- Commits: author `thatshoon <gagea45@gmail.com>`; no `Co-Authored-By` or "Generated with" lines, no emoji, no em/en dashes in any file (user rule, overrides harness attribution).
- Public repo: never commit `domains/` or any `.phro/`; run `phro.py leakcheck` before any push; push only after the user says so.
- Nothing private enters this repo: no host names, IPs, paths, metric keys or data names of the first pack, in code, tests or docs. Those values live only in the private appendix `<first pack>/.phro/accept/acceptance-plan.md`.
- A pack with `server.read_only: true` (the first private pack) is read-only: `setup` and `manifest set` refuse, and no task writes, installs, edits cron or queue, or controls processes on that VM. Other GCP instances are never touched.
- Pack defaults (spec 3.1): `stats.seeds_min` 4, `stats.bootstrap` 1000, `server.manifest` `<root>/runs.json`, `server.queue` `<root>/gpu_queue.txt`, `server.done_globs` `["runs/**/DONE"]`, `server.run_name_rules` `[["(?:.*/)?([^/]+)/DONE$", "\\1"]]`, `server.read_only` `false`, `derived_params` `[]`, `tag_alias` `{}`.
- Bootstrap must reproduce the legacy algorithm exactly: `ids = sorted(intersection)`, `rng = random.Random(seed)`, per resample `smp = [rng.choice(ids) for _ in ids]` shared across runs, quantiles `(sorted(v)[int(0.025*B)], sorted(v)[int(0.975*B) - 1])`.

## Review Focus

1. Korean text printed to a Windows console must not raise `UnicodeEncodeError` (test: Task 1 runs `check-pack` on a pack with a Korean `name` through a cp949-encoded pipe).
2. Missing or malformed `project.json` must print the `check_pack` error list and exit 1, never a traceback (test: Task 1).
3. MLflow unreachable (tunnel down) must print `ssh -N -L <port>:127.0.0.1:<port> <host>` and exit 1 (test: Task 3).
4. The guard must never crash on odd Bash input (heredocs, unbalanced quotes, Windows paths); a parse failure falls back to whitespace split, because an uncaught crash would let the call through (test: Task 6).
5. `phro_server.py queue` must treat a GPU as busy if ANY compute process uses it, including other projects' jobs (test: Task 7).

---

## File Structure

```
.claude-plugin/plugin.json          (exists)
.mcp.json                            Task 8
README.md                            Task 12
bin/phro.py                          Tasks 1-5, 7 (one file: pack, stats, REST, commands)
hooks/guard.py                       Task 6
hooks/hooks.json                     Tasks 6, 8
foreman/foreman.py                   Task 8
server/phro_server.py                Task 7
server/phro_sync.py                  Task 7
skills/phro-mlflow/SKILL.md          Task 9
skills/phro-mlflow/references/*.md   Task 9
skills/phro-onboard/SKILL.md         Task 10
skills/phro-onboard/templates/project.example.json   Task 1
<first pack>/.phro/                  Task 11 (outside the repo, never committed)
```

Run commands from the repo root. `domains/<first pack>/legacy/` (gitignored) holds the legacy skill and scripts to port from.

---

### Task 1: Pack loading, checking, leakcheck

**Files:**
- Create: `bin/phro.py`
- Create: `skills/phro-onboard/templates/project.example.json` (the spec 3.1 example with `"name": "example"`, `egress.forbidden_terms` `["secret-host"]`)

**Interfaces:**
- Produces:
  - `find_pack(start: Path | None = None) -> Path | None`: first `<dir>/.phro/project.json` walking up from `start` (default `$CLAUDE_PROJECT_DIR`, else cwd).
  - `load_pack_dict(d: dict) -> dict`: deep copy of `d` with all defaults filled in.
  - `load_pack(path: Path) -> dict`: `load_pack_dict(json)` plus `pack["_dir"] = path.parent` (the `.phro` dir).
  - `check_pack(p: dict) -> list[str]`: error strings, empty when valid.
  - `tag(run: dict, name: str, p: dict) -> str`: `run["tags"].get(p["tag_alias"].get(name, name), "")`.
  - `leak_hits(p: dict, files: dict[str, str]) -> list[tuple[str, int, str]]`: `(file, line_no, term)` for every match of `egress.forbidden_terms`, `server.host` and `server.allow_hosts`, case-insensitive, term reported lowercased.
  - CLI `python bin/phro.py <cmd> [...]`; `main(argv) -> int`; each command is `cmd_<name>(p, a) -> int`.

- [ ] **Step 1: Write the failing selftest**

```python
def selftest():
    tpl = ROOT / "skills/phro-onboard/templates/project.example.json"
    EXRAW = json.loads(tpl.read_text(encoding="utf-8"))
    EX = load_pack(tpl)
    assert check_pack(EX) == [] and EX["stats"]["bootstrap"] == 1000 and EX["server"]["queue"].endswith("gpu_queue.txt")
    bad = copy.deepcopy(EXRAW)
    bad["stages"][1]["depends_on"] = ["nope"]; bad["metrics"][0]["agg"] = {"type": "median"}; del bad["server"]["host"]
    errs = check_pack(load_pack_dict(bad))
    assert any("depends_on" in e and "nope" in e for e in errs) and any("agg" in e for e in errs) and any("server.host" in e for e in errs)
    assert tag({"tags": {"model_version": "v4"}}, "data_version", {"tag_alias": {"data_version": "model_version"}}) == "v4"
    assert leak_hits(EX, {"a.md": "ok\nuse my-gpu here\nSecret-Host"}) == [("a.md", 2, "my-gpu"), ("a.md", 3, "secret-host")]
    print("phro selftest ok")
```

- [ ] **Step 2: Run, expect failure**

Run: `python -X utf8 bin/phro.py selftest`
Expected: FAIL (`NameError` or missing template).

- [ ] **Step 3: Implement** `find_pack`, `load_pack_dict`, `load_pack`, `check_pack`, `tag`, `leak_hits`, `main`, `cmd_check_pack`, `cmd_leakcheck`.
  - `check_pack` checks: required keys `name, goal, stages, metrics, eval_sets, server, egress`; unique stage names; every `depends_on` names a stage; no DAG cycle; every metric `stage` exists; `agg.type` in `{"mean", "f1"}` with its fields (`field` / `tp, fp, fn`); `dir` in `{"max", "min"}`; `server.host` present and in `server.allow_hosts`.
  - `cmd_check_pack`: no pack -> print `no .phro/project.json found from <start>`, return 1; errors -> one per line, return 1; else print `pack ok: <name>`.
  - `cmd_leakcheck <dir>`: files from `git -C <dir> ls-files`, read as UTF-8 (skip undecodable); print `file:line: term`; return 1 on any hit, else print `leakcheck clean`.

- [ ] **Step 4: Run, expect pass; then Review Focus 1 and 2**

Run: `python -X utf8 bin/phro.py selftest` -> `phro selftest ok`
Run (no pack): `cd "$TMP" && python <repo>/bin/phro.py check-pack; echo rc=$?` -> `no .phro/project.json found ...`, `rc=1`, no traceback.
Run (cp949): in a temp dir with `.phro/project.json` = the example with `"name": "한글팩"`: `PYTHONIOENCODING=cp949 python <repo>/bin/phro.py check-pack | cat` -> `pack ok: 한글팩`, no `UnicodeEncodeError`.

- [ ] **Step 5: Commit** `git add bin/phro.py skills/phro-onboard/templates && git commit -m "Add pack loader, checker and leakcheck"`

---

### Task 2: Stats and DAG (pure functions)

**Files:** Modify `bin/phro.py`

**Interfaces:**
- Consumes: `load_pack_dict` (Task 1).
- Produces:
  - `family(name: str) -> str`: strip a trailing `_s<digits>`.
  - `agg_value(rows: list[dict], agg: dict) -> float`: `mean` -> mean of `row[field]`; `f1` -> `2*TP / (2*TP + FP + FN)` over summed counts, `0.0` when the denominator is 0.
  - `boot(units: list[dict[str, dict]], agg: dict, B: int = 1000, seed: int = 0) -> dict` with keys `n`, `value` (list), `ci` (list of `(lo, hi)`), `d_ci` (list of `(lo, hi)`, delta vs `units[0]`). Algorithm per Global Constraints. Empty intersection -> `n` 0 and empty lists.
  - `downstream(p: dict, stage: str) -> list[str]`: stages depending on `stage` directly or transitively, topological order, excluding `stage`.

- [ ] **Step 1: Get the legacy parity numbers.** Add this module constant to `bin/phro.py`:
```python
PARITY_CALLS = [{f"c{i}": (i % 3, i % 2, (i + 1) % 2) for i in range(40)},
                {f"c{i}": ((i + 1) % 3, 0, i % 2) for i in range(5, 45)}]
```
In a scratch script outside the repo, import the legacy sync module from `domains/<first pack>/legacy/server/` (its imports are stdlib only), run its `boot_f1(PARITY_CALLS)` and print `n`, `f1`, `ci`, `d_ci` rounded to 10 places. Paste them into Step 2 as `N_LEGACY`, `F1_LEGACY`, `CI_LEGACY`, `DCI_LEGACY`.

- [ ] **Step 2: Add failing asserts to `selftest`**

```python
    assert family("alpha_m1_s12") == "alpha_m1" and family("beta") == "beta"
    U = [{k: {"tp": a, "fp": b, "fn": c} for k, (a, b, c) in d.items()} for d in PARITY_CALLS]
    b = boot(U, {"type": "f1", "tp": "tp", "fp": "fp", "fn": "fn"})
    r10 = lambda xs: [tuple(round(x, 10) for x in c) for c in xs]
    assert b["n"] == N_LEGACY and [round(x, 10) for x in b["value"]] == F1_LEGACY
    assert r10(b["ci"]) == CI_LEGACY and r10(b["d_ci"]) == DCI_LEGACY
    assert boot([{"a": {"x": 1}}, {"b": {"x": 1}}], {"type": "mean", "field": "x"})["n"] == 0
    assert agg_value([{"x": 1}, {"x": 3}], {"type": "mean", "field": "x"}) == 2.0
    dag = load_pack_dict({**EXRAW, "stages": [{"name": "a", "experiment": "1", "depends_on": []},
                                              {"name": "b", "experiment": "2", "depends_on": ["a"]},
                                              {"name": "c", "experiment": "3", "depends_on": ["b"]}], "metrics": []})
    assert downstream(dag, "a") == ["b", "c"] and downstream(dag, "c") == []
```

- [ ] **Step 3: Run, expect failure** `python -X utf8 bin/phro.py selftest` -> `NameError`.
- [ ] **Step 4: Implement** the four functions.
- [ ] **Step 5: Run, expect pass** -> `phro selftest ok`.
- [ ] **Step 6: Commit** `git commit -am "Add bootstrap, families and stage DAG"`

---

### Task 3: MLflow REST reader, list / report / compare / diff

**Files:** Modify `bin/phro.py`

**Interfaces:**
- Consumes: `load_pack`, `tag`, `family` (Tasks 1-2).
- Produces:
  - `rest(p: dict, path: str, body: dict | None = None) -> dict`: POST JSON when `body` is given, else GET, to `http://127.0.0.1:<server.mlflow_port>` through the module-level opener `_open` (default `urllib.request.urlopen`; `_set_open(fn | None)` swaps it, `None` restores the default). On `URLError` raise `Unreachable` whose `str()` is `MLflow not reachable. Start the tunnel: ssh -N -L <port>:127.0.0.1:<port> <host>`. `main` catches it, prints it, returns 1.
  - `fetch_runs(p: dict) -> list[dict]`: all active runs of all experiments, paginated. `POST /api/2.0/mlflow/experiments/search {"max_results": 1000}`, `POST /api/2.0/mlflow/runs/search {"experiment_ids": [...], "max_results": 1000, "page_token": ...}`. Normalized run: `{"exp", "id", "name", "start", "end", "params": {}, "metrics": {}, "tags": {}}` (`metrics` = latest values).
  - `history(p, run_id: str, key: str) -> list[tuple[int, float]]`: `GET /api/2.0/mlflow/metrics/get-history?run_id=&metric_key=`.
  - `names_in(text: str, known: list[str]) -> list[str]`: port of the legacy `names_in`: longest name first, ASCII word boundaries (a Korean particle may follow), result in `known` order.
  - `param_diff(a: dict, b: dict, derived: set) -> list[tuple]`: `[(k, a.get(k, "-"), b.get(k, "-")) for k in sorted(set(a) | set(b)) if k not in derived and a.get(k) != b.get(k)]`.
  - `compare_table(runs: list[dict], set_: str, p: dict) -> tuple[list[str], list[list[str]]]`: header `run, experiment, gate_<set>, <keys...>`; keys = pack metrics starting with `<set>/` first, then other `<set>/` metrics without `/ci_`, at most 12; cell `0.1234`, plus ` [lo, hi]` when `<set>/ci_lo_<name>` and `ci_hi_` exist; missing -> `-`.
  - `table(head, rows) -> None`: markdown table print.
  - `cmd_list [--exp]`; `cmd_report --run N | --since H` (header with status, sweep, insight, gate tags; `intro`; `result`; then `compare_table` against runs named in its `compare` tag, once per `gate_<set>` tag); `cmd_compare A B ... --set S | --sweep S` (sweep prints `axis`, `fixed`, `pick` of its first run and each `hypothesis`); `cmd_diff NEW --vs OLD` (MLflow params; `run not found: <name>` if absent).

- [ ] **Step 1: Failing asserts.** In `selftest`, set `_open` to a fixture opener serving two experiments and three runs `r1_s0`, `r1_s1`, `r2` (one has tag `gate_heldout`, metrics `heldout/d_score`, `heldout/ci_lo_d_score`, `heldout/ci_hi_d_score`, and a `compare` tag naming `r2`); restore `_open` after:
```python
    runs = fetch_runs(EX)
    assert {r["name"] for r in runs} == {"r1_s0", "r1_s1", "r2"}
    head, rows = compare_table(runs, "heldout", EX)
    assert head[:3] == ["run", "experiment", "gate_heldout"] and any("[" in c for r in rows for c in r)
    assert names_in("run_q2 대비", ["run_q", "run_q2"]) == ["run_q2"] and names_in("nb1과 같음", ["nb1"]) == ["nb1"]
    assert param_diff({"lr": "1", "train": "9"}, {"lr": "2", "train": "8"}, {"train"}) == [("lr", "1", "2")]
    def down(*_a, **_k): raise urllib.error.URLError("x")
    _set_open(down)
    try: fetch_runs(EX); assert False
    except Unreachable as e: assert "ssh -N -L 5002:127.0.0.1:5002 my-gpu" in str(e)
    _set_open(None)   # back to urlopen
```
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** the functions above.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `git commit -am "Add MLflow REST reader and list/report/compare/diff"`

---

### Task 4: ci / baseline / cascade / audit

**Files:** Modify `bin/phro.py`

**Interfaces:**
- Consumes: `fetch_runs`, `rest`, `boot`, `family`, `downstream`, `tag`, `table` (Tasks 1-3); `snapshot` (Task 5) and `remote_manifest` (Task 7) are called by `cmd_audit` only, so stub them as `None`-returning until those tasks land.
- Produces:
  - `load_units(p, run: dict, set_: str, units_dir: Path | None) -> dict[str, dict] | None`: from `<units_dir>/<run name>.jsonl` if given, else the MLflow artifact `GET /get-artifact?path=units.jsonl&run_uuid=<id>`; keep rows with `row["set"] == set_`; key `row["unit"]`; `None` if absent.
  - `cmd_ci`: `phro.py ci RUN... --metric KEY [--units DIR]`. Metric from the pack; set = `KEY` before `/`. A run without units -> print `<run>: no units.jsonl`, return 1. Output: line `KEY, <n> units, <B> paired resamples; delta = run - <first>`, then table `run | KEY | 95% CI | delta vs <first> | delta 95% CI` with `f"{v:.4f}"`, `f"[{lo:.4f}, {hi:.4f}]"`, `f"{d:+.4f}"`, `f"[{dl:+.4f}, {dh:+.4f}]"` (the legacy format).
  - `baseline_tables(p, runs, version: str) -> list[tuple[str, list[str], list[list]]]`: `(title, head, rows)` per (stage, metric) of the pack; runs with `tag(r, "data_version") == version`, `tag(r, "status") == "done"`, `exp` starting with the stage `experiment`, metric present; rows per `family` sorted by name: `family, seeds, "mean [min, max]"` (4 decimals), gap = `target - mean` for `max`, `mean - target` for `min`, `f"{gap:+.4f}"`.
  - `cascade_runs(p, runs, stage: str, version: str) -> list[dict]`: done runs in experiments of `downstream(p, stage)` whose `data_version` differs from `version`.
  - `audit_issues(p, runs, manifest: dict | None, snap: dict | None) -> list[str]`: (a) done runs missing `intro`, `result` or `review` in the manifest entry or tags; (b) `intro` without `기대:` or `Expected:`; (c) snapshot finished run names (via `run_name`) with no MLflow run -> `finished, not synced: <name>`; (d) GPU idle (`util < 5 and mem < 500`) while `snap["waiting"]` is non-empty -> `GPU idle ...`; (e) per heldout set, number of runs with `<set>/` metrics over `look_budget`; (f) per `data_version`, 3 or more `accepted=yes` runs and no run whose `sweep` starts with `loo_` -> `... leave-one-out ...`; (g) frozen runs with `reopen_if`. `None` inputs skip their checks with one line saying so.
  - `cmd_baseline --data V`, `cmd_cascade --stage S --data V`, `cmd_audit`.

- [ ] **Step 1: Failing asserts** on the Task 3 fixture with added tags `status=done`, `data_version=v1`; `dag_pack` = the Task 2 `dag`; `dag_runs` = the fixture runs with `r2` moved to experiment `2/x` (done, `data_version` `v1`); `three_accepted` = 3 fixture runs, `data_version` `v1`, `accepted=yes`, no `loo_` sweep:
```python
    t = baseline_tables(EX, runs, "v1")
    fam = {row[0]: row for _, _, rows in t for row in rows}
    assert fam["r1"][1] == 2                                           # r1_s0 + r1_s1
    assert [r["name"] for r in cascade_runs(dag_pack, dag_runs, "a", "v2")] == ["r2"]
    iss = audit_issues(EX, runs, {"r1_s0": {"intro": "x"}}, {"gpus": [[0, 0, 0]], "waiting": ["run_x.sh"], "done": {"runs/s/zz/DONE": 1}})
    assert any("기대" in i or "Expected" in i for i in iss) and any("not synced" in i and "zz" in i for i in iss) and any("idle" in i for i in iss)
    assert any("leave-one-out" in i for i in audit_issues(EX, three_accepted, {}, None))
    assert load_units(EX, {"name": "nope", "id": "x"}, "heldout", Path(tempfile.mkdtemp())) is None
```
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `git commit -am "Add ci, baseline, cascade and audit"`

---

### Task 5: ssh helper, snapshot, status

**Files:** Modify `bin/phro.py`

**Interfaces:**
- Consumes: `load_pack` (Task 1).
- Produces:
  - `ssh(p, remote_cmd: str, stdin: str | None = None, timeout: int = 40) -> subprocess.CompletedProcess`: argv `["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", p["server"]["host"], remote_cmd]`, `encoding="utf-8"`. The only place that starts ssh (scp in Task 7 also takes the host from the pack).
  - `REMOTE`: Python source sent to `python3 -` after a first line `CFG = <json>` built by `remote_config(p) -> str` (`root`, `done_globs`, `manifest`, `queue`). It prints one JSON object: `gpus` `[[idx, util, mem_mib], ...]`, `procs` `{idx: [--out folder name or last 30 chars of args, ...]}`, `done` `{path relative to root: mtime}` (`glob(..., recursive=True)`), `docs` `{run: {"intro", "result"}}` cut to 900 chars from the manifest, `waiting` (`run_*.sh` names from `pgrep -af`), `queue` (non-comment lines). Port of the legacy foreman `REMOTE` and the legacy query script's `gpu_map`.
  - `snapshot(p) -> dict | None`: runs `REMOTE`; `None` on any ssh or JSON failure.
  - `run_name(path: str, rules: list[list[str]]) -> str | None`: first rule whose regex matches; `None` when its replacement is `""` (script end marker) or nothing matches.
  - `cmd_status`: table `GPU | mem | util | running` (`FREE` when no process), queue lines, waiting scripts.

- [ ] **Step 1: Failing asserts**
```python
    R = [["runs/([^/]+)/eval_(.+)_final\\.json$", "\\1_\\2"], [".*/([^/]+)/final$", "\\1"], [".*_DONE$", ""]]
    assert run_name("runs/alpha/eval_m1_s2_final.json", R) == "alpha_m1_s2"
    assert run_name("runs/stage2/m/beta/final", R) == "beta" and run_name("runs/alpha/ALPHA_DONE", R) is None
    assert run_name("runs/stage/myrun/DONE", EX["server"]["run_name_rules"]) == "myrun"
    compile(remote_config(EX) + "\n" + REMOTE, "remote", "exec")
```
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `git commit -am "Add ssh snapshot and status"`

---

### Task 6: Guard hook

**Files:**
- Create: `hooks/guard.py`
- Create: `hooks/hooks.json` with one `PreToolUse` entry: matcher `Bash|WebSearch|WebFetch`, command `python -X utf8 "${CLAUDE_PLUGIN_ROOT}/hooks/guard.py"`, timeout 10.

**Interfaces:**
- Consumes: `find_pack`, `load_pack` (import via `sys.path.insert(0, <plugin root>/bin)`).
- Produces: `verdict(tool: str, tool_input: dict, p: dict | None) -> str | None` (block reason or `None`). `main()` reads the hook JSON from stdin; reason -> stderr, exit 2; else exit 0. An exception inside `verdict` prints `guard error: <e>` to stderr and exits 2 when the Bash command contains `gcloud`, `ssh`, `scp` or `rsync`, else exits 0.

Rules (spec 8.2):
- Split a Bash command into segments on `;`, `&&`, `||`, `|` and newlines; tokenize each with `shlex.split` (posix), falling back to `str.split` on `ValueError`.
- `gcloud compute instances <verb>` with verb in `stop, delete, reset, start, create, suspend, resume, set-machine-type, add-metadata, remove-metadata` -> block. `gcloud compute firewall-rules create|update|delete` -> block. `gcloud compute ssh|scp <name>` -> block when there is no pack or `<name>` is not in `allow_hosts`.
- With a pack, a segment whose first token is `ssh`, `scp` or `rsync`: ssh host = first non-option token (options taking a value: `-b -c -D -E -e -F -I -i -J -L -l -m -O -o -p -Q -R -S -W -w`); scp/rsync hosts = the `host:` prefix of any token; strip `user@`; also check the `-J` value. A host not in `allow_hosts` -> block.
- WebSearch `query`, WebFetch `url` and `prompt`: any `egress.forbidden_terms` match (case-insensitive) -> block.
- Every block message ends with `Run it yourself in a terminal if you really mean it.`

- [ ] **Step 1: Failing selftest**
```python
P = {"server": {"allow_hosts": ["my-gpu", "10.9.8.7"]}, "egress": {"forbidden_terms": ["secretset"]}}
BLOCK = ["gcloud compute instances stop other-vm --zone x", "gcloud compute instances delete a", "gcloud compute firewall-rules create x",
         "ssh other-host ls", "scp f other:/tmp", "ssh 10.0.0.5 ls", "gcloud compute ssh other-vm", "cd x && ssh -J jump my-gpu ls",
         "rsync -av d/ other:~/d"]
ALLOW = ["ssh my-gpu 'nvidia-smi'", "ssh -N -L 5000:127.0.0.1:5000 my-gpu", "ssh -o ConnectTimeout=15 -o BatchMode=yes my-gpu python3 -",
         "scp x my-gpu:~/a", "ssh user@10.9.8.7 ls", "gcloud compute instances list", "echo ssh other", "git push origin main",
         "python - <<'EOF'\nprint('it''s')\nEOF", "type C:\\Users\\a b\\x.txt", "echo 'unbalanced"]
for c in BLOCK: assert verdict("Bash", {"command": c}, P), c
for c in ALLOW: assert verdict("Bash", {"command": c}, P) is None, c
assert verdict("WebSearch", {"query": "SecretSet split"}, P) and verdict("WebSearch", {"query": "label-wise attention"}, P) is None
assert verdict("Bash", {"command": "ssh anywhere ls"}, None) is None and verdict("Bash", {"command": "gcloud compute instances stop x"}, None)
print("guard selftest ok")
```
- [ ] **Step 2: Run, expect failure** `python -X utf8 hooks/guard.py selftest`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect pass** -> `guard selftest ok`. End to end: `echo '{"tool_name":"Bash","tool_input":{"command":"gcloud compute instances stop x"}}' | python -X utf8 hooks/guard.py; echo rc=$?` -> reason on stderr, `rc=2`.
- [ ] **Step 5: Commit** `git add hooks && git commit -m "Add GCP isolation and egress guard hook"`

---

### Task 7: VM side (phro_server.py, phro_sync.py) and PC commands setup / manifest

**Files:**
- Create: `server/phro_server.py`, `server/phro_sync.py`
- Modify: `bin/phro.py` (`cmd_setup`, `cmd_manifest`, `remote_manifest`)

**Interfaces:**
- `phro_server.py` (stdlib; reads `project.json` next to itself):
  - `cron_merge(current: str, project: str, lines: list[str]) -> str`: drop lines ending with `# phro:<project>`, keep every other line unchanged, append each new line + ` # phro:<project>`.
  - `port_free(port: int) -> bool`: bind test on `127.0.0.1`.
  - `busy_gpus() -> set[int]`: every GPU index with any process in `nvidia-smi --query-compute-apps=gpu_uuid` (mapped through `--query-gpu=index,uuid`).
  - `queue_step(lines: list[str], busy: set[int], gpus: int) -> tuple[list[tuple[int, str]], list[str]]`: for each free GPU, launch the first `<gpu> <script>` line for it; return (launches, remaining lines incl. comments, in order).
  - `launch(gpu: int, script: str, root: Path) -> int`: `setsid nohup` with `CUDA_VISIBLE_DEVICES=<gpu>`, log `<root>/logs/<script stem>.log`, PID to `<root>/pids/<gpu>.pid`; returns the PID.
  - `manifest_get(path, name) -> dict`, `manifest_set(path, name, pairs: dict) -> None` (atomic: temp file + `os.replace`).
  - `setup(root: Path, p: dict) -> int`: abort with a message if `not port_free(port)`; create `root/{runs,logs,pids,artifacts}`; `python3 -m venv <venv>`; `pip install mlflow==2.17.2`; start `mlflow server --host 127.0.0.1 --port <port> --backend-store-uri sqlite:///<root>/mlflow.db --artifacts-destination <root>/artifacts` with nohup and a PID file; crontab through `cron_merge` with `@reboot <start command>`, `*/10 * * * * <venv>/bin/python <root>/phro_sync.py`, `*/5 * * * * python3 <root>/phro_server.py queue`. No sudo, nothing outside `root` except the user's crontab.
  - CLI: `setup | queue | manifest get NAME | manifest set NAME k=v ... | selftest`.
- `phro_sync.py` (VM venv; `import mlflow` inside `main` only):
  - `parse_run_dir(d: Path) -> dict`: `{"name", "stage", "params", "tags", "metrics": [(key, value, step)], "units": Path | None, "done": bool}`; `"final"` steps become the largest integer step seen (0 if none).
  - `sync(root: Path, p: dict, client, only: Path | None = None) -> list[str]`: for each run dir under `root/runs` with `DONE` whose content hash differs from `.phro_synced`: delete the old run found by tag `phro_source`, create it again in the stage's experiment, log params, metrics, tags (run.json tags + manifest entry fields), artifact `units.jsonl`, write `.phro_synced`. Each pass also re-applies changed manifest fields as tags on existing runs. Returns synced names.
  - CLI: `[--path DIR] | selftest`.
- `bin/phro.py`:
  - `cmd_setup`: when `p["server"]["read_only"]`, print `server.read_only: refusing to write to <host>` and return 1 before any ssh. Else `scp server/phro_server.py server/phro_sync.py <pack project.json> <host>:<root>/`, then `ssh(p, "python3 <root>/phro_server.py setup")`.
  - `remote_manifest(p) -> dict | None`: `ssh(p, "cat <manifest>")` parsed as JSON (works on a read-only VM without the kit); `None` on failure.
  - `cmd_manifest get NAME | set NAME k=v ...`: `get` from `remote_manifest`; `set` refuses on `read_only` like `cmd_setup`, else `ssh(p, "python3 <root>/phro_server.py manifest set ...")` with each argument `shlex.quote`d.

- [ ] **Step 1: Failing selftests**
```python
# phro_server.py
cur = "0 1 * * * backup.sh\n*/5 * * * * old # phro:proj\n*/5 * * * * other # phro:proj2\n"
out = cron_merge(cur, "proj", ["*/5 * * * * new"])
assert "backup.sh" in out and "old" not in out and "# phro:proj2" in out and out.rstrip().endswith("new # phro:proj")
assert queue_step(["0 a.sh", "0 b.sh", "1 c.sh", "# note"], busy={1}, gpus=2) == ([(0, "a.sh")], ["0 b.sh", "1 c.sh", "# note"])
assert queue_step(["2 x.sh"], busy={2}, gpus=4) == ([], ["2 x.sh"])          # a foreign process on GPU2 counts as busy
manifest_set(tmp / "m.json", "r1", {"intro": "x. 기대: y"}); assert manifest_get(tmp / "m.json", "r1")["intro"].endswith("y")
# phro_sync.py: d = tmp/"runs/s1/r1_s0" with run.json, metrics.jsonl (steps 100, 200, "final"), units.jsonl, DONE
r = parse_run_dir(d)
assert r["done"] and ("heldout/f1", 0.8, 200) in r["metrics"] and r["units"].name == "units.jsonl"
```
and in `bin/phro.py` selftest: `assert cmd_setup(load_pack_dict({**EXRAW, "server": {**EXRAW["server"], "read_only": True}}), None) == 1` (no ssh attempted).
- [ ] **Step 2: Run each, expect failure:** `python -X utf8 server/phro_server.py selftest`, `python -X utf8 server/phro_sync.py selftest`, `python -X utf8 bin/phro.py selftest`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run each, expect pass.**
- [ ] **Step 5: Live install check, only on a VM the user names in chat.** Ask for the ssh alias of a fresh VM (never a `read_only` pack's VM). With a pack pointing at it: `python bin/phro.py setup`; then `ssh <alias> crontab -l` shows three `# phro:<name>` lines and the user's other lines unchanged; `python bin/phro.py list` through the tunnel prints an empty table. If the user names no VM, skip this step and say so in the task report.
- [ ] **Step 6: Commit** `git add server bin/phro.py && git commit -m "Add VM server kit, sync and remote manifest"`

---

### Task 8: Foreman (generalized) and plugin wiring

**Files:**
- Create: `foreman/foreman.py` (port of the legacy foreman in `domains/<first pack>/legacy/skills/`)
- Create: `.mcp.json`: server `phro-foreman`, command `python`, args `["-X", "utf8", "${CLAUDE_PLUGIN_ROOT}/foreman/foreman.py", "serve"]`
- Modify: `hooks/hooks.json`: add a `Stop` entry `python -X utf8 "${CLAUDE_PLUGIN_ROOT}/foreman/foreman.py" hook`, timeout 60

**Interfaces:**
- Consumes: `find_pack`, `load_pack`, `snapshot`, `run_name` (Tasks 1, 5).
- Produces: the legacy tools and CLI (`serve | hook | check | report --summary --ids --next --blocked | selftest`) with only these changes:
  - State, reports and `ENABLED` live in `<pack _dir>/foreman/`. No pack or no `ENABLED` -> `hook` does nothing.
  - Legacy `HOST`/`REMOTE`/`fetch` -> `phro.snapshot(p)`; legacy `run_name(path)` -> `phro.run_name(path, p["server"]["run_name_rules"])`.
  - `PLAN` = `<pack _dir>/plan.md`; ideas = `<pack _dir>/ledger/idea-bag.md`.
  - `expectation(intro)` matches `기대` or `Expected` followed by `:` or `：`.
  - Order texts name `python bin/phro.py report --run <n>` and `phro_server.py queue`; MCP `serverInfo.name` = `phro-foreman`; tool descriptions name no host.
  - After the legacy three candidate groups: always `phro.py audit: 전체 예산 재검증·leave-one-out 대기 확인`; and `돌파구 조사: ledger/insights.md 열린 한계` when the other candidates are empty.

- [ ] **Step 1: Port the legacy selftest** with the Task 5 rules `R` and generic run names. It points the module pack dir at a temp dir holding an empty `plan.md`. Add:
```python
    assert expectation("... Expected: F1 >= 0.8").startswith("F1") and expectation("x 기대: 0.80 이상").startswith("0.80")
    block, text = orders({"gpus": [[0, 0, 0]], "done": {}, "docs": {}, "waiting": [], "queue": []}, {"seen": []})
    assert block and "돌파구" in text            # idle GPU and empty plan -> breakthrough candidate
```
- [ ] **Step 2: Run, expect failure** `python -X utf8 foreman/foreman.py selftest`.
- [ ] **Step 3: Implement the port.**
- [ ] **Step 4: Run, expect pass**; then `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -X utf8 foreman/foreman.py serve` lists both tools; `cd "$TMP" && echo '{}' | python -X utf8 <repo>/foreman/foreman.py hook; echo rc=$?` prints nothing and `rc=0`.
- [ ] **Step 5: Commit** `git add foreman .mcp.json hooks/hooks.json && git commit -m "Add pack-based foreman and plugin wiring"`

---

### Task 9: Skill phro-mlflow

**Files:**
- Create: `skills/phro-mlflow/SKILL.md`
- Create: `skills/phro-mlflow/references/{diagnosis,eval,sweep,insights,legacy-review,lessons,probe,idea-bag,breakthrough,contract}.md`

Source: `domains/<first pack>/legacy/skills/` (read-only). Rule for every file: keep the procedure; remove every project fact (host, IP, paths, stage names, metric keys, data set names, dates, ticket and insight numbers, domain examples); where a fact was used, point to the pack (`domain.md`, a `project.json` key) or a `phro.py` command. Report language follows `project.json language`.

| Target | From | Changes |
|---|---|---|
| `SKILL.md` | legacy main skill | Frontmatter `name: phro-mlflow`; description lists Korean and English triggers (결과 어때, 끝났어, 진행상황, 남는 gpu, sweep, 다음 실험, 돌파구, 새 아이디어, MLflow, results, next experiment). Section 1: read `.phro/domain.md` goal chain, refresh numbers with `phro.py baseline`. Hard rules: numbers only from `phro.py`; the VM is `project.json server.host` only; launching needs the user's OK unless the item is in `plan.md`; name the metric key with every number. Commands table = the Tasks 3-5 and 7 commands. Workflows kept; breakthrough points to `references/breakthrough.md`. No pack -> tell the user to run phro-onboard. |
| `diagnosis.md`, `insights.md`, `legacy-review.md`, `lessons.md`, `probe.md`, `eval.md` | same names | Generalized per the rule. `insights.md` adds the "열린 한계" table (spec 7.1). `legacy-review.md` drops the trap table: "project traps live in `domain.md`". `lessons.md` keeps the 4 escalation steps, drops the project lessons. |
| `sweep.md` | legacy `sweep.md` | Manifest fields of spec 4, written with `phro.py manifest set`; adds the 5 rules of spec 9; HPO section names Optuna TPE + pruner, no project paths. |
| `idea-bag.md` | legacy header and entry format | Format only, no ideas. `targets` names an `L-ID`. Adds `evidence:` lines `research/<topic>/sources/<file>:<line> (논문/본인/정리)`. |
| `breakthrough.md` | legacy idea-scan skill + spec 7.2 | Steps 1-6 of spec 7.2; calls the installed `imozuru` skill with topic dir `.phro/research/<topic>`; required claims; privacy: generic problem statement only, the guard hook blocks pack terms; imozuru missing -> say how to install and stop. |
| `contract.md` | spec 3.1, 4 | Pack keys table, MLflow contract table, the 10-line Python snippet a training script uses to write `run.json`, `metrics.jsonl`, `units.jsonl`, `DONE`. |

- [ ] **Step 1: Write the files.**
- [ ] **Step 2: References resolve:** `grep -oh "references/[a-z-]*\.md" skills/phro-mlflow/SKILL.md skills/phro-mlflow/references/*.md | sort -u | while read f; do test -f skills/phro-mlflow/$f || echo MISSING $f; done` prints nothing.
- [ ] **Step 3: No project facts:** from inside the first pack's project dir (its `egress.forbidden_terms` and hosts are the check list), `python <repo>/bin/phro.py leakcheck <repo>` -> `leakcheck clean`. Before Task 11 exists, the private appendix gives a temporary term list. Also `grep -rnP "\x{2014}|\x{2013}" skills/` (em and en dash) prints nothing.
- [ ] **Step 4: Commit** `git add skills/phro-mlflow && git commit -m "Add phro-mlflow skill"`

---

### Task 10: Skill phro-onboard

**Files:** Create `skills/phro-onboard/SKILL.md`

Content (spec 6): frontmatter `name: phro-onboard`, triggers (새 프로젝트, 워크플로우 설계, 모델 조합, 도메인 팩, onboard, set up AutoML for). Sections:
1. Choose the mode: workflow only -> Mode 1; fixed models and structure -> Mode 2.
2. Mode 1, spec 6 steps 1-5: one imozuru dig per stage at `.phro/research/<stage>-models` with the 4 required claims; composition options table citing `sources/<file>:<line>`; the demonstration probe is proposed and run only after the user's OK.
3. Mode 2 outputs per stage (spec 6).
4. Write the pack: `project.json` from `templates/project.example.json`; `domain.md` sections (goal chain table, constraints, data and privacy rules, glossary, report style); `plan.md`; `ledger/{insights,idea-bag,lessons}.md` with the headers from the phro-mlflow references.
5. `python <plugin>/bin/phro.py check-pack` must print `pack ok`.
6. Independent review subagent; the prompt is in the file: read-only, check the pack against the stated goal, list missing stages, unmeasurable metrics, gates without CI, egress gaps.
7. Ask the user to approve; run `phro.py setup` only on a VM the user names.

- [ ] **Step 1: Write SKILL.md.**
- [ ] **Step 2: Dry run** in a temp dir: follow Mode 2 for a toy two-stage workflow ("text classification -> summary") without GPUs or imozuru; the produced `.phro/project.json` passes `python <repo>/bin/phro.py check-pack` -> `pack ok: <name>`. Delete the temp dir.
- [ ] **Step 3: Commit** `git add skills/phro-onboard && git commit -m "Add phro-onboard skill"`

---

### Task 11: First private pack and read-only acceptance test

**Files (outside the repo, never committed):** the first pack's `.phro/` (project.json, domain.md, plan.md, ledger pointers, `adapters/export_units.py`, `accept/`).

Every project value, run name and command for this task is in the private appendix `<first pack>/.phro/accept/acceptance-plan.md`. Follow it step by step. Pass criteria in generic terms:
- `check-pack` prints `pack ok`; the pack has `server.read_only: true` and no `foreman/ENABLED`.
- Units are exported read-only on the VM: an adapter imports the existing sync module and prints JSON lines to stdout, nothing is written on the VM; they are saved under `accept/units/`.
- `ci` parity: legacy query script vs `phro.py ci ... --units`, every numeric cell identical.
- `baseline` parity: per family, seed count and `mean [min, max]` identical for the stages the appendix lists; known gaps recorded.
- `compare` parity: same value in every shared key column.
- foreman parity: on one recorded snapshot, the same finished run names and idle GPU numbers in the legacy and new order texts.
- Engine bugs found here are fixed in the repo, each with a selftest assert that reproduces it, and the parity steps re-run until they pass. Results go to `accept/RESULT.md` (private). Commit only repo changes: `git commit -am "Fix <thing> found by acceptance test"`.

---

### Task 12: README, final checks, push on approval

**Files:** Create `README.md`: what it is, install as a Claude Code plugin, the two skills, the `phro.py` command table, pack location `.phro/`, the GCP isolation rules of spec 8, "domain packs are never committed".

- [ ] **Step 1: Write README.md.**
- [ ] **Step 2: All selftests:** `for f in bin/phro.py hooks/guard.py foreman/foreman.py server/phro_server.py server/phro_sync.py; do python -X utf8 $f selftest || echo FAIL $f; done` -> 5 `selftest ok` lines, no `FAIL`.
- [ ] **Step 3: Leak and hygiene:** from the first pack's project dir, `python <repo>/bin/phro.py leakcheck <repo>` -> `leakcheck clean`; `git -C <repo> ls-files | grep -E "^domains/|\.phro/"` prints nothing; `git -C <repo> log --format="%an <%ae>%n%B" | grep -iE "co-authored|generated with"` prints nothing.
- [ ] **Step 4: Commit** `git add README.md && git commit -m "Add README"`
- [ ] **Step 5: Ask the user before pushing** (public repo). On yes: `git push -u origin main`.
