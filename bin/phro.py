#!/usr/bin/env python3
"""phro: PC-side CLI and library for AutoML loops on one GCP GPU VM with MLflow."""
import copy
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path  # urllib is imported where it is used: the guard hook imports this module on every Bash call

ROOT = Path(__file__).resolve().parents[1]
PARITY_CALLS = [{f"c{i}": (i % 3, i % 2, (i + 1) % 2) for i in range(40)},
                {f"c{i}": ((i + 1) % 3, 0, i % 2) for i in range(5, 45)}]
REQUIRED = ["name", "goal", "stages", "metrics", "eval_sets", "server", "egress"]


def _real(v):
    """None for an empty value or an unexpanded ${...} placeholder (an MCP env the host left as text)."""
    return None if not v or str(v).startswith("${") else v


def find_pack(start=None):
    d = Path(_real(start) or _real(os.environ.get("CLAUDE_PROJECT_DIR")) or Path.cwd()).resolve()
    for q in [d, *d.parents]:
        f = q / ".phro" / "project.json"
        if f.is_file():
            return f
    return None


def load_pack_dict(d):
    p = copy.deepcopy(d)
    if not isinstance(p, dict):
        return p
    for k in ("stats", "server", "egress"):
        if not isinstance(p.get(k, {}), dict):
            return p
    p.setdefault("stats", {})
    p["stats"].setdefault("seeds_min", 4)
    p["stats"].setdefault("bootstrap", 1000)
    p["stats"].setdefault("min_units", 30)
    p.setdefault("derived_params", [])
    p.setdefault("tag_alias", {})
    s = p.setdefault("server", {})
    root = s["root"] if isinstance(s.get("root"), str) else ""
    s.setdefault("manifest", root + "/runs.json")
    s.setdefault("queue", root + "/gpu_queue.txt")
    s.setdefault("done_globs", ["runs/**/DONE"])
    s.setdefault("run_name_rules", [["(?:.*/)?([^/]+)/DONE$", r"\1"]])
    s.setdefault("read_only", False)
    s.setdefault("allow_hosts", [])
    p.setdefault("egress", {}).setdefault("forbidden_terms", [])
    return p


def load_pack(path):
    path = Path(path)
    p = load_pack_dict(json.loads(path.read_text(encoding="utf-8")))
    if isinstance(p, dict):
        p["_dir"] = path.parent
    return p


def _has_cycle(deps):
    state = {}

    def visit(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(m) for m in deps.get(n, []) if m in deps):
            return True
        state[n] = 2
        return False

    return any(visit(n) for n in deps)


def check_pack(p):
    if not isinstance(p, dict):
        return ["project.json: top level must be an object"]
    errs = [f"missing key: {k}" for k in REQUIRED if k not in p]
    stages = p.get("stages") if isinstance(p.get("stages"), list) else []
    for k in ("stages", "metrics"):
        if k in p and not (isinstance(p[k], list) and all(isinstance(x, dict) for x in p[k])):
            errs.append(f"{k}: must be a list of objects")
    for k in ("server", "egress", "eval_sets"):
        if k in p and not isinstance(p[k], dict):
            errs.append(f"{k}: must be an object")
    stages = [x for x in stages if isinstance(x, dict)]
    names = [s.get("name") for s in stages]
    for n in sorted({n for n in names if names.count(n) > 1}):
        errs.append(f"duplicate stage name: {n}")
    for s in stages:
        if "depends_on" in s and not isinstance(s["depends_on"], list):
            errs.append(f"stage {s.get('name')}: depends_on must be a list")
        for d in s.get("depends_on") if isinstance(s.get("depends_on"), list) else []:
            if d not in names:
                errs.append(f"stage {s.get('name')}: depends_on {d!r} is not a stage")
    if _has_cycle({s.get("name"): list(s.get("depends_on") or []) for s in stages if isinstance(s.get("depends_on", []), list)}):
        errs.append("stages: depends_on has a cycle")
    for m in p.get("metrics") if isinstance(p.get("metrics"), list) else []:
        if not isinstance(m, dict):
            continue
        k = m.get("key")
        if m.get("stage") not in names:
            errs.append(f"metric {k}: stage {m.get('stage')!r} is not a stage")
        if m.get("dir") not in ("max", "min"):
            errs.append(f"metric {k}: dir must be max or min")
        a = m.get("agg") if isinstance(m.get("agg"), dict) else {}
        need = {"mean": ["field"], "f1": ["tp", "fp", "fn"]}.get(a.get("type"))
        if need is None:
            errs.append(f"metric {k}: agg.type must be mean or f1")
        else:
            errs += [f"metric {k}: agg.{f} missing" for f in need if f not in a]
    if "stats" in p and not isinstance(p["stats"], dict):
        errs.append("stats: must be an object")
    st = p.get("stats") if isinstance(p.get("stats"), dict) else {}
    for k, lo in (("seeds_min", 2), ("bootstrap", 100), ("min_units", 1)):
        if k in st and not (isinstance(st[k], int) and not isinstance(st[k], bool) and st[k] >= lo):
            errs.append(f"stats.{k} must be an integer >= {lo}")
    if isinstance(p.get("eval_sets"), dict) and not (isinstance(p["eval_sets"].get("select"), str) and p["eval_sets"]["select"]):
        errs.append("eval_sets.select must name the selection set")
    s = p.get("server") if isinstance(p.get("server"), dict) else {}
    if not s.get("host"):
        errs.append("server.host missing")
    elif not isinstance(s["host"], str) or s["host"].startswith("-"):
        errs.append("server.host must be a name that does not start with -")
    elif s["host"] not in s.get("allow_hosts", []):
        errs.append("server.host not in server.allow_hosts")
    for k in ("root", "venv") if not s.get("read_only") else ("root",):  # venv is read only by setup, which a read_only pack refuses
        if not (isinstance(s.get(k), str) and s[k]):
            errs.append(f"server.{k} must be a non-empty string")
    for k, lo in (("mlflow_port", 1), ("gpus", 1)):
        if not (isinstance(s.get(k), int) and not isinstance(s[k], bool) and s[k] >= lo):
            errs.append(f"server.{k} must be an integer >= {lo}")
    return errs


def tag(run, name, p):
    return run["tags"].get(p["tag_alias"].get(name, name), "")


def leak_terms(p):
    s, terms = p["server"], []
    for t in [*p["egress"]["forbidden_terms"], s.get("host"), *s.get("allow_hosts", [])]:
        if t and t.lower() not in terms:
            terms.append(t.lower())
    return terms


def leak_hits(p, files):
    terms = leak_terms(p)
    hits = []
    for f, text in files.items():
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            hits += [(f, i, t) for t in terms if t in low]
    return hits


def read_pack(f):
    """Shared by main and check-pack: (pack, errors); bad shapes give errors, not tracebacks."""
    try:
        pack = load_pack(f)
    except (OSError, ValueError) as e:
        return None, [f"cannot read {f}: {e}"]
    errs = check_pack(pack)
    return (None if errs else pack), errs


def cmd_check_pack(p, a):
    f = find_pack()
    if f is None:
        print(f"no .phro/project.json found from {os.environ.get('CLAUDE_PROJECT_DIR') or Path.cwd()}")
        return 1
    pack, errs = read_pack(f)
    if errs:
        print("\n".join(errs))
        return 1
    print(f"pack ok: {pack['name']}")
    return 0


def history_hits(d, terms):
    """[(term, commit count)] for terms found in any commit reachable from any ref."""
    revs = subprocess.run(["git", "-C", d, "rev-list", "--all"], capture_output=True, text=True, check=True).stdout.split()
    out = []
    for t in terms:
        seen = set()
        for i in range(0, len(revs), 200):  # ponytail: chunked argv, a long history only costs more calls
            r = subprocess.run(["git", "-C", d, "grep", "-I", "-i", "-F", "-l", "-e", t, *revs[i:i + 200]],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            seen |= {l.split(":", 1)[0] for l in r.stdout.splitlines() if l}
        if seen:
            out.append((t, len(seen)))
    return out


def cmd_leakcheck(p, a):
    hist = "--no-history" not in a
    a = [x for x in a if not x.startswith("--")]
    d = a[0] if a else "."
    out = subprocess.run(["git", "-C", d, "ls-files", "-z"], capture_output=True, check=True).stdout
    files = {}
    for rel in filter(None, out.decode("utf-8").split("\0")):
        try:
            files[rel] = (Path(d) / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            pass
    hits = leak_hits(p, files)
    for f, n, t in hits:
        print(f"{f}:{n}: {t}")
    old = history_hits(d, leak_terms(p)) if hist else []
    for t, n in old:
        print(f"history: {t} in {n} commits")
    if hits or old:
        return 1
    print("leakcheck clean" + ("" if hist else " (working tree only)"))
    return 0


def family(name):
    return re.sub(r"_s\d+$", "", name)


def agg_value(rows, agg):
    if agg["type"] == "mean":
        return sum(r[agg["field"]] for r in rows) / len(rows)
    tp, fp, fn = (sum(r[agg[k]] for r in rows) for k in ("tp", "fp", "fn"))
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def boot(units, agg, B=1000, seed=0):
    """units: one {id: row} per run. Paired bootstrap over the common ids; d_ci is vs units[0]."""
    ids = sorted(set.intersection(*(set(u) for u in units))) if units else []
    if not ids:
        return {"n": 0, "value": [], "ci": [], "d_ci": []}
    rng = random.Random(seed)
    F = [[] for _ in units]
    for _ in range(B):
        smp = [rng.choice(ids) for _ in ids]
        for i, u in enumerate(units):
            F[i].append(agg_value([u[x] for x in smp], agg))
    q = lambda v: (sorted(v)[int(0.025 * B)], sorted(v)[int(0.975 * B) - 1])
    return {"n": len(ids), "value": [agg_value([u[x] for x in ids], agg) for u in units],
            "ci": [q(v) for v in F], "d_ci": [q([a - b for a, b in zip(v, F[0])]) for v in F]}


def downstream(p, stage):
    deps = {s["name"]: s.get("depends_on", []) for s in p["stages"]}
    out = []

    def walk(n):
        for m in deps:
            if n in deps[m] and m not in out:
                out.append(m)
                walk(m)

    walk(stage)
    done = []
    while len(done) < len(out):
        done.append(next(n for n in deps if n in out and n not in done and all(d in done or d not in out for d in deps[n])))
    return done


class Unreachable(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code  # HTTP status when MLflow answered, None when the connection failed


_open = None  # None = urllib.request.urlopen; tests swap it with _set_open


def _set_open(fn):
    global _open
    _open = fn


def _call(p, path, data=None):
    import urllib.error
    import urllib.request
    port = p["server"]["mlflow_port"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with (_open or urllib.request.urlopen)(req, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise Unreachable(f"MLflow answered HTTP {e.code} for {path}", e.code)
    except OSError:
        raise Unreachable(f"MLflow not reachable. Start the tunnel: ssh -N -L {port}:127.0.0.1:{port} {p['server']['host']}")


def rest(p, path, body=None):
    return json.loads(_call(p, path, None if body is None else json.dumps(body).encode("utf-8")) or b"{}")


def get_bytes(p, path):
    return _call(p, path)


def _kv(items):
    return {x["key"]: x["value"] for x in items or []}


def fetch_runs(p):
    """All active runs of all experiments, normalized; metrics hold the latest values."""
    exps = rest(p, "/api/2.0/mlflow/experiments/search", {"max_results": 1000}).get("experiments", [])
    out = []
    for e in exps:
        tok = None
        while True:
            body = {"experiment_ids": [e["experiment_id"]], "max_results": 1000}
            if tok:
                body["page_token"] = tok
            res = rest(p, "/api/2.0/mlflow/runs/search", body)
            for r in res.get("runs", []):
                i, d = r["info"], r.get("data", {})
                end = i.get("end_time")
                out.append({"exp": e["name"], "id": i["run_id"], "name": i.get("run_name", i["run_id"]),
                            "start": int(i.get("start_time") or 0), "end": int(end) if end else None,
                            "params": _kv(d.get("params")), "tags": _kv(d.get("tags")),
                            "metrics": {m["key"]: float(m["value"]) for m in d.get("metrics", [])}})
            tok = res.get("next_page_token")
            if not tok:
                break
    return out


def history(p, run_id, key):
    import urllib.parse
    q = urllib.parse.urlencode({"run_id": run_id, "metric_key": key})
    return [(int(m.get("step", 0)), float(m["value"]))
            for m in rest(p, "/api/2.0/mlflow/metrics/get-history?" + q).get("metrics", [])]


def names_in(text, known):
    """Known names found in text, longest first so 'a' is not matched inside 'a2'; ASCII boundaries so a Korean particle may follow."""
    out, left = [], text
    for n in sorted(known, key=len, reverse=True):
        pat = rf"(?<![A-Za-z0-9_-]){re.escape(n)}(?![A-Za-z0-9_-])"
        if re.search(pat, left):
            out.append(n)
            left = re.sub(pat, " ", left)
    return [n for n in known if n in out]


def param_diff(a, b, derived):
    return [(k, a.get(k, "-"), b.get(k, "-")) for k in sorted(set(a) | set(b)) if k not in derived and a.get(k) != b.get(k)]


def compare_table(runs, set_, p):
    pre = set_ + "/"
    keys = [m["key"] for m in p["metrics"] if m["key"].startswith(pre)]
    keys += sorted({k for r in runs for k in r["metrics"] if k.startswith(pre) and "/ci_" not in k and k not in keys})
    keys = keys[:12]

    def cell(r, k):
        m, n = r["metrics"], k[len(pre):]
        if k not in m:
            return "-"
        ci = f" [{m[f'{pre}ci_lo_{n}']:.4f}, {m[f'{pre}ci_hi_{n}']:.4f}]" if f"{pre}ci_lo_{n}" in m and f"{pre}ci_hi_{n}" in m else ""
        return f"{m[k]:.4f}{ci}"

    head = ["run", "experiment", f"gate_{set_}", *[k[len(pre):] for k in keys]]
    return head, [[r["name"], r["exp"], tag(r, f"gate_{set_}", p) or "-", *[cell(r, k) for k in keys]] for r in runs]


def table(head, rows):
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for x in rows:
        print("| " + " | ".join(str(v) for v in x) + " |")


def _opt(a, flag):
    return a[a.index(flag) + 1] if flag in a[:-1] else None


def _gates(r):
    return sorted(k for k in r["tags"] if k.startswith("gate_"))


def _status(r, p):
    return tag(r, "status", p) or "-"


def cmd_list(p, a):
    ex = _opt(a, "--exp")
    rows = [[r["exp"], r["name"], _status(r, p), tag(r, "sweep", p) or "-",
             " ".join(f"{k[5:]}={r['tags'][k]}" for k in _gates(r)) or "-"]
            for r in fetch_runs(p) if not ex or r["exp"].startswith(ex)]
    table(["experiment", "run", "status", "sweep", "gates"], sorted(rows))
    return 0


def cmd_report(p, a):
    runs = fetch_runs(p)
    name, hours = _opt(a, "--run"), _opt(a, "--since")
    since = (time.time() - float(hours) * 3600) * 1000 if hours else None
    pick = [r for r in runs if (name and r["name"] == name) or (since and (r["end"] or 0) >= since)]
    if not pick:
        print("no matching runs: use `list` for exact run names")
        return 1
    known = [r["name"] for r in runs]
    for r in sorted(pick, key=lambda r: r["end"] or 0):
        print(f"\n## {r['exp']} / {r['name']}  status={_status(r, p)} sweep={tag(r, 'sweep', p) or '-'} "
              f"insight={tag(r, 'insight', p) or '-'} gates={ {k: r['tags'][k] for k in _gates(r)} }")
        for k in ("hypothesis", "intro", "result"):
            if tag(r, k, p):
                print(f"{k}: {tag(r, k, p)}")
        others = names_in(tag(r, "compare", p), [n for n in known if n != r["name"]])
        peers = [x for x in runs if x["name"] in others]
        for g in _gates(r):
            if peers:
                print(f"\ncompare on {g[5:]}:")
                table(*compare_table([r, *peers], g[5:], p))
    return 0


def cmd_compare(p, a):
    sweep, set_ = _opt(a, "--sweep"), _opt(a, "--set") or p["eval_sets"]["select"]  # heldout is a look; ask for it with --set
    names = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--set", "--sweep"))]
    pick = [r for r in fetch_runs(p) if r["name"] in names or (sweep and sweep in tag(r, "sweep", p).split(","))]
    if sweep:
        for r in pick[:1]:
            print(f"sweep {sweep}: axis={tag(r, 'axis', p) or '-'} | fixed={tag(r, 'fixed', p) or '-'} | pick={tag(r, 'pick', p) or '-'}")
        for r in pick:
            if tag(r, "hypothesis", p):
                print(f"hypothesis[{r['name']}]: {tag(r, 'hypothesis', p)}")
    table(*compare_table(pick, set_, p))
    return 0


def cmd_diff(p, a):
    new, old = (a[0] if a and not a[0].startswith("--") else None), _opt(a, "--vs")
    by = {r["name"]: r for r in fetch_runs(p)}
    for n in (new, old):
        if n not in by:
            print(f"run not found: {n}")
            return 1
    print(f"{new} vs {old}: every row must be the declared axis or a reviewed change; anything else is a confound")
    rows = [[k, str(x)[:70], str(y)[:70]] for k, x, y in param_diff(by[new]["params"], by[old]["params"], set(p["derived_params"]))]
    table(["param", new, old], rows or [["(no difference)", "", ""]])
    return 0


def ssh(p, remote_cmd, stdin=None, timeout=40):
    """The only place that starts ssh."""
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", p["server"]["host"], remote_cmd],
                          input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def remote_config(p):
    s = p["server"]
    return "CFG = " + json.dumps({"root": s["root"], "done_globs": s["done_globs"], "manifest": s["manifest"], "queue": s["queue"]})


REMOTE = r'''
import glob, json, os, re, subprocess
root = os.path.expanduser(CFG["root"])


def sh(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, errors="replace").stdout
    except OSError:
        return ""


def lines(path):
    try:
        return [l.strip() for l in open(os.path.expanduser(path), encoding="utf-8")]
    except OSError:
        return []


idx, gpus, procs = {}, [], {}
for l in sh(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]).splitlines():
    f = [x.strip() for x in l.split(",")]
    try:
        gpus.append([int(f[0]), int(f[2]), int(f[3])])
        idx[f[1]] = f[0]
    except (ValueError, IndexError):
        pass
for l in sh(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]).splitlines():
    f = [x.strip() for x in l.split(",")]
    if len(f) != 2:
        continue
    args = sh(["ps", "-o", "args=", "-p", f[0]]).strip()
    m = re.search(r"--out\s+(\S+)", args)
    procs.setdefault(idx.get(f[1], "?"), []).append(os.path.basename(m.group(1).rstrip("/")) if m else args[-30:])
done = {}
for g in CFG["done_globs"]:
    for q in glob.glob(os.path.join(root, g), recursive=True):
        try:
            done[os.path.relpath(q, root).replace(os.sep, "/")] = int(os.path.getmtime(q))
        except OSError:
            pass
try:
    man = json.loads(open(os.path.expanduser(CFG["manifest"]), encoding="utf-8").read())
except (OSError, ValueError):
    man = {}
docs = {k: {f: str(v.get(f, ""))[:900] for f in ("intro", "result")} for k, v in man.items() if isinstance(v, dict)} if isinstance(man, dict) else {}
wait = sorted({w for l in sh(["pgrep", "-af", "run_[a-z0-9_]*[.]sh"]).splitlines() if "pgrep" not in l for w in re.findall(r"run_[a-z0-9_]*\.sh", l)})
queue = [l for l in lines(CFG["queue"]) if l and not l.startswith("#")]
print(json.dumps({"gpus": gpus, "procs": procs, "done": done, "docs": docs, "waiting": wait, "queue": queue}))
'''


def snapshot(p):
    try:
        r = ssh(p, "python3 -", remote_config(p) + "\n" + REMOTE)
        d = json.loads(r.stdout) if r.returncode == 0 else None
        return d if isinstance(d, dict) else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_name(path, rules):
    for pat, rep in rules:
        m = re.search(pat, path)
        if m:
            return m.expand(rep) or None
    return None


def cmd_status(p, a):
    s = snapshot(p)
    if s is None:
        print(f"ssh to {p['server']['host']} failed")
        return 1
    table(["GPU", "mem", "util", "running"],
          [[i, f"{m} MiB", f"{u}%", ", ".join(s["procs"].get(str(i), [])) or "FREE"] for i, u, m in s["gpus"]])
    print("queue:", *(s["queue"] or ["-"]), sep="\n  ")
    print("waiting scripts:", " ".join(s["waiting"]) or "-")
    return 0


def scp(p, files, remote_dir):
    """With ssh, the only process starter for remote work; the host comes from the pack only."""
    return subprocess.run(["scp", "-o", "BatchMode=yes", *map(str, files), f"{p['server']['host']}:{remote_dir}/"],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def rpath(path):
    """A remote path for a shell string: shlex.quote everything but a leading ~/ so the remote shell still expands it."""
    return "~/" + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)


def remote_manifest(p):
    try:
        r = ssh(p, "cat " + rpath(p["server"]["manifest"]))
        d = json.loads(r.stdout) if r.returncode == 0 else None
        return d if isinstance(d, dict) else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _refuse_ro(p):
    if p["server"]["read_only"]:
        print(f"server.read_only: refusing to write to {p['server']['host']}")
        return True
    return False


def cmd_setup(p, a):
    if _refuse_ro(p):
        return 1
    root = p["server"]["root"]
    try:
        r = ssh(p, "mkdir -p " + rpath(root))
        if r.returncode == 0:
            r = scp(p, [ROOT / "server/phro_server.py", ROOT / "server/phro_sync.py", p["_dir"] / "project.json"], root)
        if r.returncode == 0:
            r = ssh(p, "python3 " + rpath(root + "/phro_server.py") + " setup", timeout=900)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"setup failed: {e}")
        return 1
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


def cmd_manifest(p, a):
    if len(a) >= 2 and a[0] == "get":
        m = remote_manifest(p)
        if m is None:
            print(f"cannot read manifest on {p['server']['host']}")
            return 1
        print(json.dumps(m.get(a[1], {}), ensure_ascii=False))
        return 0
    if len(a) >= 3 and a[0] == "set" and all("=" in x for x in a[2:]):
        if _refuse_ro(p):
            return 1
        r = ssh(p, "python3 " + rpath(p["server"]["root"] + "/phro_server.py") + " manifest set " + " ".join(shlex.quote(x) for x in a[1:]))
        print(r.stderr.strip() if r.returncode else "manifest updated")
        return r.returncode
    print("usage: manifest get NAME | manifest set NAME key=value ...")
    return 1


def cmd_queue(p, a):
    if len(a) == 3 and a[0] == "add":
        if _refuse_ro(p):
            return 1
        r = ssh(p, "python3 " + rpath(p["server"]["root"] + "/phro_server.py") + " queue add " + " ".join(shlex.quote(x) for x in a[1:]))
        print((r.stdout + r.stderr).strip() or f"queue add exit {r.returncode}")
        return r.returncode
    if a == ["list"]:
        r = ssh(p, "cat " + rpath(p["server"]["queue"]) + " 2>/dev/null; true")
        if r.returncode:
            print(f"cannot read queue on {p['server']['host']}")
            return 1
        print("\n".join(l for l in r.stdout.splitlines() if l.strip() and not l.startswith("#")) or "queue empty")
        return 0
    print("usage: queue add GPU SCRIPT | queue list")
    return 1


class UnitsError(Exception):
    """A units.jsonl that cannot be trusted (bad line, missing unit, duplicate unit ids)."""


def load_units(p, run, set_, units_dir, agg=None):
    """{unit: row} of one run on one set, from <units_dir>/<name>.jsonl or the run's units.jsonl artifact; None if absent.
    With agg, every row on the set must hold finite numbers in the fields agg reads."""
    if units_dir:
        f = Path(units_dir) / (run["name"] + ".jsonl")
        text = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else None
    else:
        import urllib.parse
        try:
            text = get_bytes(p, "/get-artifact?" + urllib.parse.urlencode({"path": "units.jsonl", "run_uuid": run["id"]})).decode("utf-8", "replace")
        except Unreachable as e:
            if e.code is None:
                raise
            text = None
    if text is None:
        return None
    fields = [] if agg is None else [agg["field"]] if agg["type"] == "mean" else [agg["tp"], agg["fp"], agg["fn"]]
    out, dup, bad = {}, 0, 0
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            raise UnitsError(f"{run['name']}: units.jsonl line {i} is not JSON")
        if not isinstance(r, dict) or not isinstance(r.get("unit"), (str, int, float)):
            raise UnitsError(f"{run['name']}: units.jsonl line {i} has no unit id (string or number)")
        if r.get("set") == set_:
            dup += r["unit"] in out
            bad += not all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in fields)
            out[r["unit"]] = r
    if dup or bad:  # counts only: unit ids can be private
        raise UnitsError(f"{run['name']}: units.jsonl on set {set_}: {dup} duplicate unit ids, "
                         f"{bad} rows without finite numbers in {', '.join(fields) or '-'}")
    return out


def gate_verdict(lo, hi, min_delta, dir_):
    """Gate from the paired delta CI: 통과 (passed), 미달 (failed), 보류 (CI straddles min_delta: not decided)."""
    if dir_ == "min":
        lo, hi = -hi, -lo
    return "통과" if lo > min_delta else "미달" if hi < min_delta else "보류"


T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return T975[min(df, 30) - 1]  # ponytail: df > 30 reuses df=30 (slightly wide); add rows if packs ever run 30+ seeds


def _num(rows, agg):
    if agg["type"] == "mean":
        return [(float(r[agg["field"]]),) for r in rows]
    return [(float(r[agg["tp"]]), float(r[agg["fp"]]), float(r[agg["fn"]])) for r in rows]


def _agg_num(num, idx, agg):
    if agg["type"] == "mean":
        return sum(num[j][0] for j in idx) / len(idx)
    tp, fp, fn = (sum(num[j][c] for j in idx) for c in range(3))
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def paired_seeds(base, arm, agg, B=1000, seed=0):
    """Same-seed pairs: base[i] and arm[i] are {unit: row} of seed i. Delta of the seed mean with a t interval:
    mean +- t(K-1) * sqrt(s2/K + v_units), s2 = variance of the per-seed deltas (seed noise),
    v_units = variance of the mean delta when units are resampled (all pairs see the same resample)."""
    k = len(base)
    ids = sorted(set.intersection(*(set(u) for u in base + arm))) if k else []
    res = {"k": k, "n": len(ids), "base": [], "arm": [], "deltas": [], "mean": None, "lo": None, "hi": None}
    if not ids:
        return res
    nb, na = [_num([u[x] for x in ids], agg) for u in base], [_num([u[x] for x in ids], agg) for u in arm]
    allx = range(len(ids))
    res["base"], res["arm"] = [_agg_num(x, allx, agg) for x in nb], [_agg_num(x, allx, agg) for x in na]
    res["deltas"] = [a - b for a, b in zip(res["arm"], res["base"])]
    res["mean"] = sum(res["deltas"]) / k
    if k < 2:
        return res
    rng, n, means = random.Random(seed), len(ids), []
    for _ in range(B):
        smp = [rng.randrange(n) for _ in range(n)]
        means.append(sum(_agg_num(na[i], smp, agg) - _agg_num(nb[i], smp, agg) for i in range(k)) / k)
    h = t975(k - 1) * math.sqrt(statistics.variance(res["deltas"]) / k + statistics.variance(means))
    res["lo"], res["hi"] = res["mean"] - h, res["mean"] + h
    return res


def done_runs(p):
    """Finished runs only: an aborted or smoke seed never enters a decision."""
    return [r for r in fetch_runs(p) if tag(r, "status", p) == "done"]


def seed_runs(runs, fam):
    """{seed: run} for runs named <fam>_s<seed>; duplicate names (against the contract) resolve to the newest run."""
    out = {}
    for r in runs:
        m = re.fullmatch(re.escape(fam) + r"_s(\d+)", r["name"])
        if m and r["start"] >= out.get(int(m.group(1)), {"start": -1})["start"]:
            out[int(m.group(1))] = r
    return out


def pair_check(p, runs, base, arm, m, units_dir):
    """paired_seeds for one metric plus its gate; a str is an error message."""
    B, A = seed_runs(runs, base), seed_runs(runs, arm)
    seeds = sorted(set(B) & set(A))
    if not seeds:
        return f"no same-seed pairs: need finished runs (status=done) named {base}_s<N> and {arm}_s<N> with the same N"
    set_, bu, au = m["key"].split("/")[0], [], []
    for s in seeds:
        for runs_, out in ((B, bu), (A, au)):
            u = load_units(p, runs_[s], set_, units_dir, m["agg"])
            if u is None:
                return f"{runs_[s]['name']}: no units.jsonl"
            out.append(u)
    st = p["stats"]
    r = paired_seeds(bu, au, m["agg"], st["bootstrap"])
    md = (m.get("gate") or {}).get("min_delta", 0.0)
    why = ("no common units" if not r["n"] else f"{r['k']} pairs < seeds_min {st['seeds_min']}" if r["k"] < st["seeds_min"]
           else f"{r['n']} units < min_units {st['min_units']}" if r["n"] < st["min_units"] else "")
    r.update(seeds=seeds, min_delta=md, why=why, verdict="보류" if why else gate_verdict(r["lo"], r["hi"], md, m["dir"]))
    return r


def _f(x, sign=False):
    return "-" if x is None else f"{x:+.4f}" if sign else f"{x:.4f}"


def print_pairs(m, base, arm, r, B):
    print(f"{m['key']} by seed pair: {base} vs {arm}, {r['k']} pairs, {r['n']} units, {B} unit resamples")
    table(["seed", base, arm, "delta"], [[s, _f(b), _f(a), _f(d, True)] for s, b, a, d in zip(r["seeds"], r["base"], r["arm"], r["deltas"])])
    print(f"mean delta {_f(r['mean'], True)}, 95% CI [{_f(r['lo'], True)}, {_f(r['hi'], True)}] (t, df={r['k'] - 1}; seed + unit variance); "
          f"gate vs min_delta {r['min_delta']} ({m['dir']}): {r['verdict']}" + (f" ({r['why']})" if r["why"] else ""))


def _names(a, valued):
    return [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in valued)]


def cmd_ci(p, a):
    key, ud = _opt(a, "--metric"), _opt(a, "--units")
    m = next((x for x in p["metrics"] if x["key"] == key), None)
    if m is None:
        print(f"unknown metric: {key} (usage: ci A B ... --metric KEY [--units DIR] | ci BASE ARM --metric KEY --pairs)")
        return 1
    names = _names(a, ("--metric", "--units"))
    if "--pairs" in a:
        if len(names) != 2:
            print("usage: ci BASE ARM --metric KEY --pairs [--units DIR]  (BASE, ARM = run families; runs are <family>_s<N>)")
            return 1
        r = pair_check(p, done_runs(p), names[0], names[1], m, ud)
        if isinstance(r, str):
            print(r)
            return 1
        print_pairs(m, names[0], names[1], r, p["stats"]["bootstrap"])
        print(f"to decide and log it over every metric of the stage on the select set: decide {names[0]} {names[1]}")
        return 0
    by = {r["name"]: r for r in fetch_runs(p)}
    if not names or any(n not in by for n in names):
        print("run not found: " + ", ".join(n for n in names if n not in by))
        return 1
    units = []
    for n in names:
        u = load_units(p, by[n], key.split("/")[0], ud, m["agg"])
        if u is None:
            print(f"{n}: no units.jsonl")
            return 1
        units.append(u)
    B = p["stats"]["bootstrap"]
    b = boot(units, m["agg"], B)
    if not b["n"]:
        print("no common units")
        return 1
    print(f"{key}, {b['n']} units, {B} paired resamples; delta = run - {names[0]}")
    table(["run", key, "95% CI", f"delta vs {names[0]}", "delta 95% CI"],
          [[n, f"{v:.4f}", f"[{lo:.4f}, {hi:.4f}]", f"{v - b['value'][0]:+.4f}", f"[{dl:+.4f}, {dh:+.4f}]"]
           for n, v, (lo, hi), (dl, dh) in zip(names, b["value"], b["ci"], b["d_ci"])])
    print("single runs: these CIs cover unit sampling only, not seed noise. Decide with `ci BASE ARM --pairs` or `decide BASE ARM`.")
    return 0


def stage_of(p, exp):
    c = [s for s in p["stages"] if exp.startswith(s["experiment"])]
    return max(c, key=lambda s: len(s["experiment"]))["name"] if c else None


def decisions_path(p):
    return Path(p["_dir"]) / "ledger" / "decisions.jsonl" if p.get("_dir") else None


def read_decisions(p):
    f = decisions_path(p)
    if not (f and f.is_file()):
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def latest_changes(decs, stage=None, dv=None):
    """{arm: (log index of its first change decision, its latest change decision)} (on stage and data_version when given).
    A later decision on the same arm (more seeds) replaces the earlier verdict but keeps the arm's place in the stack."""
    out = {}
    for i, d in enumerate(decs):
        if d.get("kind") == "change" and (stage is None or (d.get("stage") == stage and d.get("data_version") == dv)):
            out[d.get("arm")] = (out.get(d.get("arm"), (i,))[0], d)
    return out


def accepted(decs, stage, dv):
    """[(first log index, decision)] of the arms whose latest change decision on (stage, data_version) is accept, in stack order."""
    return sorted((x for x in latest_changes(decs, stage, dv).values() if x[1].get("verdict") == "accept"), key=lambda x: x[0])


def loo_pending(decs, stage, dv):
    """Accepted changes on (stage, data_version) that no leave-one-out decision (decide ... --loo ARM) has tested."""
    return [d for i, d in accepted(decs, stage, dv) if not any(e.get("kind") == "loo" and e.get("loo_of") == d.get("arm") for e in decs[i + 1:])]


def cmd_decide(p, a):
    valued = ("--set", "--units", "--limit", "--idea", "--note", "--reverify", "--loo")
    names = _names(a, valued)
    if len(names) != 2 or any((_opt(a, f) or "--").startswith("--") for f in valued if f in a) or ("--loo" in a and "--reverify" in a):
        print("usage: decide BASE ARM [--set S] [--units DIR] [--limit L-ID] [--idea IDEA-ID] [--note TEXT]"
              " [--reverify SHORT-ARM | --loo CHANGE-ARM]")
        return 1
    base, arm = names
    set_, ud = _opt(a, "--set") or p["eval_sets"]["select"], _opt(a, "--units")
    runs = done_runs(p)
    A = seed_runs(runs, arm)
    if not A:
        print(f"no finished runs (status=done) named {arm}_s<N>")
        return 1
    stage = stage_of(p, A[min(A)]["exp"])
    ms = [m for m in p["metrics"] if m["stage"] == stage and m["key"].startswith(set_ + "/")]
    if not ms:
        print(f"no {set_}/ metric declared for stage {stage}: add one to project.json metrics")
        return 1
    res = {}
    for m in ms:
        r = pair_check(p, runs, base, arm, m, ud)
        if isinstance(r, str):
            print(r)
            return 1
        print_pairs(m, base, arm, r, p["stats"]["bootstrap"])
        res[m["key"]] = r
    vs = [r["verdict"] for r in res.values()]
    verdict = "reject" if "미달" in vs else "accept" if all(v == "통과" for v in vs) else "undecided"
    short = any(tag(r, "budget", p) == "short" for r in A.values())
    if verdict == "accept" and short:
        verdict = "accept-short"
    loo = _opt(a, "--loo")
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "set": set_, "base": base, "arm": arm,
           "kind": "loo" if loo else "change", "data_version": ",".join(sorted({tag(r, "data_version", p) for r in A.values()} - {""})),
           "budget": "short" if short else "full", "limit": _opt(a, "--limit") or "", "idea": _opt(a, "--idea") or "",
           "note": _opt(a, "--note") or "", "reverifies": _opt(a, "--reverify") or "", "loo_of": loo or "",
           "k": min(r["k"] for r in res.values()), "n": min(r["n"] for r in res.values()),
           "metrics": {k: {f: r[f] for f in ("mean", "lo", "hi", "verdict", "why")} for k, r in res.items()}, "verdict": verdict}
    f = decisions_path(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as h:
        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meaning = {"accept": f"keep the change of {loo}" if loo else f"adopt {arm}" + (" (read_only pack: record it in the ledger only)" if p["server"]["read_only"]
                                                                                  else f": manifest set {arm}_s<N> accepted=yes for each seed run"),
               "accept-short": f"short-budget pass only: rerun {arm} at budget=full as a new family, then decide {base} <new family> --reverify {arm}",
               "reject": f"the stack is worse with the change of {loo}: drop it" if loo else f"do not adopt {arm}",
               "undecided": f"no contribution of {loo} shown: more seeds, or drop it for the simpler stack" if loo
               else "not decided: more seeds, or more units, then decide again"}[verdict]
    print(f"\ndecision: {verdict} -> {meaning}\nlogged: {f}")
    return 0


def cmd_loo(p, a):
    st, v = _opt(a, "--stage"), _opt(a, "--data")
    if not st or v is None:
        print("usage: loo --stage STAGE --data VERSION")
        return 1
    decs = read_decisions(p)
    pend = loo_pending(decs, st, v)
    if not pend:
        print(f"stage {st}, data_version {v}: no accepted change waits for leave-one-out")
        return 0
    head = accepted(decs, st, v)[-1][1]["arm"]
    print(f"stage {st}, data_version {v}: {len(pend)} accepted changes not yet tested by leave-one-out; current stack = {head}")
    for d in pend:
        print(f"- arm: {head} without the change of {d['arm']} (idea {d.get('idea') or '-'}, limit {d.get('limit') or '-'}); "
              f"seeds_min seeds, then: decide <that arm> {head} --loo {d['arm']}")
    print("plan these arms through plan.md and the user's approval; nothing is queued by this command")
    return 0


def cmd_splitcheck(p, a):
    names, ud = _names(a, ("--units",)), _opt(a, "--units")
    if len(names) != 1:
        print("usage: splitcheck RUN [--units DIR]  (unit ids must not repeat across the select, heldout and final sets)")
        return 1
    run = {"name": names[0], "id": ""} if ud else next((r for r in fetch_runs(p) if r["name"] == names[0]), None)
    if run is None:
        print(f"run not found: {names[0]}")
        return 1
    sets = [p["eval_sets"][k] for k in ("select", "heldout", "final") if isinstance(p["eval_sets"].get(k), str)]
    U = {s: load_units(p, run, s, ud) for s in sets}
    if all(u is None for u in U.values()):
        print(f"{names[0]}: no units.jsonl")
        return 1
    have, missing = [s for s in sets if U[s]], [s for s in sets if not U[s]]
    if len(have) < 2:
        print(f"nothing to compare: {names[0]} has units on {', '.join(have) or 'no eval set'}; use a run evaluated on two or more sets")
        return 1
    rows, bad = [], 0
    for i, s1 in enumerate(have):
        for s2 in have[i + 1:]:
            n = len(set(U[s1]) & set(U[s2]))
            bad += n
            rows.append([f"{s1} x {s2}", len(U[s1]), len(U[s2]), n])
    table(["sets", "units A", "units B", "shared unit ids"], rows)
    print(("split ok" if not bad else "split leak: the same unit ids sit in two eval sets")
          + (f" (no units on {', '.join(missing)}: not checked)" if missing else ""))
    return 1 if bad else 0


def baseline_tables(p, runs, version):
    out = []
    for st in p["stages"]:
        for m in (x for x in p["metrics"] if x["stage"] == st["name"]):
            fam = {}
            for r in runs:
                if (tag(r, "data_version", p) == version and tag(r, "status", p) == "done"
                        and r["exp"].startswith(st["experiment"]) and m["key"] in r["metrics"]):
                    fam.setdefault(family(r["name"]), []).append(r["metrics"][m["key"]])
            rows = []
            for f_ in sorted(fam):
                v, mean = fam[f_], sum(fam[f_]) / len(fam[f_])
                gap = "-" if "target" not in m else f"{(m['target'] - mean) if m['dir'] == 'max' else (mean - m['target']):+.4f}"
                rows.append([f_, len(v), f"{mean:.4f} [{min(v):.4f}, {max(v):.4f}]", gap])
            out.append((f"{st['name']} / {m['key']}", ["family", "seeds", "mean [min, max]", "gap to target"], rows))
    return out


def cascade_runs(p, runs, stage, version):
    exps = [s["experiment"] for s in p["stages"] if s["name"] in downstream(p, stage)]
    return [r for r in runs if tag(r, "status", p) == "done" and any(r["exp"].startswith(e) for e in exps)
            and tag(r, "data_version", p) != version]


def audit_issues(p, runs, manifest, snap):
    out, done = [], [r for r in runs if tag(r, "status", p) == "done"]
    if manifest is None:
        out.append("skipped: manifest checks (manifest not read)")
    else:
        for r in done:
            doc = manifest.get(r["name"]) if isinstance(manifest.get(r["name"]), dict) else {}
            for k in ("intro", "result", "review"):
                if not (doc.get(k) or tag(r, k, p)):
                    out.append(f"{r['name']}: missing {k}")
            intro = doc.get("intro") or tag(r, "intro", p)
            if intro and "기대:" not in intro and "Expected:" not in intro:
                out.append(f"{r['name']}: intro has no 기대:/Expected:")
    if snap is None:
        out.append("skipped: server checks (snapshot unavailable)")
    else:
        known = {r["name"] for r in runs}
        for path in sorted(snap.get("done", {})):
            n = run_name(path, p["server"]["run_name_rules"])
            if n and n not in known:
                out.append(f"finished, not synced: {n}")
        if snap.get("waiting"):
            out += [f"GPU idle: gpu {i} util {u}% mem {m} MiB while waiting: {' '.join(snap['waiting'])}"
                    for i, u, m in snap.get("gpus", []) if u < 5 and m < 500]
    for s_, budget in p["eval_sets"].get("look_budget", {}).items():
        looks = {}  # a look = one run family (its seeds together) evaluated on the set, counted per data version
        for r in runs:
            if any(k.startswith(s_ + "/") for k in r["metrics"]):
                looks.setdefault(tag(r, "data_version", p) or "-", set()).add(family(r["name"]))
        out += [f"look_budget exceeded on {s_} (data_version {v}): {len(f_)} run families > {budget}"
                for v, f_ in sorted(looks.items()) if len(f_) > budget]
    decs = read_decisions(p)
    for st, v in sorted({(d.get("stage"), d.get("data_version")) for d in decs}, key=str):
        n = len(loo_pending(decs, st, v))
        if n >= 3:
            out.append(f"stage {st}, data_version {v}: {n} accepted changes not yet tested by leave-one-out (loo --stage {st} --data {v})")
    for arm, (i, d) in latest_changes(decs).items():  # a full-budget recheck closes it whether it passed or failed
        if d.get("verdict") == "accept-short" and not any(e.get("reverifies") == arm and e.get("verdict") in ("accept", "reject") for e in decs[i + 1:]):
            out.append(f"{arm}: short-budget pass not re-verified at full budget "
                       f"(rerun at budget=full as a new family, then decide BASE <family> --reverify {arm})")
    out += [f"frozen: {r['name']} reopen if: {tag(r, 'reopen_if', p)}" for r in runs
            if tag(r, "status", p) == "frozen" and tag(r, "reopen_if", p)]
    return out


def cmd_baseline(p, a):
    v = _opt(a, "--data")
    if not v:
        print("usage: baseline --data VERSION")
        return 1
    for title, head, rows in baseline_tables(p, fetch_runs(p), v):
        print(f"\n## {title}  (data_version={v}, seeds_min={p['stats']['seeds_min']})")
        table(head, rows or [["(no runs)", "", "", ""]])
    return 0


def cmd_cascade(p, a):
    st, v = _opt(a, "--stage"), _opt(a, "--data")
    if not st or not v or st not in [s["name"] for s in p["stages"]]:
        print("usage: cascade --stage STAGE --data VERSION")
        return 1
    rs = cascade_runs(p, fetch_runs(p), st, v)
    print(f"downstream of {st}: {' '.join(downstream(p, st)) or '-'}; done runs not on {v}: {len(rs)}")
    table(["experiment", "run", "data_version"], sorted([r["exp"], r["name"], tag(r, "data_version", p) or "-"] for r in rs))
    return 0


def cmd_audit(p, a):
    man = remote_manifest(p)
    snap = snapshot(p)
    iss = audit_issues(p, fetch_runs(p), man, snap)
    if man is None:
        print("manifest checks skipped")
    print("\n".join(iss) or "audit clean")
    return 1 if any(not i.startswith(("skipped:", "frozen:")) for i in iss) else 0


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
    assert EX["server"]["run_name_rules"] == [["(?:.*/)?([^/]+)/DONE$", "\\1"]]
    for shape in ({"server": "x"}, {"stages": ["a"]}, {"metrics": ["a"], "stages": [{"name": "s", "depends_on": "s"}]}, ["x"]):
        assert check_pack(load_pack_dict({**EXRAW, **shape} if isinstance(shape, dict) else shape))
    S = EXRAW["server"]
    for bad_s, word in (({**S, "root": 5}, "server.root"), ({k: v for k, v in S.items() if k != "venv"}, "server.venv"),
                        ({**S, "mlflow_port": "5002"}, "server.mlflow_port"), ({**S, "gpus": 0}, "server.gpus"),
                        ({**S, "gpus": True}, "server.gpus"), ({**S, "host": "-oProxyCommand=x", "allow_hosts": ["-oProxyCommand=x"]}, "server.host")):
        assert any(word in e for e in check_pack(load_pack_dict({**EXRAW, "server": bad_s}))), word
    assert check_pack(load_pack_dict({**EXRAW, "server": {**{k: v for k, v in S.items() if k != "venv"}, "read_only": True}})) == []
    assert "stats: must be an object" in check_pack(load_pack_dict({**EXRAW, "stats": 3}))
    dep = copy.deepcopy(EXRAW); dep["stages"][1]["depends_on"] = "embed"
    assert any("depends_on must be a list" in e for e in check_pack(load_pack_dict(dep)))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lf = Path(td) / "project.json"; lf.write_text("[1]", encoding="utf-8")
        assert read_pack(lf)[0] is None and read_pack(lf)[1]
    assert family("alpha_m1_s12") == "alpha_m1" and family("beta") == "beta"
    U = [{k: {"tp": a, "fp": b, "fn": c} for k, (a, b, c) in d.items()} for d in PARITY_CALLS]
    b = boot(U, {"type": "f1", "tp": "tp", "fp": "fp", "fn": "fn"})
    r10 = lambda xs: [tuple(round(x, 10) for x in c) for c in xs]
    N_LEGACY = 35
    F1_LEGACY = [0.6666666667, 0.7906976744]
    CI_LEGACY = [(0.5977011494, 0.7154471545), (0.7191011236, 0.852173913)]
    DCI_LEGACY = [(0.0, 0.0), (0.0213903743, 0.2328173375)]
    assert b["n"] == N_LEGACY and [round(x, 10) for x in b["value"]] == F1_LEGACY
    assert r10(b["ci"]) == CI_LEGACY and r10(b["d_ci"]) == DCI_LEGACY
    assert boot([{"a": {"x": 1}}, {"b": {"x": 1}}], {"type": "mean", "field": "x"})["n"] == 0
    assert agg_value([{"x": 1}, {"x": 3}], {"type": "mean", "field": "x"}) == 2.0
    dag = load_pack_dict({**EXRAW, "stages": [{"name": "a", "experiment": "1", "depends_on": []},
                                              {"name": "b", "experiment": "2", "depends_on": ["a"]},
                                              {"name": "c", "experiment": "3", "depends_on": ["b"]}], "metrics": []})
    assert downstream(dag, "a") == ["b", "c"] and downstream(dag, "c") == []
    ooo = load_pack_dict({**EXRAW, "stages": [{"name": "c", "experiment": "3", "depends_on": ["b"]},
                                              {"name": "a", "experiment": "1", "depends_on": []},
                                              {"name": "b", "experiment": "2", "depends_on": ["a"]}], "metrics": []})
    assert downstream(ooo, "a") == ["b", "c"]
    import contextlib, io, urllib.error
    RUN = lambda i, n, t=(), m=(), pa=(): {"info": {"run_id": i, "run_name": n, "status": "FINISHED", "start_time": "5"},
        "data": {"metrics": [{"key": k, "value": v, "timestamp": 1, "step": 0} for k, v in m], "params": [{"key": k, "value": v} for k, v in pa],
                 "tags": [{"key": k, "value": v} for k, v in t]}}
    DB = {"1": [[RUN("a", "r1_s0", [("gate_heldout", "pass"), ("compare", "vs r2 and r1_s1"), ("status", "done")],
                     [("heldout/d_score", 0.5), ("heldout/ci_lo_d_score", 0.25), ("heldout/ci_hi_d_score", 0.75)], [("lr", "1"), ("train", "9")])],
                [RUN("b", "r1_s1", m=[("heldout/d_score", 0.4)], pa=[("lr", "2"), ("train", "8")])]],
          "2": [[RUN("c", "r2", m=[("heldout/d_score", 0.3)])]]}

    class Resp:
        def __init__(s, b): s.b = b
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s): return s.b

    def fx(req, timeout=None):
        u, body = req.full_url, json.loads(req.data) if req.data else {}
        assert u.startswith("http://127.0.0.1:5002/")
        if u.endswith("experiments/search"):
            return Resp(json.dumps({"experiments": [{"experiment_id": "1", "name": "exp_one"}, {"experiment_id": "2", "name": "exp_two"}]}).encode())
        if u.endswith("runs/search"):
            pages = DB[body["experiment_ids"][0]]
            i = int(body.get("page_token", 0))
            return Resp(json.dumps({"runs": pages[i], **({"next_page_token": str(i + 1)} if i + 1 < len(pages) else {})}).encode())
        if "get-history" in u:
            return Resp(json.dumps({"metrics": [{"key": "k", "value": 0.5, "timestamp": 1, "step": 3}]}).encode())
        return Resp(b"raw text")

    _set_open(fx)
    runs = fetch_runs(EX)
    assert {r["name"] for r in runs} == {"r1_s0", "r1_s1", "r2"} and runs[0]["start"] == 5 and runs[0]["exp"] == "exp_one"
    head, rows = compare_table(runs, "heldout", EX)
    assert head[:3] == ["run", "experiment", "gate_heldout"] and any("[" in c for r in rows for c in r)
    j = head.index("d_score")
    assert rows[0][j] == "0.5000 [0.2500, 0.7500]" and rows[1][j] == "0.4000" and rows[0][2] == "pass" and rows[1][2] == "-"
    assert history(EX, "a", "k") == [(3, 0.5)] and get_bytes(EX, "/get-artifact?run_id=a&path=x") == b"raw text"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cmd_report(EX, ["--run", "r1_s0"]) == 0 and cmd_compare(EX, ["r1_s1", "r2", "--set", "heldout"]) == 0
        assert cmd_compare(EX, ["r1_s1"]) == 0  # default set = eval_sets.select, never heldout
        assert cmd_diff({**EX, "derived_params": ["train"]}, ["r1_s0", "--vs", "r1_s1"]) == 0 and cmd_list(EX, ["--exp", "exp_one"]) == 0
        assert cmd_diff(EX, ["zz", "--vs", "r2"]) == 1
    o = out.getvalue()
    assert "compare on heldout" in o and "| r2 |" in o and "| lr | 1 | 2 |" in o and "| train |" not in o and "run not found: zz" in o
    assert "| run | experiment | gate_dev |" in o
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:  # leakcheck sees a term that only lives in an old commit
        g = lambda *c: subprocess.run(["git", "-C", td, "-c", "user.name=t", "-c", "user.email=t@t", *c], capture_output=True, check=True)
        g("init", "-q"); (Path(td) / "a.txt").write_text("Secret-Host"); g("add", "a.txt"); g("commit", "-qm", "a")
        (Path(td) / "a.txt").write_text("clean"); g("commit", "-qam", "b")
        assert history_hits(td, ["secret-host", "nothing"]) == [("secret-host", 1)]
        with contextlib.redirect_stdout(io.StringIO()) as lo:
            assert cmd_leakcheck(EX, [td]) == 1 and cmd_leakcheck(EX, [td, "--no-history"]) == 0
        assert "history: secret-host in 1 commits" in lo.getvalue()
    assert names_in("run_q2 대비", ["run_q", "run_q2"]) == ["run_q2"] and names_in("nb1과 같음", ["nb1"]) == ["nb1"]
    assert param_diff({"lr": "1", "train": "9"}, {"lr": "2", "train": "8"}, {"train"}) == [("lr", "1", "2")]
    def down(*_a, **_k): raise urllib.error.URLError("x")
    _set_open(down)
    try: fetch_runs(EX); assert False
    except Unreachable as e: assert "ssh -N -L 5002:127.0.0.1:5002 my-gpu" in str(e)
    try: get_bytes(EX, "/x"); assert False
    except Unreachable: pass
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / ".phro").mkdir(); (Path(td) / ".phro/project.json").write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
        old, os.environ["CLAUDE_PROJECT_DIR"] = os.environ.get("CLAUDE_PROJECT_DIR"), td
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["list"])
        if old is None: del os.environ["CLAUDE_PROJECT_DIR"]
        else: os.environ["CLAUDE_PROJECT_DIR"] = old
        assert code == 1 and "ssh -N -L 5002:127.0.0.1:5002 my-gpu" in out.getvalue()
    _set_open(fx)
    DONE = lambda r, **t: {**r, "tags": {"status": "done", "data_version": "v1", **t}}
    base = [DONE(r) for r in fetch_runs(EX)]
    for r in base:
        r["exp"] = "2_classify"; r["metrics"] = {"heldout/f1": 0.5 + 0.1 * len(r["name"])}
    t = baseline_tables(EX, base, "v1")
    fam = {row[0]: row for _, _, rows in t for row in rows}
    assert fam["r1"][1] == 2 and fam["r2"][1] == 1 and fam["r1"][3].startswith("-")
    dag_runs = [{**r, "exp": "2/x" if r["name"] == "r2" else "1/x"} for r in base]
    assert [r["name"] for r in cascade_runs(dag, dag_runs, "a", "v2")] == ["r2"]
    iss = audit_issues(EX, base, {"r1_s0": {"intro": "x"}}, {"gpus": [[0, 0, 0]], "waiting": ["run_x.sh"], "done": {"runs/s/zz/DONE": 1}})
    assert any("기대" in i or "Expected" in i for i in iss) and any("not synced" in i and "zz" in i for i in iss) and any("idle" in i for i in iss)
    over =[{**r, "metrics": {"final/f1": 1.0}} for r in base]
    assert any("look_budget" in i and "final" in i for i in audit_issues(EX, over, {}, None))
    frozen = [{**base[0], "tags": {"status": "frozen", "reopen_if": "later"}}]
    assert any("reopen if: later" in i for i in audit_issues(EX, frozen, {}, None))
    old_fns = {k: globals()[k] for k in ("remote_manifest", "snapshot", "fetch_runs")}
    globals().update(remote_manifest=lambda q: {}, snapshot=lambda q: {"gpus": [], "waiting": [], "done": {}}, fetch_runs=lambda q: frozen)
    with contextlib.redirect_stdout(io.StringIO()):
        assert cmd_audit(EX, []) == 0  # a frozen line alone is information, not a failure
    globals().update(old_fns)
    assert gate_verdict(0.01, 0.05, 0.0, "max") == "통과" and gate_verdict(-0.05, -0.01, 0.0, "max") == "미달"
    assert gate_verdict(-0.01, 0.05, 0.0, "max") == "보류" and gate_verdict(0.01, 0.05, 0.02, "max") == "보류"
    assert gate_verdict(-0.05, -0.01, 0.0, "min") == "통과" and gate_verdict(0.01, 0.05, 0.0, "min") == "미달"
    with tempfile.TemporaryDirectory() as td:
        ud = Path(td)
        assert load_units(EX, {"name": "nope", "id": "x"}, "heldout", ud) is None
        for n, k in (("r1_s0", 1), ("r1_s1", 2)):
            (ud / (n + ".jsonl")).write_text("".join(json.dumps({"set": "heldout", "unit": f"c{i}", "tp": k * (i % 3), "fp": i % 2, "fn": 1}) + "\n" for i in range(12)) + json.dumps({"set": "other", "unit": "z", "tp": 1, "fp": 1, "fn": 1}) + "\n", encoding="utf-8")
        assert len(load_units(EX, {"name": "r1_s0", "id": "a"}, "heldout", ud)) == 12
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cmd_ci(EX, ["r1_s0", "r1_s1", "--metric", "heldout/f1", "--units", str(ud)]) == 0
            assert cmd_ci(EX, ["r1_s0", "r2", "--metric", "heldout/f1", "--units", str(ud)]) == 1
            assert cmd_ci(EX, ["r1_s0", "--metric", "nope"]) == 1
        o = out.getvalue()
        assert "heldout/f1, 12 units, 1000 paired resamples; delta = run - r1_s0" in o and "| run | heldout/f1 | 95% CI | delta vs r1_s0 | delta 95% CI |" in o
        assert "| gate |" not in o and "not seed noise" in o and "--pairs" in o  # single runs never print a verdict
        assert "r2: no units.jsonl" in o and "unknown metric: nope" in o
        (ud / "r1_s0.jsonl").write_text("".join(json.dumps({"set": "heldout", "unit": "c1", "tp": 1, "fp": 0, "fn": 0}) + "\n" for _ in range(2)), encoding="utf-8")
        try: load_units(EX, {"name": "r1_s0"}, "heldout", ud); assert False
        except UnitsError as e: assert "1 duplicate unit ids" in str(e) and "c1" not in str(e)
        (ud / "r1_s0.jsonl").write_text("{bad\n", encoding="utf-8")
        try: load_units(EX, {"name": "r1_s0"}, "heldout", ud); assert False
        except UnitsError as e: assert "line 1 is not JSON" in str(e)
    def http404(req, timeout=None): raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
    _set_open(http404)
    assert load_units(EX, {"name": "n", "id": "x"}, "heldout", None) is None
    _set_open(down)
    try: load_units(EX, {"name": "n", "id": "x"}, "heldout", None); assert False
    except Unreachable as e: assert e.code is None
    _set_open(None)
    R = [["runs/([^/]+)/eval_(.+)_final\\.json$", "\\1_\\2"], [".*/([^/]+)/final$", "\\1"], [".*_DONE$", ""]]
    assert run_name("runs/alpha/eval_m1_s2_final.json", R) == "alpha_m1_s2"
    assert run_name("proj/runs/alpha/eval_m1_s2_final.json", R) == "alpha_m1_s2"  # the match only, no glued prefix
    assert run_name("runs/stage2/m/beta/final", R) == "beta" and run_name("runs/alpha/ALPHA_DONE", R) is None
    assert run_name("runs/stage/myrun/DONE", EX["server"]["run_name_rules"]) == "myrun" and run_name("zz", R) is None
    compile(remote_config(EX) + "\n" + REMOTE, "remote", "exec")
    with tempfile.TemporaryDirectory() as td:
        rt = Path(td)
        (rt / "runs/r1").mkdir(parents=True); (rt / "runs/r1/DONE").write_text("")
        (rt / "runs.json").write_text(json.dumps({"r1": {"intro": "i" * 1000, "result": 5}, "bad": 3}), encoding="utf-8")
        (rt / "gpu_queue.txt").write_text("# c\njob a\n\njob b\n", encoding="utf-8")
        lp = {"server": {**EX["server"], "root": str(rt), "manifest": str(rt / "runs.json"), "queue": str(rt / "gpu_queue.txt")}}
        run = lambda q: json.loads(subprocess.run([sys.executable, "-"], input=remote_config(q) + "\n" + REMOTE, capture_output=True, text=True, encoding="utf-8").stdout)
        d = run(lp)
        assert list(d["done"]) == ["runs/r1/DONE"] and d["queue"] == ["job a", "job b"] and isinstance(d["gpus"], list) and isinstance(d["waiting"], list)
        assert len(d["docs"]["r1"]["intro"]) == 900 and d["docs"]["r1"]["result"] == "5" and "bad" not in d["docs"]
        d = run({"server": {**lp["server"], "manifest": str(rt / "none"), "queue": str(rt / "none")}})
        assert d["docs"] == {} and d["queue"] == []
    snap = {"gpus": [[0, 5, 100], [1, 0, 0]], "procs": {"0": ["beta"]}, "done": {}, "docs": {}, "waiting": ["run_a.sh"], "queue": ["job a"]}
    sn = [snap]
    old_snap, globals()["snapshot"] = snapshot, lambda q: sn[0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cmd_status(EX, []) == 0
        sn[0] = None
        assert cmd_status(EX, []) == 1
    globals()["snapshot"] = old_snap
    o = out.getvalue()
    assert "| 0 | 100 MiB | 5% | beta |" in o and "| 1 | 0 MiB | 0% | FREE |" in o and "job a" in o and "run_a.sh" in o and "ssh to my-gpu failed" in o
    RO = load_pack_dict({**EXRAW, "server": {**EXRAW["server"], "read_only": True}})
    calls = []
    old_ssh, old_scp = globals()["ssh"], globals()["scp"]
    globals()["ssh"] = lambda q, c, *a, **k: calls.append(("ssh", c)) or subprocess.CompletedProcess([], 0, "", "")
    globals()["scp"] = lambda q, fs, d: calls.append(("scp", d)) or subprocess.CompletedProcess([], 0, "", "")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cmd_setup(RO, None) == 1 and calls == [] and cmd_manifest(RO, ["set", "r1", "a=b"]) == 1 and calls == []
        assert cmd_queue(RO, ["add", "1", "run_a.sh"]) == 1 and calls == []  # read_only: refused before any ssh
        assert cmd_manifest(EX, ["set", "r1", "intro=x y; z"]) == 0
        assert cmd_queue(EX, ["add", "1", "run a.sh"]) == 0 and cmd_queue(EX, ["list"]) == 0 and cmd_queue(EX, ["x"]) == 1
    assert "server.read_only: refusing to write to my-gpu" in out.getvalue()
    assert calls == [("ssh", "python3 ~/phro/proj/phro_server.py manifest set r1 'intro=x y; z'"),
                     ("ssh", "python3 ~/phro/proj/phro_server.py queue add 1 'run a.sh'"),
                     ("ssh", "cat ~/phro/proj/gpu_queue.txt 2>/dev/null; true")]
    assert rpath("~/a b/c") == "~/'a b/c'" and rpath("/x y") == "'/x y'"
    del calls[:]
    with contextlib.redirect_stdout(io.StringIO()):
        assert cmd_setup(EX, None) == 0
    assert calls == [("ssh", "mkdir -p ~/phro/proj"), ("scp", "~/phro/proj"), ("ssh", "python3 ~/phro/proj/phro_server.py setup")]
    seen = []
    globals()["ssh"] = lambda q, c, *a, **k: seen.append(c) or subprocess.CompletedProcess([], 0, "{}", "")
    remote_manifest(EX)
    assert seen == ["cat ~/phro/proj/runs.json"]
    globals()["ssh"] = lambda q, c, *a, **k: subprocess.CompletedProcess([], 0, '{"r1": {"intro": "i"}}', "")
    assert remote_manifest(EX) == {"r1": {"intro": "i"}}
    globals()["ssh"] = lambda q, c, *a, **k: subprocess.CompletedProcess([], 1, "", "")
    assert remote_manifest(EX) is None
    globals()["ssh"], globals()["scp"] = old_ssh, old_scp
    # seed pairs, decision log, leave-one-out, budget, split check
    assert _real("${CLAUDE_PROJECT_DIR}") is None and _real("") is None and _real("x") == "x"
    assert t975(1) == 12.706 and t975(3) == 3.182 and t975(99) == 2.042
    for bad_st, word in (({"seeds_min": 1}, "stats.seeds_min"), ({"bootstrap": 0}, "stats.bootstrap"), ({"min_units": 0}, "stats.min_units")):
        assert any(word in e for e in check_pack(load_pack_dict({**EXRAW, "stats": {**EXRAW["stats"], **bad_st}}))), word
    assert any("eval_sets.select" in e for e in check_pack(load_pack_dict({**EXRAW, "eval_sets": {"heldout": "heldout"}})))
    MEAN = {"type": "mean", "field": "c"}

    def draw(p_hit, seed, n=120):
        g = random.Random(seed)
        return {f"u{i}": {"set": "dev", "unit": f"u{i}", "c": 1.0 if g.random() < p_hit else 0.0} for i in range(n)}

    better = paired_seeds([draw(0.5, s) for s in range(4)], [draw(0.75, 10 + s) for s in range(4)], MEAN, 300)
    same = paired_seeds([draw(0.5, s) for s in range(4)], [draw(0.5, 10 + s) for s in range(4)], MEAN, 300)
    assert better["k"] == 4 and better["n"] == 120 and gate_verdict(better["lo"], better["hi"], 0.0, "max") == "통과"
    assert gate_verdict(same["lo"], same["hi"], 0.0, "max") == "보류" and same["lo"] < same["mean"] < same["hi"]
    one = paired_seeds([draw(0.5, 0)], [draw(0.75, 1)], MEAN, 50)
    assert one["k"] == 1 and one["mean"] is not None and one["lo"] is None
    with tempfile.TemporaryDirectory() as td:
        ud = Path(td) / "units"
        ud.mkdir()
        TP = load_pack_dict({**EXRAW, "stats": {"seeds_min": 4, "bootstrap": 200, "min_units": 50}})
        TP["_dir"], TP["metrics"] = Path(td) / ".phro", [{"key": "dev/acc", "stage": "classify", "dir": "max", "agg": MEAN, "gate": {"min_delta": 0.0}}]
        FR = []

        def fam(name, p_hit, seed0, **t):
            for s in range(4):
                n = f"{name}_s{s}"
                FR.append({"exp": "2_classify", "id": n, "name": n, "start": 0, "end": 1, "params": {}, "metrics": {},
                           "tags": {"status": "done", "data_version": "v1", **t}})
                (ud / (n + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in draw(p_hit, seed0 + s).values()), encoding="utf-8")

        fam("base", 0.5, 0); fam("arm", 0.75, 10); fam("arm2", 0.75, 20)
        fam("shortarm", 0.75, 40, budget="short"); fam("shortarm_full", 0.75, 40, budget="full"); fam("same", 0.5, 50)
        fam("short2", 0.75, 60, budget="short"); fam("short2_full", 0.25, 70); fam("abort", 0.75, 80)
        FR[-1]["tags"]["status"] = "aborted"  # abort_s3: never paired
        assert seed_runs([{"name": "f_s0", "start": 2, "id": "new"}, {"name": "f_s0", "start": 1, "id": "old"}], "f")[0]["id"] == "new"
        assert set(seed_runs(FR, "shortarm")) == {0, 1, 2, 3}  # shortarm_full_s<N> is another family
        old_fr, globals()["fetch_runs"] = globals()["fetch_runs"], lambda q: FR
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cmd_ci(TP, ["base", "arm", "--metric", "dev/acc", "--pairs", "--units", str(ud)]) == 0
            assert cmd_ci(TP, ["base", "--metric", "dev/acc", "--pairs"]) == 1
            assert cmd_decide(TP, ["base", "arm", "--units", str(ud), "--idea", "IDEA-001", "--limit", "L-001"]) == 0
            assert cmd_decide(TP, ["base", "same", "--units", str(ud)]) == 0
            assert cmd_decide(TP, ["base", "shortarm", "--units", str(ud)]) == 0
            assert cmd_decide(TP, ["base", "nope", "--units", str(ud)]) == 1
            assert cmd_decide(TP, ["base", "arm", "--set", "heldout", "--units", str(ud)]) == 1  # no heldout/ metric declared
        o = out.getvalue()
        decs = read_decisions(TP)
        assert [d["verdict"] for d in decs] == ["accept", "undecided", "accept-short"], [d["verdict"] for d in decs]
        assert decs[0]["k"] == 4 and decs[0]["data_version"] == "v1" and decs[0]["idea"] == "IDEA-001" and decs[0]["limit"] == "L-001"
        assert decs[0]["metrics"]["dev/acc"]["verdict"] == "통과" and decs[2]["budget"] == "short"
        assert "dev/acc by seed pair: base vs arm, 4 pairs, 120 units" in o and "decision: accept -> adopt arm" in o
        assert "no finished runs (status=done) named nope_s<N>" in o and "no heldout/ metric declared for stage classify" in o
        iss = audit_issues(TP, FR, {}, None)
        assert any("shortarm: short-budget pass not re-verified" in i for i in iss) and not any("leave-one-out" in i for i in iss)
        with contextlib.redirect_stdout(io.StringIO()):
            assert cmd_decide(TP, ["base", "shortarm_full", "--units", str(ud), "--reverify", "shortarm"]) == 0
            assert cmd_decide(TP, ["base", "arm2", "--units", str(ud)]) == 0
            assert cmd_decide(TP, ["base", "arm", "--units", str(ud)]) == 0  # more seeds later: replaces the verdict, keeps the stack order
            assert cmd_decide(TP, ["base", "short2", "--units", str(ud)]) == 0
            assert any("short2: short-budget" in i for i in audit_issues(TP, FR, {}, None))
            assert cmd_decide(TP, ["base", "short2_full", "--units", str(ud), "--reverify", "short2"]) == 0
            assert cmd_decide(TP, ["base", "abort", "--units", str(ud)]) == 0
        decs = read_decisions(TP)
        rv = [d for d in decs if d["reverifies"]]
        assert [d["verdict"] for d in rv] == ["accept", "reject"] and decs[-1]["k"] == 3 and decs[-1]["verdict"] == "undecided"
        iss = audit_issues(TP, FR, {}, None)
        assert not any("short-budget" in i for i in iss)  # a failed recheck closes it too
        assert any("3 accepted changes not yet tested by leave-one-out" in i for i in iss)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cmd_loo(TP, ["--stage", "classify", "--data", "v1"]) == 0
            assert cmd_decide(TP, ["same", "arm2", "--units", str(ud), "--loo"]) == 1  # --loo names the change left out
            assert cmd_decide(TP, ["same", "arm2", "--loo", "arm", "--reverify", "x"]) == 1
            for ch in ("arm", "shortarm_full"):
                assert cmd_decide(TP, ["same", "arm2", "--units", str(ud), "--loo", ch]) == 0
            assert cmd_loo(TP, ["--stage", "classify", "--data", "v1"]) == 0
            assert cmd_decide(TP, ["same", "arm2", "--units", str(ud), "--loo", "arm2"]) == 0
            assert cmd_loo(TP, ["--stage", "classify", "--data", "v1"]) == 0 and cmd_loo(TP, []) == 1
        o = out.getvalue()
        assert "3 accepted changes not yet tested by leave-one-out; current stack = arm2" in o and "--loo shortarm_full" in o
        assert "1 accepted changes not yet tested by leave-one-out; current stack = arm2" in o  # a partial leave-one-out leaves the rest
        assert "no accepted change waits for leave-one-out" in o and "decision: accept -> keep the change of arm" in o
        assert not any("leave-one-out" in i for i in audit_issues(TP, FR, {}, None))
        globals()["fetch_runs"] = old_fr
        sp = ud / "sp.jsonl"
        sp.write_text("".join(json.dumps({"set": s, "unit": u, "c": 1}) + "\n" for s, u in (("dev", "a"), ("dev", "b"), ("heldout", "c"), ("final", "d"))), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as so:
            assert cmd_splitcheck(TP, ["sp", "--units", str(ud)]) == 0
            sp.write_text(sp.read_text(encoding="utf-8") + json.dumps({"set": "heldout", "unit": "a", "c": 1}) + "\n", encoding="utf-8")
            assert cmd_splitcheck(TP, ["sp", "--units", str(ud)]) == 1
            (ud / "one.jsonl").write_text(json.dumps({"set": "dev", "unit": "a", "c": 1}) + "\n", encoding="utf-8")
            assert cmd_splitcheck(TP, ["one", "--units", str(ud)]) == 1 and cmd_splitcheck(TP, ["none", "--units", str(ud)]) == 1
        o = so.getvalue()
        assert "split ok" in o and "split leak" in o and "nothing to compare" in o and "none: no units.jsonl" in o
        for rows, word in (([{"set": "dev", "unit": "a"}], "1 rows without finite numbers in c"), ([{"set": "dev", "unit": None, "c": 1}], "no unit id"),
                           ([{"set": "dev", "unit": "a", "c": 1}] * 2, "1 duplicate unit ids"), ([{"set": "dev", "unit": ["a"], "c": 1}], "no unit id")):
            (ud / "bad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            try:
                load_units(TP, {"name": "bad"}, "dev", ud, MEAN)
                raise AssertionError(word)
            except UnitsError as e:
                assert word in str(e), (word, str(e))
    print("phro selftest ok")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: phro.py selftest | check-pack | leakcheck [dir] [--no-history] | list | report | compare | diff | status | ci | decide | loo"
              " | splitcheck | baseline | cascade | audit | setup | manifest | queue")
        return 0
    cmd, a = argv[0], argv[1:]
    if cmd == "selftest":
        selftest()
        return 0
    if cmd == "check-pack":
        return cmd_check_pack(None, a)
    f = find_pack()
    if f is None:
        print(f"no .phro/project.json found from {os.environ.get('CLAUDE_PROJECT_DIR') or Path.cwd()}")
        return 1
    p, errs = read_pack(f)
    if errs:
        print("\n".join(errs))
        return 1
    fn = globals().get("cmd_" + cmd.replace("-", "_"))
    if fn is None:
        print(f"unknown command: {cmd}")
        return 1
    try:
        return fn(p, a)
    except (Unreachable, UnitsError) as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
