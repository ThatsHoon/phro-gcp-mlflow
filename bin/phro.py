#!/usr/bin/env python3
"""phro: PC-side CLI and library for AutoML loops on one GCP GPU VM with MLflow."""
import copy
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY_CALLS = [{f"c{i}": (i % 3, i % 2, (i + 1) % 2) for i in range(40)},
                {f"c{i}": ((i + 1) % 3, 0, i % 2) for i in range(5, 45)}]
REQUIRED = ["name", "goal", "stages", "metrics", "eval_sets", "server", "egress"]


def find_pack(start=None):
    d = Path(start or os.environ.get("CLAUDE_PROJECT_DIR") or Path.cwd()).resolve()
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


_open = urllib.request.urlopen


def _set_open(fn):
    global _open
    _open = fn or urllib.request.urlopen


def _call(p, path, data=None):
    port = p["server"]["mlflow_port"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with _open(req, timeout=30) as r:
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
    sweep, set_ = _opt(a, "--sweep"), _opt(a, "--set") or p["eval_sets"].get("heldout", "test")
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


def load_units(p, run, set_, units_dir):
    """{unit: row} of one run on one set, from <units_dir>/<name>.jsonl or the run's units.jsonl artifact; None if absent."""
    if units_dir:
        f = Path(units_dir) / (run["name"] + ".jsonl")
        text = f.read_text(encoding="utf-8") if f.is_file() else None
    else:
        try:
            text = get_bytes(p, "/get-artifact?" + urllib.parse.urlencode({"path": "units.jsonl", "run_uuid": run["id"]})).decode("utf-8")
        except Unreachable as e:
            if e.code is None:
                raise
            text = None
    if text is None:
        return None
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return {r["unit"]: r for r in rows if r.get("set") == set_}


def gate_verdict(lo, hi, min_delta, dir_):
    """Gate from the paired delta CI: 통과 (passed), 미달 (failed), 보류 (CI straddles min_delta: not decided)."""
    if dir_ == "min":
        lo, hi = -hi, -lo
    return "통과" if lo > min_delta else "미달" if hi < min_delta else "보류"


def cmd_ci(p, a):
    key, ud = _opt(a, "--metric"), _opt(a, "--units")
    m = next((x for x in p["metrics"] if x["key"] == key), None)
    if m is None:
        print(f"unknown metric: {key}")
        return 1
    names = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--metric", "--units"))]
    by = {r["name"]: r for r in fetch_runs(p)}
    if not names or any(n not in by for n in names):
        print("run not found: " + ", ".join(n for n in names if n not in by))
        return 1
    units = []
    for n in names:
        u = load_units(p, by[n], key.split("/")[0], ud)
        if u is None:
            print(f"{n}: no units.jsonl")
            return 1
        units.append(u)
    B = p["stats"]["bootstrap"]
    b = boot(units, m["agg"], B)
    if not b["n"]:
        print("no common units")
        return 1
    md = (m.get("gate") or {}).get("min_delta", 0.0)
    print(f"{key}, {b['n']} units, {B} paired resamples; delta = run - {names[0]}; gate: delta CI vs min_delta {md} ({m['dir']})")
    table(["run", key, "95% CI", f"delta vs {names[0]}", "delta 95% CI", "gate"],
          [[n, f"{v:.4f}", f"[{lo:.4f}, {hi:.4f}]", f"{v - b['value'][0]:+.4f}", f"[{dl:+.4f}, {dh:+.4f}]",
            "-" if i == 0 else gate_verdict(dl, dh, md, m["dir"])]
           for i, (n, v, (lo, hi), (dl, dh)) in enumerate(zip(names, b["value"], b["ci"], b["d_ci"]))])
    print(f"record it with: manifest set <run> gate_{key.split('/')[0]}=<verdict>")
    return 0


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
        n = sum(any(k.startswith(s_ + "/") for k in r["metrics"]) for r in runs)
        if n > budget:
            out.append(f"look_budget exceeded on {s_}: {n} runs > {budget}")
    for v in sorted({tag(r, "data_version", p) for r in runs} - {""}):
        mine = [r for r in runs if tag(r, "data_version", p) == v]
        acc = sum(tag(r, "accepted", p) == "yes" for r in mine)
        if acc >= 3 and not any(s_.startswith("loo_") for r in mine for s_ in tag(r, "sweep", p).split(",")):
            out.append(f"data_version {v}: {acc} accepted runs and no leave-one-out sweep")
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
        assert cmd_diff({**EX, "derived_params": ["train"]}, ["r1_s0", "--vs", "r1_s1"]) == 0 and cmd_list(EX, ["--exp", "exp_one"]) == 0
        assert cmd_diff(EX, ["zz", "--vs", "r2"]) == 1
    o = out.getvalue()
    assert "compare on heldout" in o and "| r2 |" in o and "| lr | 1 | 2 |" in o and "| train |" not in o and "run not found: zz" in o
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
    three_accepted = [DONE(r, accepted="yes") for r in base]
    assert any("leave-one-out" in i for i in audit_issues(EX, three_accepted, {}, None))
    assert not any("leave-one-out" in i for i in audit_issues(EX, [DONE(r, accepted="yes", sweep="loo_a") for r in base], {}, None))
    over = [{**r, "metrics": {"final/f1": 1.0}} for r in base]
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
        assert "heldout/f1, 12 units, 1000 paired resamples; delta = run - r1_s0" in o and "| run | heldout/f1 | 95% CI | delta vs r1_s0 | delta 95% CI | gate |" in o
        assert "manifest set <run> gate_heldout=<verdict>" in o and " | - |" in o
        assert "r2: no units.jsonl" in o and "unknown metric: nope" in o
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
    print("phro selftest ok")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: phro.py selftest | check-pack | leakcheck [dir] [--no-history] | list | report | compare | diff | status | ci | baseline | cascade | audit | setup | manifest | queue")
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
    except Unreachable as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
