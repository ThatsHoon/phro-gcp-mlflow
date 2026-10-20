#!/usr/bin/env python3
"""phro VM sync: finished run folders -> MLflow. Run inside the VM venv; only main imports mlflow."""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _lines(f):
    try:
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def parse_run_dir(d):
    d = Path(d)
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    rows = _lines(d / "metrics.jsonl")
    last = max([r["step"] for r in rows if isinstance(r.get("step"), int)], default=0)
    rows = [r for r in rows if r["step"] != "final"] + [r for r in rows if r["step"] == "final"]  # final last: it gets the newest timestamp
    metrics = [(f"{r['set']}/{r['key']}" if r.get("set") else r["key"], r["value"], last if r["step"] == "final" else r["step"])
               for r in rows]
    if meta.get("name") and meta["name"] != d.name:
        print(f"warning {d}: run.json name {meta['name']!r} differs from the folder name", file=sys.stderr)
    done, tags = (d / "DONE").is_file(), dict(meta.get("tags", {}))
    if done:
        tags.setdefault("status", "done")
    return {"name": meta.get("name") or d.name, "stage": meta.get("stage", ""), "params": meta.get("params", {}), "tags": tags,
            "metrics": metrics, "units": d / "units.jsonl" if (d / "units.jsonl").is_file() else None, "done": done}


def _hash(d):
    h = hashlib.sha256()
    for n in ("run.json", "metrics.jsonl", "units.jsonl", "DONE"):
        f = d / n
        h.update(n.encode() + (f.read_bytes() if f.is_file() else b"-"))
    return h.hexdigest()


def _manifest(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _tags(r, man):
    ent = man.get(r["name"]) if isinstance(man.get(r["name"]), dict) else {}
    return {str(k): str(v)[:5000] for k, v in {**r["tags"], **ent}.items()}


def _find(client, exp_id, src):
    return client.search_runs([exp_id], f"tags.phro_source = '{src}'")


def sync(root, p, client, only=None):
    root = Path(root)
    s = p["server"]
    man = _manifest(os.path.expanduser(s.get("manifest") or str(root) + "/runs.json"))
    exps = {x["name"]: x["experiment"] for x in p["stages"]}
    dirs = [Path(only)] if only else sorted(f.parent for f in (root / "runs").glob("**/run.json"))
    out = []
    for d in dirs:
        if not (d / "DONE").is_file():
            continue
        try:
            out += _sync_one(root, d, man, exps, client)
        except Exception as e:  # one bad run must not stall the rest
            print(f"sync error {d}: {e}", file=sys.stderr)
    return out


def _sync_one(root, d, man, exps, client):
    r = parse_run_dir(d)
    src = d.resolve().relative_to((root / "runs").resolve()).as_posix()
    ename = exps.get(r["stage"], r["stage"] or "default")
    e = client.get_experiment_by_name(ename)
    eid = e.experiment_id if e else client.create_experiment(ename)
    tags, h = _tags(r, man), _hash(d)
    mark, tmark = d / ".phro_synced", d / ".phro_tags"
    if not mark.is_file() or mark.read_text().strip() != h:
        for old in _find(client, eid, src):
            client.delete_run(old.info.run_id)
        rid = client.create_run(eid, run_name=r["name"], tags={**tags, "phro_source": src}).info.run_id
        for k, v in r["params"].items():
            client.log_param(rid, k, str(v)[:500])
        now = int(time.time() * 1000)
        for i, (k, v, st) in enumerate(r["metrics"]):  # strictly increasing timestamps: MLflow's latest value is the final row
            client.log_metric(rid, k, v, now + i, st)
        if r["units"]:
            client.log_artifact(rid, str(r["units"]))
        client.set_terminated(rid, "FINISHED")
        mark.write_text(h)
        tmark.write_text(json.dumps(tags, sort_keys=True))
        return [r["name"]]
    try:
        applied = json.loads(tmark.read_text())
    except (OSError, ValueError):
        applied = {}
    changed = {k: v for k, v in tags.items() if applied.get(k) != v}
    if changed:
        for old in _find(client, eid, src):
            for k, v in changed.items():
                client.set_tag(old.info.run_id, k, v)
        tmark.write_text(json.dumps(tags, sort_keys=True))
    return []


class _Fake:
    """Records calls; stands in for MlflowClient."""

    def __init__(self):
        self.calls, self.runs, self.exps = [], {}, {}

    def get_experiment_by_name(self, n):
        return self.exps.get(n)

    def create_experiment(self, n):
        self.exps[n] = type("E", (), {"experiment_id": "E" + str(len(self.exps))})()
        self.calls.append(("create_experiment", n))
        return self.exps[n].experiment_id

    def search_runs(self, ids, flt):
        src = flt.split("'")[1]
        return [type("R", (), {"info": type("I", (), {"run_id": k})()})() for k, v in self.runs.items() if v == src]

    def delete_run(self, rid):
        self.calls.append(("delete_run", rid))
        self.runs.pop(rid)

    def create_run(self, eid, run_name=None, tags=None):
        rid = f"run{len(self.calls)}"
        self.runs[rid] = tags["phro_source"]
        self.calls.append(("create_run", eid, run_name, tags))
        return type("R", (), {"info": type("I", (), {"run_id": rid})()})()

    def log_param(self, rid, k, v): self.calls.append(("param", k, v))
    def log_metric(self, rid, k, v, ts, st): self.calls.append(("metric", k, v, st, ts))
    def log_artifact(self, rid, f): self.calls.append(("artifact", Path(f).name))
    def set_terminated(self, rid, s): self.calls.append(("end", s))
    def set_tag(self, rid, k, v): self.calls.append(("tag", k, v))


def selftest():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        d = tmp / "runs/s1/r1_s0"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"name": "r1_s0", "stage": "s1", "params": {"lr": 1}, "tags": {"status": "done"}}), encoding="utf-8")
        (d / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (
            {"step": 100, "set": "heldout", "key": "f1", "value": 0.5}, {"step": "final", "set": "heldout", "key": "f1", "value": 0.8},
            {"step": 200, "set": "heldout", "key": "f1", "value": 0.7})), encoding="utf-8")
        (d / "units.jsonl").write_text('{"unit": "a"}\n', encoding="utf-8")
        assert not parse_run_dir(d)["done"]
        (d / "DONE").write_text("")
        r = parse_run_dir(d)
        assert r["done"] and ("heldout/f1", 0.8, 200) in r["metrics"] and r["units"].name == "units.jsonl"
        p = {"server": {"manifest": str(tmp / "runs.json")}, "stages": [{"name": "s1", "experiment": "1_s1"}]}
        c = _Fake()
        assert sync(tmp, p, c) == ["r1_s0"]
        assert [x[0] for x in c.calls].count("metric") == 3 and ("artifact", "units.jsonl") in c.calls and ("create_experiment", "1_s1") in c.calls
        ms = [x for x in c.calls if x[0] == "metric"]
        fin = next(x for x in ms if x[2] == 0.8)
        assert fin[4] == max(x[4] for x in ms if x[3] == fin[3]) and len({x[4] for x in ms}) == 3  # final row is newest at its step
        made = next(x for x in c.calls if x[0] == "create_run")
        assert made[3]["phro_source"] == "s1/r1_s0" and made[3]["status"] == "done"
        n = len(c.calls)
        assert sync(tmp, p, c) == [] and len(c.calls) == n  # unchanged: nothing happens
        (tmp / "runs.json").write_text(json.dumps({"r1_s0": {"intro": "i. 기대: j"}}), encoding="utf-8")
        assert sync(tmp, p, c) == [] and c.calls[n][:2] == ("tag", "intro")  # manifest change: tag only
        n = len(c.calls)
        assert sync(tmp, p, c) == [] and len(c.calls) == n
        (d / "DONE").write_text("again")
        assert sync(tmp, p, c) == ["r1_s0"] and any(x[0] == "delete_run" for x in c.calls) and len(c.runs) == 1
        bad = tmp / "runs/s0_bad"  # sorts before the good runs
        bad.mkdir()
        (bad / "run.json").write_text("{", encoding="utf-8")
        (bad / "DONE").write_text("")
        d2 = tmp / "runs/s1/r3"
        d2.mkdir()
        (d2 / "run.json").write_text('{"name": "r3", "stage": "s1"}')
        (d2 / "DONE").write_text("")
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert sync(tmp, p, c) == ["r3"]
        assert "sync error" in err.getvalue() and "s0_bad" in err.getvalue()
        (tmp / "runs/s1/r2").mkdir()
        (tmp / "runs/s1/r2/run.json").write_text('{"name": "r2"}')
        assert sync(tmp, p, c) == []  # no DONE: skipped
        d4 = tmp / "runs/s1/r4"
        d4.mkdir()
        (d4 / "run.json").write_text('{"stage": "s1"}')
        (d4 / "DONE").write_text("")
        r4 = parse_run_dir(d4)
        assert r4["name"] == "r4" and r4["tags"] == {"status": "done"}  # defaults: folder name, status=done
        (d4 / "run.json").write_text('{"name": "other", "tags": {"status": "smoke"}}')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert parse_run_dir(d4)["tags"]["status"] == "smoke"
        assert "differs from the folder name" in err.getvalue()
    print("phro_sync selftest ok")


def main(argv):
    if argv and argv[0] == "selftest":
        selftest()
        return 0
    import mlflow.tracking
    p = json.loads(Path(__file__).with_name("project.json").read_text(encoding="utf-8"))
    root = Path(os.path.expanduser(p["server"]["root"]))
    only = Path(argv[argv.index("--path") + 1]) if "--path" in argv else None
    import phro_server
    lk = phro_server.lock(root / "pids" / "sync.lock")
    if lk is None:
        return 0
    c = mlflow.tracking.MlflowClient(f"http://127.0.0.1:{p['server']['mlflow_port']}")
    for n in sync(root, p, c, only):
        print("synced", n)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
