#!/usr/bin/env python3
"""phro VM kit: setup, GPU queue, run manifest. Stdlib only; no sudo; writes only under the project root and the user's crontab."""
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

MLFLOW = "mlflow==2.17.2"


def cron_merge(current, project, lines):
    tag = f"# phro:{project}"
    keep = [l for l in current.splitlines() if not l.rstrip().endswith(tag)]
    return "\n".join(keep + [f"{l} {tag}" for l in lines]) + "\n"


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _smi(*args):
    """nvidia-smi output, or None when it cannot be trusted (missing, failed, hung)."""
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def busy_gpus():
    """Every GPU with ANY compute process, ours or not. None = unknown (fail closed: caller launches nothing)."""
    g, a = _smi("--query-gpu=index,uuid"), _smi("--query-compute-apps=gpu_uuid")
    if g is None or a is None:
        return None
    idx = {}
    for l in g.splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) == 2 and f[0].isdigit():
            idx[f[1]] = int(f[0])
    if not idx:
        return None
    used = [u.strip() for u in a.splitlines() if u.strip()]
    return set(idx.values()) if any(u not in idx for u in used) else {idx[u] for u in used}


def pid_busy(root):
    """GPUs whose launched job is still alive (it may not hold a CUDA context yet). Read-only toward processes."""
    out = set()
    for f in Path(root, "pids").glob("*.pid"):
        if f.stem.isdigit():
            try:
                os.kill(int(f.read_text().strip()), 0)
                out.add(int(f.stem))
            except (OSError, ValueError):
                pass
    return out


def lock(path, wait=False):
    """Exclusive lock (non-blocking unless wait); returns the open file (keep it referenced) or None if held. POSIX only, a no-op elsewhere."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        import fcntl
    except ImportError:  # Windows PC: selftest only, the VM is Linux
        return f
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except OSError:
        f.close()
        return None
    return f


def cron_current(rc, out, err):
    """Existing crontab text, or None when crontab -l failed for any reason other than 'no crontab'."""
    if rc == 0:
        return out
    return "" if "no crontab" in err.lower() else None


def queue_step(lines, busy, gpus):
    launches, taken, rest = [], set(), []
    for l in lines:
        f = l.split(None, 1)
        if len(f) == 2 and f[0].isdigit() and int(f[0]) < gpus and int(f[0]) not in busy and int(f[0]) not in taken:
            taken.add(int(f[0]))
            launches.append((int(f[0]), f[1].strip()))
        else:
            rest.append(l)
    return launches, rest


def launch(gpu, script, root):
    root = Path(root)
    for d in ("logs", "pids"):
        (root / d).mkdir(exist_ok=True)
    log = open(root / "logs" / (Path(script).stem + ".log"), "ab")
    pr = subprocess.Popen(["setsid", "nohup", "bash", script], cwd=root, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    (root / "pids" / f"{gpu}.pid").write_text(str(pr.pid))
    return pr.pid


def _atomic(path, text):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _load(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def manifest_get(path, name):
    d = _load(path).get(name)
    return d if isinstance(d, dict) else {}


def manifest_set(path, name, pairs):
    d = _load(path)
    d[name] = {**(d.get(name) if isinstance(d.get(name), dict) else {}), **pairs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _atomic(path, json.dumps(d, ensure_ascii=False, indent=1))


def _x(s):
    return os.path.expanduser(s)


def _path(s, key, default):
    """The pack on the VM is the raw file, so the PC loader defaults are applied here too."""
    return _x(s.get(key) or s["root"] + "/" + default)


def mlflow_running(root):
    """True when pids/mlflow.pid names a live process (a setup re-run)."""
    try:
        os.kill(int((Path(root) / "pids" / "mlflow.pid").read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def setup(root, p):
    s = p["server"]
    port, root = s["mlflow_port"], Path(root)
    running = mlflow_running(root)
    if not running and not port_free(port):
        print(f"port {port} is taken on 127.0.0.1: aborting, nothing changed")
        return 1
    c = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    cur = cron_current(c.returncode, c.stdout, c.stderr)
    if cur is None:
        print(f"crontab -l failed ({c.stderr.strip()}): aborting, nothing changed")
        return 1
    for d in ("runs", "logs", "pids", "artifacts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    venv = Path(_x(s["venv"]))
    py = venv / "bin" / "python"
    subprocess.run(["python3", "-m", "venv", str(venv)], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", MLFLOW], check=True)
    start = (f"nohup {venv}/bin/mlflow server --host 127.0.0.1 --port {port} --backend-store-uri sqlite:///{root}/mlflow.db "
             f"--artifacts-destination {root}/artifacts >> {root}/logs/mlflow.log 2>&1 & echo $! > {root}/pids/mlflow.pid")
    if not running:
        subprocess.run(["bash", "-c", start], check=True)
    new = cron_merge(cur, p["name"], [f"@reboot {start}", f"*/10 * * * * {py} {root}/phro_sync.py >> {root}/logs/sync.log 2>&1",
                                      f"*/5 * * * * python3 {root}/phro_server.py queue >> {root}/logs/queue.log 2>&1"])
    subprocess.run(["crontab", "-"], input=new, text=True, check=True)
    print(f"phro setup ok: mlflow {'already running' if running else 'started'} on 127.0.0.1:{port}, cron tagged # phro:{p['name']}")
    return 0


def queue(p, busy=busy_gpus, start=launch):
    s = p["server"]
    root, qf = Path(_x(s["root"])), Path(_path(s, "queue", "gpu_queue.txt"))
    lk = lock(root / "pids" / "queue.lock")
    if lk is None:
        return 0

    def read():
        try:
            return qf.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    b = busy()
    if b is None:
        print("nvidia-smi unavailable or unmapped GPU process: launching nothing")
        return 0
    launches, _ = queue_step(read(), b | pid_busy(root), s["gpus"])
    if not launches:
        return 0
    fresh = read()  # re-read just before the rewrite: keeps lines the PC appended meanwhile
    for g, script in launches:
        k = next((i for i, l in enumerate(fresh) if l.split(None, 1) == [str(g), script]), None)
        if k is not None:
            del fresh[k]
    _atomic(qf, "".join(l + "\n" for l in fresh))  # rewrite first: a crash loses a launch, never double-runs a job
    for g, script in launches:
        start(g, script, root)
        print(f"launched gpu {g}: {script}")
    return 0


def queue_add(p, gpu, script):
    """Append one '<GPU> <script>' line under the queue lock (the launcher rewrites the file under the same lock)."""
    s = p["server"]
    if s.get("read_only"):
        print("server.read_only: refusing to queue")
        return 1
    if not (gpu.isdigit() and int(gpu) < s["gpus"]) or not script.strip() or "\n" in script:
        print(f"usage: queue add GPU SCRIPT (GPU 0..{s['gpus'] - 1}, one-line script)")
        return 1
    root, qf = Path(_x(s["root"])), Path(_path(s, "queue", "gpu_queue.txt"))
    lk = lock(root / "pids" / "queue.lock", wait=True)
    with open(qf, "a", encoding="utf-8") as f:
        f.write(f"{int(gpu)} {script.strip()}\n")
    lk.close()
    print(f"queued gpu {int(gpu)}: {script.strip()}")
    return 0


def selftest():
    cur = "0 1 * * * backup.sh\n*/5 * * * * old # phro:proj\n*/5 * * * * other # phro:proj2\n"
    out = cron_merge(cur, "proj", ["*/5 * * * * new"])
    assert "backup.sh" in out and "old" not in out and "# phro:proj2" in out and out.rstrip().endswith("new # phro:proj")
    assert cron_merge("", "proj", ["a"]) == "a # phro:proj\n"
    assert queue_step(["0 a.sh", "0 b.sh", "1 c.sh", "# note"], busy={1}, gpus=2) == ([(0, "a.sh")], ["0 b.sh", "1 c.sh", "# note"])
    assert queue_step(["2 x.sh"], busy={2}, gpus=4) == ([], ["2 x.sh"])  # a foreign process on GPU2 counts as busy
    assert queue_step(["9 x.sh", "bad"], busy=set(), gpus=4) == ([], ["9 x.sh", "bad"])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        manifest_set(tmp / "m.json", "r1", {"intro": "x. 기대: y"})
        manifest_set(tmp / "m.json", "r1", {"result": "z"})
        m = manifest_get(tmp / "m.json", "r1")
        assert m["intro"].endswith("y") and m["result"] == "z" and manifest_get(tmp / "m.json", "none") == {}
    assert cron_current(0, "x\n", "") == "x\n" and cron_current(1, "", "no crontab for u") == "" and cron_current(1, "", "denied") is None
    real = _smi
    try:
        globals()["_smi"] = lambda *a: None
        assert busy_gpus() is None
        globals()["_smi"] = lambda *a: "0, U0\n1, U1\n" if "--query-gpu=index,uuid" in a else "U1\n"
        assert busy_gpus() == {1}
        globals()["_smi"] = lambda *a: "0, U0\n" if "--query-gpu=index,uuid" in a else "ghost\n"
        assert busy_gpus() == {0}  # unmapped process: all GPUs busy
        globals()["_smi"] = lambda *a: "" if "--query-gpu=index,uuid" in a else "U1\n"
        assert busy_gpus() is None
    finally:
        globals()["_smi"] = real
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pids").mkdir()
        (root / "gpu_queue.txt").write_text("0 a.sh\n1 b.sh\n", encoding="utf-8")
        pk = {"server": {"root": str(root), "gpus": 2}}
        started = []
        assert queue(pk, busy=lambda: None, start=lambda *a: started.append(a)) == 0 and started == []
        assert (root / "gpu_queue.txt").read_text() == "0 a.sh\n1 b.sh\n"  # failed query: nothing launched, queue untouched
        (root / "pids" / "1.pid").write_text(str(os.getpid()))
        assert pid_busy(root) == {1}
        seen = []
        queue(pk, busy=lambda: set(), start=lambda g, s_, r: seen.append((g, s_, (root / "gpu_queue.txt").read_text())))
        assert seen == [(0, "a.sh", "1 b.sh\n")]  # gpu1 busy via live pid; queue rewritten before launch
        assert lock(root / "pids" / "x.lock") is not None
        assert queue_add(pk, "1", "c.sh") == 0 and queue_add(pk, "2", "d.sh") == 1 and queue_add(pk, "x", "d.sh") == 1
        assert queue_add(pk, "0", "a\nb") == 1 and queue_add({"server": {**pk["server"], "read_only": True}}, "0", "e.sh") == 1
        assert (root / "gpu_queue.txt").read_text().splitlines() == ["1 b.sh", "1 c.sh"] and (root / "pids" / "queue.lock").exists()
        assert not mlflow_running(root)
        (root / "pids" / "mlflow.pid").write_text(str(os.getpid()))
        assert mlflow_running(root)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        assert not port_free(s.getsockname()[1])
    print("phro_server selftest ok")


def main(argv):
    if not argv:
        print("usage: phro_server.py setup | queue | queue add GPU SCRIPT | manifest get NAME | manifest set NAME k=v ... | selftest")
        return 0
    if argv[0] == "selftest":
        selftest()
        return 0
    p = json.loads(Path(__file__).with_name("project.json").read_text(encoding="utf-8"))
    s = p["server"]
    if argv[0] == "setup":
        return setup(Path(_x(s["root"])), p)
    if argv[0] == "queue" and len(argv) == 4 and argv[1] == "add":
        return queue_add(p, argv[2], argv[3])
    if argv[0] == "queue" and len(argv) == 1:
        return queue(p)
    if argv[0] == "manifest" and len(argv) >= 3 and argv[1] in ("get", "set"):
        path = _path(s, "manifest", "runs.json")
        if argv[1] == "get":
            print(json.dumps(manifest_get(path, argv[2]), ensure_ascii=False))
        else:
            manifest_set(path, argv[2], dict(a.split("=", 1) for a in argv[3:]))
        return 0
    print(f"unknown command: {argv[0]}")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
