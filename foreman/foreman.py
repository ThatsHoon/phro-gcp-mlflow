"""작업반장(foreman): 서버를 점검해 끝난 런·빈 GPU를 찾고, 보고와 다음 일을 다그친다. 표준 라이브러리만 쓴다.
  python foreman.py serve         # MCP stdio 서버(.mcp.json): 도구 foreman_check, foreman_report
  python foreman.py hook          # Claude Code Stop 훅: 보고 안 된 끝난 런이나 빈 GPU가 있으면 턴을 못 끝내게 막는다
  python foreman.py check         # 점검 결과(지시문)를 출력
  python foreman.py report --summary "..." [--ids all] [--next "run_x.sh GPU2"] [--blocked "사용자 결정 대기: ..."]
  python foreman.py selftest
상태는 <팩>/.phro/foreman/state.json, 보고 기록은 reports.jsonl. 숫자만 다룬다: 전사·pred 파일은 읽지 않는다.
훅은 <팩>/.phro/foreman/ENABLED 파일이 있을 때만 일한다. 내용이 "claim"이면 처음 훅을 부른 세션이 주인이 되고, 그 세션만 막는다. 끄려면 ENABLED를 지운다.
"""
import json, os, re, shlex, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import phro

PHRO = "python " + shlex.quote((Path(__file__).resolve().parents[1] / "bin" / "phro.py").as_posix())  # 지시문에 넣는 CLI 경로
FOREMAN = "python " + shlex.quote(Path(__file__).resolve().as_posix())
CACHE_S = 120  # 점검 결과(실패 포함)를 이만큼 다시 쓴다(턴마다 ssh 하지 않게)
ACK_LAUNCH_S = 45 * 60  # 띄웠다고 보고한 빈 GPU는 이만큼 봐준다(모델 적재·평가 대기)
ACK_BLOCKED_S = 60 * 60  # 못 띄우는 이유를 보고한 빈 GPU
MAX_SAME_BLOCK = 3  # 같은 이유로 연달아 막은 횟수가 이만큼이면 놓아 준다(풀 수 없는 상태의 무한 반복 방지)
P = None  # 현재 팩(use_pack으로 정한다)
STATE = REPORTS = PLAN = IDEAS = ENABLED = None


def use_pack(p):
    global P, STATE, REPORTS, PLAN, IDEAS, ENABLED
    P, d = p, Path(p["_dir"])
    STATE, REPORTS, ENABLED = d / "foreman" / "state.json", d / "foreman" / "reports.jsonl", d / "foreman" / "ENABLED"
    PLAN, IDEAS = d / "plan.md", d / "ledger" / "idea-bag.md"


def find(start=None):
    """(팩, 오류문). 팩이 없거나 잘못되면 팩은 None."""
    f = phro.find_pack(start or os.environ.get("CLAUDE_PROJECT_DIR") or None)
    if f is None:
        return None, "no .phro pack (run phro-onboard)"
    p, errs = phro.read_pack(f)
    return (p, "") if p else (None, "invalid .phro pack: " + "; ".join(errs[:3]))


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def snapshot(s, fresh=False):
    now = time.time()
    if not fresh and s.get("snap") and now - s.get("snap_t", 0) < CACHE_S:
        return s["snap"]
    if not fresh and now - s.get("fail_t", 0) < CACHE_S:  # 서버가 죽어 있으면 Stop마다 ssh 시간 초과를 기다리지 않는다
        return None
    snap = phro.snapshot(P)
    if snap is None:
        s["fail_t"] = now
    else:
        s["snap"], s["snap_t"], s["fail_t"] = snap, now, 0
        if "seen" not in s:  # 처음 켤 때 이미 있던 끝남 표시는 보고된 것으로 친다(옛 일로 몰아붙이지 않게)
            s["seen"] = sorted(snap["done"])
    return snap


def run_name(path):
    """끝남 파일 -> 매니페스트 런 이름. 규칙은 팩의 server.run_name_rules."""
    return phro.run_name(path, P["server"]["run_name_rules"])


def expectation(intro):
    m = re.search(r"(?:기대|Expected)\s*[:：]\s*(.+)", intro or "")
    return m.group(1).strip() if m else "(서론에 기대가 없음: 그것부터 문제다)"


def idle_gpus(snap, s, now):
    ack, queued = s.get("ack", {}), {l.split()[0] for l in snap.get("queue", [])}  # 대기열에 줄이 있는 GPU는 서버 대기열 크론이 5분 안에 채운다
    return [g for g, util, mem in snap["gpus"] if util < 5 and mem < 500 and ack.get(str(g), 0) < now and str(g) not in queued]


def plan_items():
    try:
        return [l[5:].strip() for l in PLAN.read_text(encoding="utf-8").splitlines() if l.startswith("- [ ]")]
    except OSError:
        return []


def queued_ideas():
    try:
        t = IDEAS.read_text(encoding="utf-8")
    except OSError:
        return []
    return [m.group(1) for m in re.finditer(r"### (IDEA-\d+ [^\n]+)\n(?:- [^\n]*\n)*?- status: queued", t)]


def orders(snap, s, now=None):
    """(막을지, 지시문). 끝났는데 보고 안 된 런, 빈 GPU, 다음 일 후보."""
    now = now or time.time()
    new = sorted(set(snap["done"]) - set(s.get("seen", [])))
    idle = idle_gpus(snap, s, now)
    L = []
    for p in new:
        n = run_name(p)
        d = snap["docs"].get(n or "", {})
        if n:
            L.append(f"- 끝남 `{p}` = 런 `{n}`. 서론의 기대: {expectation(d.get('intro'))}\n"
                     f"  -> `{PHRO} report --run {n}`(또는 compare)로 숫자를 뽑아 기대와 맞았는지 판정하고, `result`·`insight`를 서버 매니페스트에 써라."
                     + ("" if d.get("intro") else " 서론도 없이 돌렸다. 그것도 써라."))
        else:
            L.append(f"- 스크립트 끝 표시 `{p}`: 그 스크립트가 띄운 런 전부 판정하라.")
    for g in idle:
        L.append(f"- GPU{g} 놀고 있다. 지금 채워라(현재 계획 안의 다음 실험). 준비된 다음 일은 `{PHRO} queue add <GPU> <script>`로 서버 대기열에도 넣어 두어라(세션 밖에서도 채워진다. read_only 서버면 제안만). 못 채우면 이유를 보고해라.")
    if idle:
        cand = (plan_items()[:5] + [f"{k}: 서론만 있고 결과 없음" for k, v in snap["docs"].items() if v.get("intro") and not v.get("result")][:5]
                + queued_ideas()[:3])
        if not cand:
            cand = ["돌파구 조사: ledger/insights.md 열린 한계"]
        cand.append(f"{PHRO} audit: 전체 예산 재검증·leave-one-out 대기 확인")
        L.append("  다음 일 후보(plan.md 먼저, 그다음 결과 없는 런, 대기 아이디어):\n" + "\n".join(f"  - {c}" for c in cand)
                 if cand else "  다음 일 후보 없음: plan.md를 채워라.")
    head = (f"서버: {', '.join(f'GPU{g} {u}%/{m}MiB' for g, u, m in snap['gpus'])} | 대기 스크립트: {', '.join(snap['waiting']) or '-'}"
            f" | GPU 대기열: {'; '.join(snap.get('queue', [])) or '비어 있음'}")
    if not L:
        return False, head + "\n이상 없음. 다음 끝남을 기다리는 감시가 걸려 있는지만 확인해라."
    return True, (head + f"\n작업반장: 할 일이 남았다. 끝내기 전에 처리하고 `foreman_report`(또는 `{FOREMAN} report`)로 보고해라.\n"
                  + "\n".join(L) + "\n방향 전환·새 방법·삭제는 사용자에게 물어라. 그 경우 blocked에 이유를 적어 보고해라.")


def report(s, summary, ids="all", next_action="", blocked=""):
    """보고를 받는다. 끝남 표시를 보고된 것으로 치고, 빈 GPU는 띄웠거나 못 띄우는 이유가 있으면 한동안 봐준다."""
    if len((summary or "").strip()) < 20:
        return False, "보고가 너무 짧다. 무엇이 끝났고 숫자(지표 키 이름과 함께)가 어떻고 기대와 맞았는지 써라."
    snap = s.get("snap") or {"done": {}, "gpus": []}
    now = time.time()
    idle = idle_gpus(snap, s, now)
    if idle and not (next_action or blocked):
        return False, f"GPU{idle} 비어 있다. 띄운 스크립트(next)나 못 띄우는 이유(blocked)를 같이 보고해라."
    new = sorted(set(snap["done"]) - set(s.get("seen", [])))
    take = new if ids in ("all", "", None) else [p for p in new if any(i in p for i in str(ids).split(","))]
    s["seen"] = sorted(set(s.get("seen", [])) | set(take))
    for g in idle:
        s.setdefault("ack", {})[str(g)] = now + (ACK_LAUNCH_S if next_action else ACK_BLOCKED_S)
    s["snap_t"], s["blocks"] = 0, []  # 다음 점검은 새로
    with open(REPORTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "events": take, "summary": summary, "next": next_action,
                            "blocked": blocked, "idle": idle}, ensure_ascii=False) + "\n")
    left = sorted(set(new) - set(take))
    return True, f"보고 받음: 끝남 {len(take)}건 처리" + (f", 남은 것 {left}" if left else "") + (f". GPU{idle} 봐줌." if idle else ".")


# ---------- Stop 훅 ----------
def hook():
    inp = json.loads(sys.stdin.read() or "{}")
    p, _ = find(os.environ.get("CLAUDE_PROJECT_DIR") or inp.get("cwd") or None)
    if p is None:
        return
    use_pack(p)
    if not ENABLED.exists():
        return
    owner, sid = ENABLED.read_text(encoding="utf-8").strip(), inp.get("session_id", "")
    if owner == "claim" and sid:
        ENABLED.write_text(sid, encoding="utf-8")
        owner = sid
    if owner != sid:
        return
    s = load_state()
    snap = snapshot(s)
    if snap is None:
        save_state(s)
        return  # 서버를 못 봤다: 막지 않는다
    block, text = orders(snap, s)
    fp = text.split("\n", 1)[-1]
    s["blocks"] = (s.get("blocks", []) + [fp])[-MAX_SAME_BLOCK:] if block else []
    save_state(s)
    if block and not (len(s["blocks"]) == MAX_SAME_BLOCK and len(set(s["blocks"])) == 1):
        print(json.dumps({"decision": "block", "reason": text}, ensure_ascii=True))


# ---------- MCP stdio 서버(JSON-RPC 2.0, 줄 단위) ----------
TOOLS = [
    {"name": "foreman_check", "description": "서버를 지금 점검해 끝난 런, 빈 GPU, 다음 일 후보와 지시를 돌려준다. 각 런 서론의 '기대'와 맞춰 판정하라고 다그친다.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "foreman_report", "description": "작업반장에게 보고한다. 끝난 런의 판정(지표 키와 숫자, 기대와 맞았는지)을 summary에, 빈 GPU에 띄운 것을 next에, 사용자 결정 대기 등 못 띄우는 이유를 blocked에 쓴다. 보고해야 Stop 훅이 풀린다.",
     "inputSchema": {"type": "object", "required": ["summary"], "properties": {
         "summary": {"type": "string"}, "ids": {"type": "string", "description": "처리한 끝남 표시(경로 일부, 쉼표). 기본 all"},
         "next": {"type": "string"}, "blocked": {"type": "string"}}}},
]


def call(name, args):
    if P is None:
        p, err = find()
        if p is None:
            return True, err
        use_pack(p)
    if not ENABLED.exists():
        return True, "foreman disabled: create .phro/foreman/ENABLED"
    s = load_state()
    if name == "foreman_check":
        snap = snapshot(s, fresh=True)
        save_state(s)
        return (True, "서버 점검 실패(ssh). 연결부터 확인해라.") if snap is None else (False, orders(snap, s)[1])
    if name == "foreman_report":
        snapshot(s, fresh=True)
        ok, text = report(s, args.get("summary", ""), args.get("ids", "all"), args.get("next", ""), args.get("blocked", ""))
        save_state(s)
        return not ok, text
    return True, f"모르는 도구 {name}"


def handle(msg):
    mid, method = msg.get("id"), msg.get("method", "")
    if method == "initialize":
        res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}},
               "serverInfo": {"name": "phro-foreman", "version": "0.1.0"}}
    elif method == "tools/list":
        res = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        err, text = call(p.get("name"), p.get("arguments") or {})
        res = {"content": [{"type": "text", "text": text}], "isError": err}
    elif method == "ping":
        res = {}
    elif mid is None:
        return None  # 알림(notifications/*)에는 답하지 않는다
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def serve():
    for line in sys.stdin:
        if line.strip():
            out = handle(json.loads(line))
            if out is not None:
                sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    R = [[r"(?:.*/)?([^/]+)/eval_(.+)_final\.json$", r"\1_\2"], [r"(?:.*/)?([^/]+)/final$", r"\1"]]
    use_pack({"_dir": tmp, "server": {"run_name_rules": R}})
    (tmp / "plan.md").write_text("", encoding="utf-8")
    assert not (tmp / "foreman" / "ENABLED").exists() and call("foreman_check", {}) == (True, "foreman disabled: create .phro/foreman/ENABLED")
    assert run_name("runs/runx/eval_m1_s2_final.json") == "runx_m1_s2" and run_name("runs/refine/gx/final") == "gx"
    assert run_name("runs/runx/RUNXS2_DONE") is None and expectation("... 기대: F1 0.8 이상. 아니면 x").startswith("F1 0.8")
    assert expectation("... Expected: F1 >= 0.8").startswith("F1") and expectation("x 기대: 0.80 이상").startswith("0.80")
    block, text = orders({"gpus": [[0, 0, 0]], "done": {}, "docs": {}, "waiting": [], "queue": []}, {"seen": []})
    assert block and "돌파구" in text and "phro.py audit" in text  # idle GPU and empty plan -> breakthrough candidate
    STATE, REPORTS = tmp / "s.json", tmp / "r.jsonl"
    globals().update(STATE=STATE, REPORTS=REPORTS)
    snap = {"gpus": [[0, 100, 8000], [1, 0, 0]], "done": {"runs/runx/eval_m1_s2_final.json": 1}, "waiting": [],
            "docs": {"runx_m1_s2": {"intro": "x. 기대: 0.80 이상", "result": ""}}}
    s = {"snap": snap, "snap_t": time.time(), "seen": []}
    block, text = orders(snap, s)
    assert block and "runx_m1_s2" in text and "0.80 이상" in text and "GPU1" in text and f"{PHRO} report --run runx_m1_s2" in text and FOREMAN + " report" in text
    msg = "runx m1 s2 끝, heldout/acc 0.61로 기대(0.80) 못 미침"
    assert not report(s, "짧음")[0] and not report(s, msg)[0]  # 빈 GPU1 설명 없음
    ok, _ = report(s, msg, next_action="run_runx_m1_s3.sh GPU1")
    assert ok and s["seen"] == ["runs/runx/eval_m1_s2_final.json"] and not orders(snap, s)[0]
    assert idle_gpus({"gpus": [[1, 0, 0], [2, 0, 0]], "queue": ["2 run_x.sh"]}, {}, time.time()) == [1]
    # hook: stubbed snapshot and pack, claim ownership, loop breaker
    import io, contextlib
    (tmp / "foreman").mkdir(exist_ok=True)
    (tmp / "foreman" / "ENABLED").write_text("claim", encoding="utf-8")
    STATE.unlink(missing_ok=True)
    phro.snapshot = lambda p: {"gpus": [[0, 100, 8000]], "done": {"runs/runx/final": 1}, "docs": {}, "waiting": [], "queue": []}
    pack = {"_dir": tmp, "server": {"run_name_rules": R}}
    globals()["find"] = lambda start=None: (pack, "")
    def run_hook(sid):
        sys.stdin, out = io.StringIO(json.dumps({"session_id": sid})), io.StringIO()
        with contextlib.redirect_stdout(out):
            hook()
        return out.getvalue()
    use_pack(pack); save_state({"seen": []})
    assert '"decision": "block"' in run_hook("a") and ENABLED.read_text(encoding="utf-8") == "a"
    assert PHRO.endswith("bin/phro.py") and Path(shlex.split(PHRO)[1]).is_file() and Path(shlex.split(FOREMAN)[1]).is_file()
    assert run_hook("b") == ""  # other session is not blocked
    assert run_hook("a") != "" and run_hook("a") == ""  # third identical block is let through (MAX_SAME_BLOCK=3)
    n_ssh = []  # negative cache: a down server is asked once per CACHE_S, not on every Stop
    phro.snapshot = lambda p: n_ssh.append(1)
    s = {}
    assert snapshot(s) is None and snapshot(s) is None and len(n_ssh) == 1 and snapshot(s, fresh=True) is None and len(n_ssh) == 2
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"][0]["name"] == "foreman_check"
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["serverInfo"]["name"] == "phro-foreman"
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle({"jsonrpc": "2.0", "id": 2, "method": "nope"})["error"]["code"] == -32601
    print("foreman selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "serve":
        serve()
    elif cmd == "hook":
        hook()
    elif cmd == "selftest":
        selftest()
    elif cmd == "check":
        print(call("foreman_check", {})[1])
    elif cmd == "report":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("--summary", required=True); ap.add_argument("--ids", default="all")
        ap.add_argument("--next", default=""); ap.add_argument("--blocked", default="")
        a = ap.parse_args()
        err, text = call("foreman_report", {"summary": a.summary, "ids": a.ids, "next": a.next, "blocked": a.blocked})
        print(text)
        sys.exit(1 if err else 0)
