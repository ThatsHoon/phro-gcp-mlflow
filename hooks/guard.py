import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

TAIL = " Run it yourself in a terminal if you really mean it."
INST_VERBS = {"stop", "delete", "reset", "start", "create", "suspend", "resume",
              "set-machine-type", "add-metadata", "remove-metadata"}
SSH_VAL = set("-b -c -D -E -e -F -I -i -J -L -l -m -O -o -p -Q -R -S -W -w".split())
WRAP = {"sudo", "env", "exec", "command", "nohup", "time", "timeout", "nice", "stdbuf"}
WRAP_VAL = {"sudo": set("-u -g -C -D -h -p -r -t -U".split()), "timeout": {"-s", "-k"}, "nice": {"-n"}, "env": {"-u"}}
KW = {"do", "then", "else", "elif", "!", "if", "while", "until"}
SHELLS = {"bash", "sh", "zsh"}
NET = re.compile(r"gcloud\b.*\bcompute\s+(instances\s+(" + "|".join(sorted(INST_VERBS)) + r")|firewall-rules\s+(create|update|delete))(?![\w-])")
READ = {"list", "describe", "info", "version"}
MUT = re.compile(r"(delete|create|update|stop|start|reset|suspend|resume|set|unset|deploy|ssh|scp|import|export|move|patch|(add|remove|set)-.*)$")


def _split(s):
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


def code_lines(cmd):
    """Command lines with heredoc bodies removed."""
    end = None
    for line in cmd.splitlines():
        if end is not None:
            if line.strip() == end:
                end = None
            continue
        m = re.search(r"<<-?\s*\\?(['\"]?)([A-Za-z0-9_.-]+)\1", re.sub(r"\(\(.*?\)\)", "", line).replace("<<<", " "))
        if m:
            end = m.group(2)
        yield line


def segments(cmd):
    for line in code_lines(cmd):
        for seg in re.split(r"&&|\|\||;|\||&|\$\(|`", line):
            t = _split(seg)
            if t:
                yield t


def _base(t):
    b = re.split(r"[\\/]", t)[-1].lower()
    return re.sub(r"\.(exe|cmd|bat)$", "", b)


def _lead(t):
    """Drop ( { keywords, wrappers with their options, and VAR=x prefixes; return (binary, args)."""
    i, w = 0, ""
    while i < len(t):
        x = re.sub(r"^[({]+", "", t[i])
        if not x or x in KW:
            i += 1
        elif _base(x) in WRAP:
            w, i = _base(x), i + 1
        elif w and x in WRAP_VAL.get(w, ()):
            i += 2
        elif re.match(r"[A-Za-z_]\w*=", x) or (w and (x.startswith("-") or re.fullmatch(r"\d+(\.\d+)?[smhd]?", x))):
            i += 1
        else:
            return _base(x), t[i + 1:]
    return "", []


def _host(h):
    return h.rpartition("@")[2].strip().strip("'\"").rstrip(")}").lower()


def _colon_host(tok):
    if tok.startswith("-"):
        return None
    h = _host(tok)
    if ":" not in h:
        return None
    h = h.split(":", 1)[0]
    return h if len(h) > 1 else None


def _bad(hosts, allow):
    return [h for h in hosts if h and h not in allow]


def _gpos(args):
    """gcloud positionals; a `--flag value` pair drops the value unless it looks like a mutating verb."""
    pos, i = [], 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("-") and not MUT.match(args[i + 1]):
            i += 2
            continue
        if not a.startswith("-"):
            pos.append(a)
        i += 1
    return pos


def _gcloud(args, allow):
    """Allowlist: read-only verbs, and compute ssh/scp to an allowed host. Everything else is blocked."""
    pos = _gpos(args)
    c = pos[pos.index("compute"):] if "compute" in pos else []
    if c[:2] in (["compute", "ssh"], ["compute", "scp"]):
        if allow is None:
            return "gcloud compute " + c[1] + " is blocked without a valid project pack."
        hosts = ([_host(c[2])] if len(c) > 2 else []) if c[1] == "ssh" else [_colon_host(a) for a in c[2:]]
        if not any(hosts) or _bad(hosts, allow):
            return "gcloud compute " + c[1] + " target is not in server.allow_hosts."
        return None
    k = next((i for i, x in enumerate(pos) if x in READ or x.startswith(("get-", "list-"))), None)
    if not pos or (k is not None and k >= len(pos) - 2 and not any(MUT.match(x) for x in pos[:k])):
        return None
    return "gcloud " + " ".join(pos[:4]) + " is blocked: only read-only gcloud (list, describe, info, version, get-*, list-*) runs here."


def _ssh(b, args, allow):
    """(reason or None, command strings to check too: the remote command and any ProxyCommand)."""
    hosts, subs, i = [], [], 0
    while i < len(args):
        a = args[i]
        if b in ("ssh", "scp") and a[:2] in ("-o", "-J"):
            v = a[2:] or (args[i + 1] if i + 1 < len(args) else "")
            i += 1 if a[2:] else 2
            m = re.match(r"\s*(\w+)\s*[=\s]\s*(.*)", ("ProxyJump=" if a[:2] == "-J" else "") + v, re.S)
            k, v = (m.group(1).lower(), m.group(2)) if m else ("", "")
            if k == "proxyjump":
                hosts += [_host(re.sub(r"^ssh://", "", x)).split(":")[0] for x in v.split(",")]
            elif k == "hostname":
                hosts.append(_host(v))
            elif k == "proxycommand":
                subs.append(v)
            continue
        if a in SSH_VAL and (b == "ssh" or a == "-J"):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        if b == "ssh":
            hosts.append(_host(a))
            subs.append(" ".join(args[i + 1:]))
            break
        hosts.append(_colon_host(a))
        i += 1
    bad = _bad(hosts, allow)
    return ("host not in server.allow_hosts: " + ", ".join(bad) + ".") if bad else None, subs


def check_bash(c, allow, depth=0):
    if depth > 4:
        return "Command nesting is too deep to check."
    for t in segments(c):
        b, args = _lead(t)
        r = None
        if b == "gcloud":
            r = _gcloud(args, allow)
        elif b in ("ssh", "scp", "rsync") and allow is not None:
            r, subs = _ssh(b, args, allow)
            r = r or next((x for x in (check_bash(s, allow, depth + 1) for s in subs) if x), None)
        elif b in SHELLS:
            k = next((j for j, x in enumerate(args) if re.fullmatch(r"-[a-z]*c[a-z]*", x)), None)
            if k is not None and k + 1 < len(args):
                r = check_bash(args[k + 1], allow, depth + 1)
        if r:
            return r
    return None


def pack_lists(p):
    """(allow_hosts set or None without a pack, forbidden_terms) from a raw pack, whether or not check_pack passes."""
    if p is None:
        return None, []
    s = p.get("server") if isinstance(p, dict) else None
    e = p.get("egress") if isinstance(p, dict) else None
    a = s.get("allow_hosts") if isinstance(s, dict) else None
    t = e.get("forbidden_terms") if isinstance(e, dict) else None
    return ({str(x).lower() for x in a} if isinstance(a, list) else set()), (t if isinstance(t, list) else [])


def pack_for(f):
    """The pack with defaults even when invalid; {} when the file cannot be parsed (then every ssh/scp/rsync is blocked)."""
    import phro
    try:
        return phro.load_pack(f)
    except Exception:
        return {}


def verdict(tool, ti, p):
    ti = ti if isinstance(ti, dict) else {}
    allow, terms = pack_lists(p)
    if tool == "Bash":
        c = str(ti.get("command", ""))
        if NET.search(c):  # raw text, heredoc bodies included
            return "Destructive gcloud compute command is blocked." + TAIL
        r = check_bash(c, allow)
        if r:
            return r + TAIL
    elif tool in ("WebSearch", "WebFetch"):
        text = " ".join(str(ti.get(k, "")) for k in ("query", "url", "prompt")).lower()
        for term in terms:
            if str(term).strip() and str(term).lower() in text:
                return "Outbound text matches egress.forbidden_terms." + TAIL
    return None


def selftest():
    P = {"server": {"allow_hosts": ["my-gpu", "10.9.8.7"]}, "egress": {"forbidden_terms": ["secretset"]}}
    BLOCK = ["gcloud compute instances stop other-vm --zone x", "gcloud compute instances delete a", "gcloud compute firewall-rules create x",
             "ssh other-host ls", "scp f other:/tmp", "ssh 10.0.0.5 ls", "gcloud compute ssh other-vm", "cd x && ssh -J jump my-gpu ls",
             "rsync -av d/ other:~/d", "sudo /usr/bin/gcloud compute instances stop a", "gcloud.cmd compute instances start a",
             "env X=1 /usr/bin/ssh OTHER ls", "ssh me@Other ls", "echo hi\nssh other ls", "python - <<'EOF'\nx\nEOF\nssh other ls",
             # I2.1 gcloud in heredoc bodies; I2.2 gcloud allowlist
             "python - <<'EOF'\nssh other\ngcloud compute instances stop x\nEOF", "bash <<'EOF'\ngcloud compute instances stop x\nEOF",
             "gcloud compute instances add-labels vm --labels a=b", "gcloud config set project x", "gcloud projects delete p",
             "gcloud compute instances delete vm1 list", "gcloud storage rm gs://b/x", "gcloud auth print-access-token",
             # I2.3 lead stripping, keywords, single &
             "(ssh other ls)", "{ ssh other ls; }", "x=$(ssh other ls)", "echo `ssh other ls`", "for h in a; do ssh other ls; done",
             "if true; then ssh other ls; fi", "if x; then :; else ssh other ls; fi", "if x; then :; elif ssh other ls; then :; fi",
             "! ssh other ls", "true & ssh other ls", "sleep 1 &ssh other ls",
             # I2.4 sh -c and the remote command of an allowed ssh; ProxyJump, ProxyCommand
             "bash -c 'ssh other ls'", "sh -c \"ssh other ls\"", "zsh -lc 'scp f other:/tmp'", "ssh my-gpu 'ssh other ls'",
             "ssh my-gpu ssh other ls", "bash -c \"ssh my-gpu 'ssh other ls'\"", "ssh my-gpu 'gcloud config set project x'",
             "ssh -o ProxyJump=other my-gpu", "ssh -oProxyJump=other my-gpu", "ssh -Jother my-gpu", "scp -o ProxyJump=other f my-gpu:~/a",
             "ssh -o 'ProxyCommand=ssh -W %h:%p other' my-gpu", "ssh -o ProxyCommand='ssh other -W %h:%p' my-gpu",
             "ssh -o 'ProxyCommand ssh other -W %h:%p' my-gpu", "ssh -o HostName=other my-gpu",
             # I2.5 wrapper options with values
             "sudo -u bob ssh other ls", "timeout -s KILL 30 ssh other ls", "nice -n 5 ssh other ls", "env -u X ssh other ls",
             # I2.6 heredoc detection: arithmetic shift is not a heredoc; delimiters with - and .
             "echo $((1<<2))\nssh other ls", "cat <<my-end.x\nbody\nmy-end.x\nssh other ls"]
    ALLOW = ["ssh my-gpu 'nvidia-smi'", "ssh -N -L 5000:127.0.0.1:5000 my-gpu", "ssh -o ConnectTimeout=15 -o BatchMode=yes my-gpu python3 -",
             "scp x my-gpu:~/a", "ssh user@10.9.8.7 ls", "gcloud compute instances list", "echo ssh other", "git push origin main",
             "python - <<'EOF'\nprint('it''s')\nEOF", "type C:\\Users\\a b\\x.txt", "echo 'unbalanced",
             "python - <<'EOF'\nssh other\nEOF", "scp C:\\x\\f.txt MY-GPU:~/a", "ssh ME@My-Gpu ls",
             "gcloud compute ssh my-gpu", "gcloud --project x compute instances list", "timeout 30 ssh my-gpu ls", "scp -p x my-gpu:~/a",
             "gcloud compute instances describe vm", "gcloud config list", "gcloud config get-value project", "gcloud auth list",
             "gcloud version", "gcloud info", "gcloud --version", "gcloud compute instances list --format json --filter name=x",
             "gcloud compute zones list-x", "echo $((1<<2))", "x=$(ssh my-gpu ls)", "bash -c 'ssh my-gpu ls'",
             "ssh my-gpu 'cd x && nvidia-smi'", "sudo -u bob ssh my-gpu ls", "timeout -s KILL 30 ssh my-gpu ls", "make 2>&1 | tee log",
             "ssh -o ProxyJump=my-gpu 10.9.8.7 ls", "cat <<my-end.x\nssh other\nmy-end.x"]
    for c in BLOCK:
        assert verdict("Bash", {"command": c}, P), c
    for c in ALLOW:
        assert verdict("Bash", {"command": c}, P) is None, c
    assert verdict("WebSearch", {"query": "SecretSet split"}, P) and verdict("WebSearch", {"query": "label-wise attention"}, P) is None
    assert verdict("WebFetch", {"url": "http://x", "prompt": "secretset"}, P)
    assert verdict("Bash", {"command": "ssh anywhere ls"}, None) is None and verdict("Bash", {"command": "gcloud compute instances stop x"}, None)
    assert verdict("Bash", {"command": "gcloud config set project x"}, None) and verdict("Bash", {"command": "gcloud compute ssh x"}, None)
    assert verdict("Bash", None, P) is None and verdict("Bash", {"command": 5}, P) is None
    # I1: an invalid pack still gives its lists; an unparseable one blocks every ssh
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "project.json"
        f.write_text(json.dumps({"server": {"allow_hosts": ["my-gpu"]}, "egress": {"forbidden_terms": ["secretset"]}}), encoding="utf-8")
        q = pack_for(f)
        assert verdict("Bash", {"command": "ssh other ls"}, q) and verdict("Bash", {"command": "ssh my-gpu ls"}, q) is None
        assert verdict("WebSearch", {"query": "secretset"}, q)
        f.write_text("{", encoding="utf-8")
        q = pack_for(f)
        assert q == {} and verdict("Bash", {"command": "ssh my-gpu ls"}, q) and verdict("Bash", {"command": "gcloud compute instances stop x"}, q)
        f.write_text(json.dumps({"server": {"allow_hosts": "my-gpu"}, "egress": []}), encoding="utf-8")
        assert verdict("Bash", {"command": "ssh my-gpu ls"}, pack_for(f)) and verdict("WebSearch", {"query": "x"}, pack_for(f)) is None
    print("guard selftest ok")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    cmd = ""
    try:
        d = json.loads(sys.stdin.read())
        tool, ti = d.get("tool_name", ""), d.get("tool_input", {})
        cmd = str(ti.get("command", "")) if isinstance(ti, dict) else ""
        import phro
        f = phro.find_pack(os.environ.get("CLAUDE_PROJECT_DIR") or d.get("cwd"))
        r = verdict(tool, ti, pack_for(f) if f else None)
    except Exception as e:  # a crash must not wave a risky call through
        print(f"guard error: {e}", file=sys.stderr)
        sys.exit(2 if re.search(r"gcloud|ssh|scp|rsync", cmd) else 0)
    if r:
        print(r, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
