"""Upload local files to a remote Jupyter server via the kernel for files >100MB.

Strategy:
- For small files (< 50 MB), use the contents PUT API directly with base64.
- For large files, kick off a background download from the kernel side (the kernel runs
  on the box and can pull from a file we serve over a temporary HTTP-like channel).
  Since direct write of huge bodies via PUT can be flaky, we instead ship the file
  in fixed-size base64 chunks, telling the kernel to assemble it.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, sys, time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request

CHUNK = 8 * 1024 * 1024  # 8 MB raw -> ~10.7 MB base64 per request


def kernel_run(server: str, token: str, code: str, timeout: float = 1800.0) -> tuple[int, str]:
    """Execute a Python snippet on the remote Jupyter kernel."""
    import websocket  # type: ignore
    import uuid as _uuid

    base = server.rstrip("/")
    headers = {"Authorization": f"token {token}"}
    # Start a kernel.
    req = Request(f"{base}/api/kernels?{urlencode({'token': token})}", method="POST", data=b"{}", headers={**headers, "Content-Type": "application/json"})
    with urlopen(req, timeout=30) as r:
        kernel_id = json.load(r)["id"]
    ws_url = f"{'wss' if base.startswith('https') else 'ws'}://{base.split('://',1)[1]}/api/kernels/{kernel_id}/channels?{urlencode({'token': token})}"
    ws = websocket.create_connection(ws_url, timeout=30)
    session_id = _uuid.uuid4().hex
    msg = {
        "header": {"msg_id": _uuid.uuid4().hex, "username": "agent", "session": session_id, "msg_type": "execute_request", "version": "5.3", "date": ""},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell", "buffers": [],
    }
    ws.send(json.dumps(msg))
    parent = msg["header"]["msg_id"]
    deadline = time.time() + timeout
    rc = 0
    out_buf = []
    try:
        while time.time() < deadline:
            ws.settimeout(max(1.0, deadline - time.time()))
            try:
                raw = ws.recv()
            except Exception:
                continue
            if not raw:
                continue
            data = json.loads(raw)
            if (data.get("parent_header") or {}).get("msg_id") != parent:
                continue
            mt = data.get("msg_type"); content = data.get("content", {})
            if mt == "stream":
                out_buf.append(content.get("text", ""))
            elif mt in ("display_data", "execute_result"):
                out_buf.append(((content.get("data") or {}).get("text/plain") or "") + "\n")
            elif mt == "error":
                out_buf.append("\n".join(content.get("traceback", [])))
                rc = 1
            elif mt == "status" and content.get("execution_state") == "idle":
                break
    finally:
        try: ws.close()
        finally:
            urlopen(Request(f"{base}/api/kernels/{kernel_id}?{urlencode({'token': token})}", method="DELETE", headers=headers), timeout=30).read()
    return rc, "".join(out_buf)


def upload_chunked(server: str, token: str, local_path: Path, remote_path: str) -> None:
    size = local_path.stat().st_size
    sha = hashlib.sha256()
    with open(local_path, "rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(blk)
    digest = sha.hexdigest()
    print(f"[upload] {local_path} -> {remote_path} ({size} bytes, sha256={digest[:12]})")

    # 1) prepare empty target file on remote
    code_init = f"""
import os, hashlib
remote = {remote_path!r}
os.makedirs(os.path.dirname(remote), exist_ok=True)
open(remote, 'wb').close()
print('init ok', remote)
"""
    rc, out = kernel_run(server, token, code_init, timeout=60)
    print(out)
    if rc != 0:
        raise SystemExit("init failed")

    # 2) stream chunks
    n_chunks = (size + CHUNK - 1) // CHUNK
    with open(local_path, "rb") as f:
        for i in range(n_chunks):
            blob = f.read(CHUNK)
            b64 = base64.b64encode(blob).decode()
            code_append = (
                "import base64\n"
                f"with open({remote_path!r}, 'ab') as g:\n"
                f"    g.write(base64.b64decode({b64!r}))\n"
                f"print('appended chunk {i+1}/{n_chunks} ({len(blob)} bytes)', flush=True)\n"
            )
            t0 = time.time()
            rc, out = kernel_run(server, token, code_append, timeout=600)
            dt = time.time() - t0
            print(f"  chunk {i+1}/{n_chunks} {len(blob)/1e6:.1f} MB in {dt:.1f}s", end="")
            sys.stdout.flush()
            if rc != 0:
                print()
                print("ERR:", out)
                raise SystemExit("append failed")
            print(f"  ({(len(blob)/dt)/1e6:.1f} MB/s)")

    # 3) verify
    code_verify = f"""
import hashlib
remote = {remote_path!r}
h = hashlib.sha256()
with open(remote, 'rb') as g:
    for blk in iter(lambda: g.read(1<<20), b''):
        h.update(blk)
import os
print('size', os.path.getsize(remote))
print('sha256', h.hexdigest())
"""
    rc, out = kernel_run(server, token, code_verify, timeout=300)
    print(out)
    if digest not in out:
        raise SystemExit(f"sha256 mismatch! expected {digest}")
    print("OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("pairs", nargs="+", help="local:remote pairs")
    args = ap.parse_args()
    for pair in args.pairs:
        local_s, remote = pair.split(":", 1)
        local = Path(local_s)
        if not local.is_file():
            print("missing", local); return 2
        upload_chunked(args.server, args.token, local, remote)
    return 0


if __name__ == "__main__":
    sys.exit(main())
