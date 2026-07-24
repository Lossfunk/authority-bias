"""Upload a big file as many small remote pieces (Jupyter contents API), then ask the
remote kernel to concatenate them and verify SHA-256.

This avoids WebSocket frame-size limits.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, sys, time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request


def put_contents(server: str, token: str, remote_path: str, blob: bytes) -> None:
    body = json.dumps({"type": "file", "format": "base64", "content": base64.b64encode(blob).decode()}).encode()
    url = f"{server.rstrip('/')}/api/contents/{quote(remote_path)}?{urlencode({'token': token})}"
    req = Request(url, data=body, method="PUT", headers={"Content-Type": "application/json", "Authorization": f"token {token}"})
    with urlopen(req, timeout=600) as r:
        r.read()


def kernel_run(server: str, token: str, code: str, timeout: float = 600.0) -> tuple[int, str]:
    import websocket  # type: ignore
    import uuid as _uuid
    base = server.rstrip("/")
    headers = {"Authorization": f"token {token}"}
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
    rc = 0; out = []
    try:
        while time.time() < deadline:
            ws.settimeout(max(1.0, deadline - time.time()))
            try: raw = ws.recv()
            except Exception: continue
            if not raw: continue
            data = json.loads(raw)
            if (data.get("parent_header") or {}).get("msg_id") != parent: continue
            mt = data.get("msg_type"); content = data.get("content", {})
            if mt == "stream": out.append(content.get("text",""))
            elif mt in ("display_data","execute_result"): out.append(((content.get("data") or {}).get("text/plain") or "") + "\n")
            elif mt == "error": out.append("\n".join(content.get("traceback", []))); rc = 1
            elif mt == "status" and content.get("execution_state") == "idle": break
    finally:
        try: ws.close()
        finally:
            urlopen(Request(f"{base}/api/kernels/{kernel_id}?{urlencode({'token': token})}", method="DELETE", headers=headers), timeout=30).read()
    return rc, "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True, help="absolute path on the remote container, e.g. /home/persona-vectors/...")
    ap.add_argument("--remote-relative", required=True, help="contents-API-relative path (no leading slash) for chunk drops, e.g. persona-vectors/_uploads")
    ap.add_argument("--chunk-mb", type=int, default=16)
    args = ap.parse_args()

    local = Path(args.local)
    if not local.is_file(): print("missing", local); return 2
    chunk_size = args.chunk_mb * 1024 * 1024
    size = local.stat().st_size
    sha = hashlib.sha256()
    with open(local, "rb") as f:
        for blk in iter(lambda: f.read(1024*1024), b""): sha.update(blk)
    digest = sha.hexdigest()
    print(f"[upload] {local} ({size/1e9:.2f} GB sha256={digest[:12]})")

    n = (size + chunk_size - 1) // chunk_size
    chunk_dir = args.remote_relative.rstrip("/") + "/" + Path(local).stem + "_" + digest[:8]
    print(f"[upload] chunks -> {chunk_dir}/  ({n} chunks of {chunk_size/1e6:.0f}MB)")

    # Make sure the chunk dir exists.
    rc, out = kernel_run(args.server, args.token,
        f"import os, pathlib\n"
        f"os.makedirs({('/home/' + chunk_dir)!r}, exist_ok=True)\n"
        f"print('chunk_dir ok')\n")
    print(out.strip())
    if rc: return 1

    t0 = time.time(); sent = 0
    with open(local, "rb") as f:
        for i in range(n):
            blob = f.read(chunk_size)
            piece_remote = f"{chunk_dir}/part_{i:05d}.bin"
            t = time.time()
            put_contents(args.server, args.token, piece_remote, blob)
            dt = time.time() - t
            sent += len(blob)
            avg = sent / max(time.time()-t0, 1e-6) / 1e6
            print(f"  {i+1}/{n} {len(blob)/1e6:.1f} MB in {dt:.1f}s ({len(blob)/dt/1e6:.1f} MB/s, avg {avg:.1f} MB/s, ETA {(size-sent)/(sent/(time.time()-t0)):.0f}s)", flush=True)

    # Concatenate on remote and verify
    code = (
        "import os, hashlib, glob\n"
        f"chunk_dir = '/home/{chunk_dir}'\n"
        f"REMOTE = {args.remote!r}\n"
        "os.makedirs(os.path.dirname(REMOTE), exist_ok=True)\n"
        "parts = sorted(glob.glob(os.path.join(chunk_dir, 'part_*.bin')))\n"
        "with open(REMOTE,'wb') as g:\n"
        "    for p in parts:\n"
        "        with open(p,'rb') as r:\n"
        "            g.write(r.read())\n"
        "h=hashlib.sha256()\n"
        "with open(REMOTE,'rb') as g:\n"
        "    for blk in iter(lambda: g.read(1<<20), b''): h.update(blk)\n"
        "print('final size', os.path.getsize(REMOTE))\n"
        "print('final sha256', h.hexdigest())\n"
        "import shutil; shutil.rmtree(chunk_dir, ignore_errors=True); print('cleaned chunks')\n"
    )
    rc, out = kernel_run(args.server, args.token, code, timeout=900)
    print(out)
    if digest not in out:
        print("SHA mismatch!"); return 3
    print("OK"); return 0

if __name__ == "__main__":
    sys.exit(main())
