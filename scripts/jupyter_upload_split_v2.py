"""Robust chunked uploader: small chunks via Jupyter contents API with retry/backoff and a persistent requests Session."""
from __future__ import annotations
import argparse, base64, hashlib, json, os, sys, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=10, backoff_factor=1.0, status_forcelist=(500, 502, 503, 504, 522, 524), allowed_methods=("PUT", "POST", "DELETE", "GET"))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def put_chunk(session: requests.Session, server: str, token: str, remote_path: str, blob: bytes, attempts: int = 6) -> None:
    payload = json.dumps({"type": "file", "format": "base64", "content": base64.b64encode(blob).decode()}).encode()
    url = f"{server.rstrip('/')}/api/contents/{remote_path}"
    last = None
    for k in range(attempts):
        try:
            r = session.put(url, params={"token": token}, data=payload, headers={"Content-Type": "application/json"}, timeout=600)
            if r.status_code in (200, 201):
                return
            last = (r.status_code, r.text[:300])
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout, BrokenPipeError, OSError) as e:
            last = ("EXC", repr(e))
        backoff = min(30.0, 2.0 ** k)
        print(f"    retry {k+1}/{attempts} after {backoff:.1f}s: {last}", flush=True)
        time.sleep(backoff)
    raise SystemExit(f"failed after {attempts} attempts: {last}")


def kernel_run(server: str, token: str, code: str, timeout: float = 900.0) -> tuple[int, str]:
    import websocket  # type: ignore
    import uuid as _uuid
    base = server.rstrip("/")
    headers = {"Authorization": f"token {token}"}
    r = requests.post(f"{base}/api/kernels", params={"token": token}, json={}, headers=headers, timeout=30)
    r.raise_for_status()
    kernel_id = r.json()["id"]
    ws_url = f"{'wss' if base.startswith('https') else 'ws'}://{base.split('://',1)[1]}/api/kernels/{kernel_id}/channels?token={token}"
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
            requests.delete(f"{base}/api/kernels/{kernel_id}", params={"token": token}, headers=headers, timeout=30)
    return rc, "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--remote-relative", required=True)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--start-from", type=int, default=0, help="resume from chunk index N")
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

    rc, out = kernel_run(args.server, args.token,
        "import os\n"
        f"os.makedirs({('/home/' + chunk_dir)!r}, exist_ok=True)\n"
        "print('chunk_dir ok')\n", timeout=60)
    print(out.strip())
    if rc: return 1

    sess = make_session()
    t0 = time.time(); sent = args.start_from * chunk_size
    with open(local, "rb") as f:
        if args.start_from:
            f.seek(args.start_from * chunk_size)
        for i in range(args.start_from, n):
            blob = f.read(chunk_size)
            if not blob: break
            piece_remote = f"{chunk_dir}/part_{i:05d}.bin"
            t = time.time()
            put_chunk(sess, args.server, args.token, piece_remote, blob)
            dt = time.time() - t
            sent += len(blob)
            elapsed = time.time() - t0
            avg = sent / max(elapsed, 1e-6) / 1e6
            eta = (size - sent) / max(sent / max(elapsed, 1e-6), 1e-6)
            print(f"  {i+1}/{n} {len(blob)/1e6:.1f} MB in {dt:.1f}s ({len(blob)/dt/1e6:.1f} MB/s, avg {avg:.1f} MB/s, ETA {eta:.0f}s)", flush=True)

    code = (
        "import os, hashlib, glob, shutil\n"
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
        "shutil.rmtree(chunk_dir, ignore_errors=True); print('cleaned chunks')\n"
    )
    rc, out = kernel_run(args.server, args.token, code, timeout=900)
    print(out)
    if digest not in out: print("SHA mismatch!"); return 3
    print("OK"); return 0


if __name__ == "__main__":
    sys.exit(main())
