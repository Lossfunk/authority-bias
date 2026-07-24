"""Upload a single file to a remote Jupyter server via one persistent kernel.

Uses a single kernel + WebSocket connection across all chunks for maximum throughput.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, sys, time, uuid
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

CHUNK = 32 * 1024 * 1024  # 32 MB raw -> ~42 MB base64 per request


def main() -> int:
    import websocket  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--chunk-mb", type=int, default=32)
    args = ap.parse_args()

    local = Path(args.local)
    if not local.is_file():
        print("missing", local); return 2

    chunk_size = args.chunk_mb * 1024 * 1024
    size = local.stat().st_size
    sha = hashlib.sha256()
    with open(local, "rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(blk)
    digest = sha.hexdigest()
    print(f"[upload] {local} ({size/1e9:.2f} GB, sha256={digest[:12]}) -> {args.remote}")

    base = args.server.rstrip("/")
    headers = {"Authorization": f"token {args.token}"}

    # Start kernel
    req = Request(f"{base}/api/kernels?{urlencode({'token': args.token})}", method="POST", data=b"{}", headers={**headers, "Content-Type": "application/json"})
    with urlopen(req, timeout=30) as r:
        kernel_id = json.load(r)["id"]
    print(f"[kernel] started {kernel_id}", flush=True)

    ws_url = f"{'wss' if base.startswith('https') else 'ws'}://{base.split('://',1)[1]}/api/kernels/{kernel_id}/channels?{urlencode({'token': args.token})}"
    ws = websocket.create_connection(ws_url, timeout=30)
    session_id = uuid.uuid4().hex

    def execute(code: str, timeout: float = 600.0) -> tuple[int, str]:
        msg = {
            "header": {"msg_id": uuid.uuid4().hex, "username": "agent", "session": session_id, "msg_type": "execute_request", "version": "5.3", "date": ""},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "channel": "shell", "buffers": [],
        }
        ws.send(json.dumps(msg))
        parent = msg["header"]["msg_id"]
        deadline = time.time() + timeout
        rc = 0; out = []
        while time.time() < deadline:
            ws.settimeout(max(1.0, deadline - time.time()))
            try: raw = ws.recv()
            except Exception: continue
            if not raw: continue
            data = json.loads(raw)
            if (data.get("parent_header") or {}).get("msg_id") != parent:
                continue
            mt = data.get("msg_type"); content = data.get("content", {})
            if mt == "stream":
                out.append(content.get("text",""))
            elif mt in ("display_data","execute_result"):
                out.append(((content.get("data") or {}).get("text/plain") or "") + "\n")
            elif mt == "error":
                out.append("\n".join(content.get("traceback", [])))
                rc = 1
            elif mt == "status" and content.get("execution_state") == "idle":
                break
        return rc, "".join(out)

    try:
        rc, out = execute(
            "import os, base64\n"
            f"REMOTE = {args.remote!r}\n"
            "os.makedirs(os.path.dirname(REMOTE), exist_ok=True)\n"
            "F = open(REMOTE, 'wb')\n"
            "print('init ok', REMOTE)\n",
            timeout=60,
        )
        print(out)
        if rc != 0: raise SystemExit("init failed")

        n_chunks = (size + chunk_size - 1) // chunk_size
        t_start = time.time()
        sent = 0
        with open(local, "rb") as f:
            for i in range(n_chunks):
                blob = f.read(chunk_size)
                b64 = base64.b64encode(blob).decode()
                t0 = time.time()
                rc, out = execute(
                    f"F.write(base64.b64decode({b64!r}))\n"
                    "F.flush()\n"
                    f"print('chunk {i+1}/{n_chunks} {len(blob)} bytes ok')\n",
                    timeout=600,
                )
                dt = time.time() - t0
                sent += len(blob)
                speed = sent / max(time.time() - t_start, 1e-6) / 1e6
                inst = len(blob) / max(dt, 1e-6) / 1e6
                eta = (size - sent) / max(sent / max(time.time() - t_start, 1e-6), 1e-6)
                print(f"  {i+1}/{n_chunks}: {len(blob)/1e6:.1f} MB in {dt:.1f}s ({inst:.1f} MB/s, avg {speed:.1f} MB/s, ETA {eta:.0f}s)", flush=True)
                if rc != 0:
                    print("ERR:", out[-2000:]); raise SystemExit("append failed")

        rc, out = execute(
            "F.close()\n"
            "import hashlib, os\n"
            "h = hashlib.sha256()\n"
            "with open(REMOTE, 'rb') as g:\n"
            "    for blk in iter(lambda: g.read(1<<20), b''):\n"
            "        h.update(blk)\n"
            "print('final size', os.path.getsize(REMOTE))\n"
            "print('final sha256', h.hexdigest())\n",
            timeout=300,
        )
        print(out)
        if digest not in out:
            raise SystemExit("sha256 mismatch")
        print("OK")
    finally:
        try: ws.close()
        finally:
            urlopen(Request(f"{base}/api/kernels/{kernel_id}?{urlencode({'token': args.token})}", method="DELETE", headers=headers), timeout=30).read()
            print(f"[kernel] stopped {kernel_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
