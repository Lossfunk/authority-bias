"""Execute a Python snippet on a remote Jupyter server via the kernel WebSocket API.

Usage:
    python scripts/jupyter_run.py --server URL --token TOKEN --code-file path.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from urllib.parse import urlparse

import requests
import websocket


def _ws_url(server: str) -> str:
    p = urlparse(server)
    scheme = "wss" if p.scheme == "https" else "ws"
    return f"{scheme}://{p.netloc}{p.path.rstrip('/')}"


def _make_msg(msg_type: str, content: dict, session: str) -> dict:
    return {
        "header": {
            "msg_id": uuid.uuid4().hex,
            "username": "agent",
            "session": session,
            "msg_type": msg_type,
            "version": "5.3",
            "date": "",
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "channel": "shell",
        "buffers": [],
    }


def run(server: str, token: str, code: str, timeout: float = 600.0) -> int:
    base = server.rstrip("/")
    sess = requests.Session()
    sess.headers.update({"Authorization": f"token {token}"})

    # Start a kernel.
    r = sess.post(f"{base}/api/kernels", params={"token": token}, timeout=30)
    r.raise_for_status()
    kernel_id = r.json()["id"]
    print(f"[kernel] started {kernel_id}", flush=True)

    ws_base = _ws_url(base)
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?token={token}"
    ws = websocket.create_connection(ws_url, timeout=30)
    session_id = uuid.uuid4().hex
    msg = _make_msg("execute_request", {
        "code": code,
        "silent": False,
        "store_history": False,
        "user_expressions": {},
        "allow_stdin": False,
        "stop_on_error": True,
    }, session_id)
    ws.send(json.dumps(msg))
    parent_id = msg["header"]["msg_id"]

    deadline = time.time() + timeout
    rc = 0
    last_status = None
    try:
        while time.time() < deadline:
            ws.settimeout(max(1.0, deadline - time.time()))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                continue
            data = json.loads(raw)
            ph = data.get("parent_header", {}) or {}
            if ph.get("msg_id") != parent_id:
                continue
            mt = data.get("msg_type")
            content = data.get("content", {})
            if mt in ("stream",):
                sys.stdout.write(content.get("text", ""))
                sys.stdout.flush()
            elif mt in ("display_data", "execute_result"):
                txt = (content.get("data") or {}).get("text/plain", "")
                if txt:
                    print(txt, flush=True)
            elif mt == "error":
                print("\n".join(content.get("traceback", [])), flush=True)
                rc = 1
            elif mt == "status":
                last_status = content.get("execution_state")
                if last_status == "idle":
                    break
    finally:
        try:
            ws.close()
        finally:
            sess.delete(f"{base}/api/kernels/{kernel_id}", params={"token": token}, timeout=30)
            print(f"\n[kernel] stopped {kernel_id} (rc={rc}, last_status={last_status})", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--code-file", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    code = open(args.code_file, "r", encoding="utf-8").read()
    return run(args.server, args.token, code, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
