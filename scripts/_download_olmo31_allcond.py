import json, base64
import os
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
SERVER = os.environ["JUPYTER_SERVER_URL"]
TOKEN = os.environ["JUPYTER_TOKEN"]
REL = []
for variant in ["trivia_authority","trivia_assistant","trivia_residualized"]:
    for fn in ["steering_summary.json","steering_meta.json","steering_rows.jsonl"]:
        REL.append(f"neurips-results/exp19/ab_project_out_allcond/olmo31/{variant}/{fn}")
REL.append("neurips-results/exp19/ab_project_out_allcond/checkpoint_olmo31_v2.json")
for rel in REL:
    url = f"{SERVER.rstrip('/')}/api/contents/{quote('persona-vectors/' + rel)}?{urlencode({'token': TOKEN})}"
    try:
        with urlopen(Request(url, headers={"Accept":"application/json"}), timeout=120) as r:
            d = json.load(r)
    except Exception as e: print("ERR",rel,e); continue
    fmt=d.get("format"); c=d.get("content")
    if c is None: print("EMPTY",rel); continue
    if fmt=="base64": blob=base64.b64decode(c)
    elif fmt=="json": blob=json.dumps(c, indent=2).encode("utf-8")
    else: blob=c.encode("utf-8")
    dest=ROOT/rel; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(blob)
    print("ok",rel,len(blob))
