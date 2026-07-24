"""Download small mechanism artifacts for OLMo-2/3.1 from the remote notebook server."""
from __future__ import annotations
import argparse, base64, json, sys
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request

REMOTE = [
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/summary.json",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/directions.pt",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/primary_direction.pt",
    "neurips-results/olmo2/mechanism/assistant_axis_hardened/assistant_axis.pt",
    "neurips-results/olmo2/mechanism/assistant_axis_hardened/assistant_axis_layer.pt",
    "neurips-results/olmo2/mechanism/assistant_axis_hardened/overlap_summary.json",
    "neurips-results/olmo2/mechanism/olmo2_authority_activations/label_masks.json",
    "neurips-results/olmo2/mechanism/olmo2_authority_activations/metadata.jsonl",
    "neurips-results/olmo2/mechanism/olmo2_authority_activations/summary.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/summary.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/directions.pt",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/primary_direction.pt",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/null_controls.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/probe_results.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/projection_analysis.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/cosine_similarities.json",
    "neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/layer_position_sweep.json",
    "neurips-results/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/label_masks.json",
    "neurips-results/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/metadata.jsonl",
    "neurips-results/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/summary.json",
]

# OLMo-2 also has compliance_analysis (separate path from olmo31). Add equivalent files where they exist.
REMOTE += [
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/null_controls.json",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/probe_results.json",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/projection_analysis.json",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/cosine_similarities.json",
    "neurips-results/olmo2/mechanism/olmo2_compliance_analysis/layer_position_sweep.json",
]

def fetch(server: str, token: str, remote_rel: str, dest: Path) -> tuple[bool, int]:
    path = "persona-vectors/" + remote_rel
    url = f"{server.rstrip('/')}/api/contents/{quote(path)}?{urlencode({'token': token})}"
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=120) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  ERR {remote_rel}: {e}")
        return False, 0
    fmt = data.get("format")
    content = data.get("content")
    if content is None:
        print(f"  EMPTY {remote_rel}")
        return False, 0
    if fmt == "base64":
        blob = base64.b64decode(content)
    elif fmt == "json":
        blob = json.dumps(content, indent=2).encode("utf-8")
    else:
        blob = content.encode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return True, len(blob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    total = 0
    ok = 0
    for rel in REMOTE:
        dest = root / rel
        success, size = fetch(args.server, args.token, rel, dest)
        if success:
            ok += 1
            total += size
            print(f"  ok {rel} ({size} bytes)")
    print(f"\ndownloaded {ok}/{len(REMOTE)} files, {total} bytes")

if __name__ == "__main__":
    main()
