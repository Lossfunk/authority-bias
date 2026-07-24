"""Box-2 (GPT-OSS / Gemma) sanity check before launching."""
import os, subprocess, sys
print("== whoami / host / gpu ==")
print(subprocess.run(["bash", "-lc", "whoami; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"], capture_output=True, text=True).stdout)

print("== top-level layout ==")
print(subprocess.run(["bash", "-lc", "ls /home 2>/dev/null; ls / 2>/dev/null"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "find / -maxdepth 4 -type d -name 'persona-vectors' 2>/dev/null | head -5"], capture_output=True, text=True).stdout)

# Try the canonical path and pivot if elsewhere
roots = ["/home/persona-vectors", "/root/persona-vectors", "/workspace/persona-vectors"]
chosen = None
for r in roots:
    if os.path.isdir(r):
        chosen = r
        break
if not chosen:
    found = subprocess.run(["bash", "-lc", "find / -maxdepth 6 -type d -name 'persona-vectors' 2>/dev/null"], capture_output=True, text=True).stdout.strip().splitlines()
    chosen = found[0] if found else None
print("repo root:", chosen)
if not chosen:
    sys.exit(0)
os.chdir(chosen)
print("cwd =", os.getcwd())

print("\n== git head ==")
print(subprocess.run(["bash", "-lc", "git log --oneline -3 2>/dev/null; git rev-parse --abbrev-ref HEAD 2>/dev/null"], capture_output=True, text=True).stdout)

print("\n== prereq files for gpt_oss / gemma4 ==")
print(subprocess.run(["bash", "-lc", "ls -la neurips-results/gpt-oss/mechanism/ 2>/dev/null"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "ls -la neurips-results/gemma4/mechanism/ 2>/dev/null"], capture_output=True, text=True).stdout)

print("\n== assistant_axis*.pt ==")
print(subprocess.run(["bash", "-lc", "find neurips-results external -maxdepth 6 -name 'assistant_axis*.pt' 2>/dev/null"], capture_output=True, text=True).stdout)

print("\n== compliance directions (.pt) ==")
print(subprocess.run(["bash", "-lc", "find neurips-results -maxdepth 6 -name 'primary_direction.pt' -o -name 'directions.pt' 2>/dev/null"], capture_output=True, text=True).stdout)

print("\n== authority activations (need activations.pt and metadata.jsonl) ==")
print(subprocess.run(["bash", "-lc", "find neurips-results -maxdepth 6 -path '*authority_activations*' -type f -name 'activations.pt' -o -name 'metadata.jsonl' 2>/dev/null | head -40"], capture_output=True, text=True).stdout)

print("\n== model_specs ==")
sys.path.insert(0, os.getcwd())
from pathlib import Path
try:
    from src.exp19.common import default_model_specs
    specs = default_model_specs(repo_root=Path("."))
    for name in ["gpt_oss", "gemma4"]:
        spec = specs.get(name)
        if not spec:
            print(name, "no spec"); continue
        print(name)
        for attr in ["axis_path", "direction_path", "masks_path", "extraction_dir"]:
            p = getattr(spec, attr, None)
            print(f"   {attr}: {p}", "exists", os.path.exists(str(p)) if p else None)
except Exception as e:
    print("err loading specs:", e)
