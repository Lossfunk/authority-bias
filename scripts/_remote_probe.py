import json, os, subprocess, sys
print("== whoami ==")
print(subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip())
print("== hostname ==")
print(subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip())
print("== gpu ==")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
    print(out.strip())
except Exception as e:
    print("nvidia-smi err:", e)
print("== cwd ==")
print(os.getcwd())
print("== persona-vectors top ==")
for entry in sorted(os.listdir("persona-vectors")):
    print(entry)
print("== git branch + head ==")
print(subprocess.run(["git", "-C", "persona-vectors", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip())
print(subprocess.run(["git", "-C", "persona-vectors", "log", "--oneline", "-5"], capture_output=True, text=True).stdout.strip())
print("== exp19/principled_correction_w1 model dirs ==")
import pathlib
root = pathlib.Path("persona-vectors/neurips-results/exp19/principled_correction_w1")
if root.is_dir():
    for d in sorted(root.iterdir()):
        if d.is_dir():
            print(d.name, "->", sorted(p.name for p in d.iterdir()))
print("== presence of ab_project_out_allcond / c1_preservation ==")
for cand in [
    "persona-vectors/causal-deconfound/qwen35_ab_project_out_allcond",
    "persona-vectors/causal-deconfound/olmo2_ab_project_out_allcond",
    "persona-vectors/causal-deconfound/olmo31_ab_project_out_allcond",
    "persona-vectors/causal-deconfound/gpt_oss_ab_project_out_allcond",
    "persona-vectors/causal-deconfound/gemma4_ab_project_out_allcond",
    "persona-vectors/neurips-results/exp19/c1_preservation",
]:
    print(cand, "exists" if os.path.isdir(cand) else "MISSING")
print("== existing src.exp19 entry points ==")
out = subprocess.run(["ls", "persona-vectors/src/exp19"], capture_output=True, text=True).stdout
print(out)
