import os, json, sys
sys.path.insert(0, "persona-vectors")
print("== gpu and ps ==")
import subprocess
print(subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv && ps -p 2100 -o pid,etime,cmd 2>/dev/null || echo 'PID 2100 not running'"], capture_output=True, text=True).stdout)

print("\n== gpt_oss / gemma4 mechanism artifacts ==")
roots = {
    "gpt_oss": [
        "persona-vectors/neurips-results/gpt-oss/mechanism/assistant_axis_hardened",
        "persona-vectors/neurips-results/gpt-oss/mechanism/gpt_oss_compliance_analysis",
        "persona-vectors/neurips-results/gpt-oss/mechanism/gpt_oss_authority_activations",
    ],
    "gemma4": [
        "persona-vectors/neurips-results/gemma4/mechanism/assistant_axis_hardened",
        "persona-vectors/neurips-results/gemma4/mechanism/gemma4_compliance_analysis",
        "persona-vectors/neurips-results/gemma4/mechanism/gemma4_authority_activations",
        "persona-vectors/neurips-results/gemma4/mechanism/gemma4_authority_activations_no_thinking",
    ],
}
for name, dirs in roots.items():
    print(f"\n[{name}]")
    for d in dirs:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                size = os.path.getsize(p) if os.path.isfile(p) else None
                print("   ", size, p)
        else:
            print("   MISSING DIR:", d)

print("\n== model_specs ==")
from src.exp19.common import default_model_specs
from pathlib import Path
specs = default_model_specs(repo_root=Path("persona-vectors"))
for name in ["gpt_oss", "gemma4"]:
    spec = specs.get(name)
    if not spec:
        print(name, "no spec")
        continue
    print(name, "axis", spec.axis_path, "exists", os.path.isfile(spec.axis_path))
    print("    direction", spec.direction_path, "exists", os.path.isfile(spec.direction_path))
    print("    masks", getattr(spec, "masks_path", None), "exists", os.path.isfile(getattr(spec, "masks_path", "") or ""))
    print("    extraction_dir", getattr(spec, "extraction_dir", None), "exists", os.path.isdir(getattr(spec, "extraction_dir", "") or ""))
