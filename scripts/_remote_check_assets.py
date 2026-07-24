import os, json
print("== assistant axes & directions on remote ==")
roots = {
    "olmo2":  "persona-vectors/neurips-results/olmo2/mechanism/assistant_axis_hardened",
    "olmo31": "persona-vectors/neurips-results/olmo31/mechanism/assistant_axis_hardened",
    "gpt_oss": "persona-vectors/neurips-results/gpt-oss/mechanism/assistant_axis_hardened",
    "gemma4": "persona-vectors/neurips-results/gemma4/mechanism/assistant_axis_hardened",
    "qwen35": "persona-vectors/neurips-results/qwen35/mechanism/assistant_axis_hardened",
}
for name, root in roots.items():
    print(name, os.listdir(root) if os.path.isdir(root) else "MISSING")

print("\n== compliance_analysis directions ==")
caroots = {
    "olmo2": "persona-vectors/neurips-results/olmo2/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
    "olmo31": "persona-vectors/neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
    "gpt_oss": "persona-vectors/neurips-results/gpt-oss/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
    "gemma4": "persona-vectors/neurips-results/gemma4/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
    "qwen35": "persona-vectors/neurips-results/qwen35/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
}
for name, root in caroots.items():
    if os.path.isdir(root):
        files = sorted(os.listdir(root))
        print(name, files[:20])
    else:
        print(name, "MISSING")

print("\n== exp19 principled_correction_w1 model dirs ==")
pcroot = "persona-vectors/neurips-results/exp19/principled_correction_w1"
print(sorted(os.listdir(pcroot)) if os.path.isdir(pcroot) else "MISSING")

print("\n== model_specs.default + axis_path config ==")
import sys
sys.path.insert(0, "persona-vectors")
try:
    from src.exp19.common import default_model_specs
    specs = default_model_specs(repo_root=__import__("pathlib").Path("persona-vectors"))
    for name, spec in specs.items():
        print(name, "axis", getattr(spec, "axis_path", None), "exists",
              os.path.isfile(getattr(spec, "axis_path", "")) if getattr(spec, "axis_path", None) else None)
        print("  direction", getattr(spec, "direction_path", None), "exists",
              os.path.isfile(getattr(spec, "direction_path", "")) if getattr(spec, "direction_path", None) else None)
except Exception as e:
    print("err loading specs", e)
