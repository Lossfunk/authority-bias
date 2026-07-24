import os, subprocess, sys
sys.path.insert(0, "persona-vectors")
print("== run_assistant_axis_causal_deconfound.py header ==")
print(open("persona-vectors/src/exp19/run_assistant_axis_causal_deconfound.py").read()[:6000])
print("\n== run_principled_correction.py header ==")
print(open("persona-vectors/src/exp19/run_principled_correction.py").read()[:4000])
print("\n== existing C1-preservation related runs ==")
out = subprocess.run(["bash", "-lc", "find persona-vectors/causal-deconfound -maxdepth 2 -type d | head -100"], capture_output=True, text=True).stdout
print(out)
