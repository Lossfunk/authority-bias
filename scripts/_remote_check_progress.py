import os, subprocess
os.chdir("persona-vectors")
log = "logs/exp19_ab_project_out_allcond_olmo2_olmo31.log"
print("== ps ==")
print(subprocess.run(["bash", "-lc", "ps -ef | grep run_assistant_axis_causal_deconfound | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== log tail (60 lines) ==")
print(subprocess.run(["tail", "-n", "60", log], capture_output=True, text=True).stdout)
print("== output tree so far ==")
print(subprocess.run(["bash", "-lc", "find neurips-results/exp19/ab_project_out_allcond -maxdepth 4 -type f | head -80"], capture_output=True, text=True).stdout)
