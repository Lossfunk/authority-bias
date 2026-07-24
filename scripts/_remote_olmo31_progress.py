import os, subprocess
print("== ps for olmo31 v2 ==")
print(subprocess.run(["bash","-lc","ps -ef | grep run_assistant_axis_causal_deconfound | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== log tail ==")
print(subprocess.run(["bash","-lc","tail -n 60 /home/persona-vectors/logs/exp19_ab_project_out_allcond_olmo31_v2.log"], capture_output=True, text=True).stdout)
print("== output dirs (olmo31) ==")
print(subprocess.run(["bash","-lc","find /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond/olmo31 -maxdepth 4 -type f 2>/dev/null | sort | head -40"], capture_output=True, text=True).stdout)
