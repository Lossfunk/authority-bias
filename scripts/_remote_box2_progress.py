import os, subprocess
print("== ps ==")
print(subprocess.run(["bash","-lc","ps -ef | grep run_assistant_axis_causal_deconfound | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== gemma log tail ==")
print(subprocess.run(["bash","-lc","tail -n 40 /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4.log"], capture_output=True, text=True).stdout)
print("== outputs ==")
print(subprocess.run(["bash","-lc","find /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond/gemma4 -maxdepth 4 -type f 2>/dev/null"], capture_output=True, text=True).stdout)
