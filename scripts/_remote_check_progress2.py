import os, subprocess
os.chdir("persona-vectors")
log = "logs/exp19_ab_project_out_allcond_olmo2_olmo31.log"
print("== ps ==")
print(subprocess.run(["bash", "-lc", "ps -ef | grep run_assistant_axis_causal_deconfound | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== log tail ==")
print(subprocess.run(["bash", "-lc", f"tail -n 60 {log}"], capture_output=True, text=True).stdout)
print("== output dirs ==")
print(subprocess.run(["bash", "-lc", "find neurips-results/exp19/ab_project_out_allcond -maxdepth 4 -name '*summary.json' -o -name '*manifest.json' -o -name 'checkpoint.json' | head -40"], capture_output=True, text=True).stdout)
