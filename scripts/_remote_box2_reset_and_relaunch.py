import os, subprocess, time
print("== confirm gpu is empty ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== any python procs holding gpu? ==")
print(subprocess.run(["bash","-lc","fuser -v /dev/nvidia* 2>&1 | head; ps -ef | grep -E 'python|torch' | grep -v grep | head"], capture_output=True, text=True).stdout)
print("== reset GPU if still busy ==")
print(subprocess.run(["bash","-lc","nvidia-smi --gpu-reset 2>&1 | head -3 || true"], capture_output=True, text=True).stdout)

# Relaunch Gemma quick (smaller batch size for safety)
log = "/home/persona-vectors/logs/exp19_gemma4_quick_v3.log"
inner = (
    "/home/persona-vectors/.venv/bin/python -m src.exp19.run_assistant_axis_causal_deconfound "
    "--models gemma4 "
    "--variants authority,assistant,residualized "
    "--output-root neurips-results/exp19/ab_project_out_allcond_quick "
    "--trivia-alpha-override 1.0 "
    "--trivia-conditions N0_note,W1_note,C1_note "
    "--trivia-intervention-mode project_out_direction "
    "--trivia-norm-scaling none "
    "--trivia-max-items 200 "
    "--trivia-batch-size 4 "
    "--allow-missing-frozen-settings "
    "--run-trivia --dedupe-alpha-zero --no-compile --execute "
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond_quick/checkpoint_gemma4.json"
)
cmd = f"nohup bash -lc 'cd /home/persona-vectors && {inner}' > {log} 2>&1 & echo $!"
print(">>", cmd)
out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
print("rc", out.returncode, "pid", out.stdout.strip())
time.sleep(10)
print("\n== log tail ==")
print(subprocess.run(["bash","-lc",f"tail -n 25 {log}"], capture_output=True, text=True).stdout)
print("== gpu after ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
