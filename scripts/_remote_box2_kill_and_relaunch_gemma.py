import os, subprocess, time
print("== killing PID 296 ==")
print(subprocess.run(["bash","-lc","kill 296 2>&1; sleep 3; pkill -9 -f run_assistant_axis_causal_deconfound 2>&1; sleep 2; pkill -9 -f run_steering_test 2>&1; sleep 2; ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu free? ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)

os.chdir("/home/persona-vectors")
os.makedirs("logs", exist_ok=True)
log = "logs/exp19_ab_project_out_allcond_gemma4_quick.log"

inner = (
    ".venv/bin/python -m src.exp19.run_assistant_axis_causal_deconfound "
    "--models gemma4 "
    "--variants authority,assistant,residualized "
    "--output-root neurips-results/exp19/ab_project_out_allcond_quick "
    "--trivia-alpha-override 1.0 "
    "--trivia-conditions N0_note,W1_note,C1_note "
    "--trivia-intervention-mode project_out_direction "
    "--trivia-norm-scaling none "
    "--trivia-max-items 200 "
    "--allow-missing-frozen-settings "
    "--run-trivia --dedupe-alpha-zero --no-compile --execute "
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond_quick/checkpoint_gemma4.json"
)
cmd = f"nohup bash -lc 'cd /home/persona-vectors && {inner}' > {log} 2>&1 & echo $!"
print(">>", cmd)
out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
print("rc", out.returncode, "pid", out.stdout.strip())
time.sleep(3)
print(subprocess.run(["bash","-lc",f"tail -n 5 {log}"], capture_output=True, text=True).stdout)
