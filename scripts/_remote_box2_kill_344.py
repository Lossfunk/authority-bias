import subprocess, time
print("== before ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep"], capture_output=True, text=True).stdout)
print("== killing PID 344 with SIGKILL ==")
print(subprocess.run(["bash","-lc","kill -9 344 2>&1; sleep 2; kill -9 2088 2>&1; sleep 1; pkill -9 -f 'run_steering_test' 2>&1; sleep 5"], capture_output=True, text=True).stdout)
print("== after ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep || echo 'all dead'"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== wait 15s for memory release ==")
time.sleep(15)
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)

# Now relaunch Gemma quick fresh
log = "/home/persona-vectors/logs/exp19_gemma4_quick_final.log"
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
import time as _t
_t.sleep(8)
print(subprocess.run(["bash","-lc",f"tail -n 8 {log}"], capture_output=True, text=True).stdout)
