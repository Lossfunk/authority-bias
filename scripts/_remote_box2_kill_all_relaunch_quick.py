import subprocess, time, os
print("== current procs ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test|nohup' | grep -v grep"], capture_output=True, text=True).stdout)
print("== quick log status ==")
print(subprocess.run(["bash","-lc","ls -la /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4*.log 2>/dev/null"], capture_output=True, text=True).stdout)
print("== gemma full log tail (last 5) ==")
print(subprocess.run(["bash","-lc","tail -n 3 /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4.log"], capture_output=True, text=True).stdout)
print("== gemma quick log tail (last 5) ==")
print(subprocess.run(["bash","-lc","tail -n 5 /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4_quick.log 2>/dev/null"], capture_output=True, text=True).stdout)

# Kill ALL gemma processes hard
print("== killing all run_assistant_axis / run_steering_test ==")
print(subprocess.run(["bash","-lc","pkill -9 -f run_assistant_axis_causal_deconfound; sleep 1; pkill -9 -f run_steering_test; sleep 2; ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep || echo 'all killed'"], capture_output=True, text=True).stdout)
print("== gpu after kill ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)

# Relaunch quick-only with clear new dir
log = "/home/persona-vectors/logs/exp19_gemma4_quick_v2.log"
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
    "--allow-missing-frozen-settings "
    "--run-trivia --dedupe-alpha-zero --no-compile --execute "
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond_quick/checkpoint_gemma4.json"
)
cmd = f"nohup bash -lc 'cd /home/persona-vectors && {inner}' > {log} 2>&1 & echo $!"
print(">>", cmd)
out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
print("rc", out.returncode, "pid", out.stdout.strip())
time.sleep(5)
print(subprocess.run(["bash","-lc",f"tail -n 15 {log}"], capture_output=True, text=True).stdout)
