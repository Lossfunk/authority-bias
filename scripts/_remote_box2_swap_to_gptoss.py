import subprocess, time
print("== before kill ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep"], capture_output=True, text=True).stdout)
print("== killing all gemma processes ==")
print(subprocess.run(["bash","-lc","pkill -9 -f run_assistant_axis_causal_deconfound; sleep 2; pkill -9 -f run_steering_test; sleep 5"], capture_output=True, text=True).stdout)
print("== after kill ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep || echo all_dead"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)
print("== wait 15s for memory release ==")
time.sleep(15)
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)

# Launch GPT-OSS quick
log = "/home/persona-vectors/logs/exp19_gpt_oss_quick_final.log"
inner = (
    "/home/persona-vectors/.venv/bin/python -m src.exp19.run_assistant_axis_causal_deconfound "
    "--models gpt_oss "
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
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond_quick/checkpoint_gpt_oss.json"
)
cmd = f"nohup bash -lc 'cd /home/persona-vectors && {inner}' > {log} 2>&1 & echo $!"
print(">>", cmd)
out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
print("rc", out.returncode, "pid", out.stdout.strip())
time.sleep(10)
print("\n== log tail ==")
print(subprocess.run(["bash","-lc",f"tail -n 30 {log}"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"], capture_output=True, text=True).stdout)
