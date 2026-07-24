import os, subprocess, time
print("== gpu before ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,memory.total --format=csv"], capture_output=True, text=True).stdout)

os.chdir("/home/persona-vectors")
os.makedirs("logs", exist_ok=True)
log = "/home/persona-vectors/logs/exp19_gpt_oss_quick_v1.log"

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
    "--allow-missing-frozen-settings "
    "--run-trivia --dedupe-alpha-zero --no-compile --execute "
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond_quick/checkpoint_gpt_oss.json"
)
# Don't launch yet if Gemma is still occupying the GPU heavily; we'll let the user/agent decide.
# Just run a dry run first to confirm it composes.
print("== dryrun (no --execute) ==")
dryrun_cmd = inner.replace(" --execute ", " ")
out = subprocess.run(dryrun_cmd, shell=True, capture_output=True, text=True, timeout=180)
print("rc", out.returncode)
print("stdout (last 4000 chars):")
print(out.stdout[-4000:])
print("stderr (last 2000 chars):")
print(out.stderr[-2000:])
