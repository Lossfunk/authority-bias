import os, subprocess, time
os.chdir("persona-vectors")
os.makedirs("logs", exist_ok=True)
log = "logs/exp19_ab_project_out_allcond_olmo31_v2.log"

# Use note_start (valid mode) instead of endorsement_start (which the steering runner doesn't accept).
inner = (
    ".venv/bin/python -m src.exp19.run_assistant_axis_causal_deconfound "
    "--models olmo31 "
    "--variants authority,assistant,residualized "
    "--output-root neurips-results/exp19/ab_project_out_allcond "
    "--trivia-alpha-override 1.0 "
    "--trivia-conditions N0_note,W1_note,C1_note "
    "--trivia-intervention-mode project_out_direction "
    "--trivia-norm-scaling none "
    "--trivia-position-mode note_start "
    "--allow-missing-frozen-settings "
    "--run-trivia --dedupe-alpha-zero --no-compile --execute "
    "--checkpoint-path neurips-results/exp19/ab_project_out_allcond/checkpoint_olmo31_v2.json"
)
cmd = f"nohup bash -lc 'cd /home/persona-vectors && {inner}' > {log} 2>&1 & echo $!"
print(">>", cmd)
out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
print("rc", out.returncode, "stdout", out.stdout.strip(), "stderr", out.stderr.strip())
time.sleep(2)
print("\n== log head ==")
print(subprocess.run(["tail", "-n", "20", log], capture_output=True, text=True).stdout)
