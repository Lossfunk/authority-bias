import os, subprocess
os.chdir("/home/persona-vectors")

print("== full gpt-oss tree ==")
print(subprocess.run(["bash","-lc","find neurips-results/gpt-oss -maxdepth 5 -type f 2>/dev/null | sort | head -40"], capture_output=True, text=True).stdout)

print("\n== gemma4 dryrun (no --execute) ==")
cmd = [
    ".venv/bin/python", "-m", "src.exp19.run_assistant_axis_causal_deconfound",
    "--models", "gemma4",
    "--variants", "authority,assistant,residualized",
    "--output-root", "neurips-results/exp19/ab_project_out_allcond",
    "--trivia-alpha-override", "1.0",
    "--trivia-conditions", "N0_note,W1_note,C1_note",
    "--trivia-intervention-mode", "project_out_direction",
    "--trivia-norm-scaling", "none",
    "--allow-missing-frozen-settings",
    "--run-trivia", "--dedupe-alpha-zero", "--no-compile",
]
out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print("rc", out.returncode)
print("stdout (last 4000 chars):")
print(out.stdout[-4000:])
print("stderr (last 2000 chars):")
print(out.stderr[-2000:])
