import os, subprocess, sys
os.chdir("persona-vectors")
cmd = [
    ".venv/bin/python", "-m", "src.exp19.run_assistant_axis_causal_deconfound",
    "--models", "olmo2",
    "--variants", "authority,assistant,residualized",
    "--output-root", "neurips-results/exp19/ab_project_out_allcond",
    "--trivia-alpha-override", "1.0",
    "--trivia-conditions", "N0_note,W1_note,C1_note",
    "--trivia-intervention-mode", "project_out_direction",
    "--trivia-norm-scaling", "none",
    "--run-trivia",
    "--dedupe-alpha-zero",
    "--no-compile",
]
print(">>", " ".join(cmd))
out = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
print("rc", out.returncode)
print("stdout (last 4000 chars):")
print(out.stdout[-4000:])
print("stderr (last 4000 chars):")
print(out.stderr[-4000:])
