import subprocess
print("== trivia_authority steering_summary.json ==")
print(subprocess.run(["bash","-lc","cat /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond_quick/gpt_oss/trivia_authority/steering_summary.json"], capture_output=True, text=True).stdout)
print("\n== trivia_authority steering_meta.json ==")
print(subprocess.run(["bash","-lc","cat /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond_quick/gpt_oss/trivia_authority/steering_meta.json"], capture_output=True, text=True).stdout)
