import subprocess
print("== running run_assistant_axis processes ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep"], capture_output=True, text=True).stdout)
print("== output_root_quick contents (if any) ==")
print(subprocess.run(["bash","-lc","ls -la /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond_quick 2>/dev/null"], capture_output=True, text=True).stdout)
print("== gemma quick log tail ==")
print(subprocess.run(["bash","-lc","tail -n 20 /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4_quick.log 2>/dev/null"], capture_output=True, text=True).stdout)
print("== gemma full log tail (the one running now) ==")
print(subprocess.run(["bash","-lc","tail -n 5 /home/persona-vectors/logs/exp19_ab_project_out_allcond_gemma4.log 2>/dev/null"], capture_output=True, text=True).stdout)
