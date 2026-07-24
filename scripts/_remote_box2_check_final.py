import subprocess
print("== running ==")
print(subprocess.run(["bash","-lc","ps -ef | grep -E 'run_assistant_axis|run_steering_test' | grep -v grep"], capture_output=True, text=True).stdout)
print("== gpu ==")
print(subprocess.run(["bash","-lc","nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"], capture_output=True, text=True).stdout)
print("== gemma quick final tail ==")
print(subprocess.run(["bash","-lc","tail -n 25 /home/persona-vectors/logs/exp19_gemma4_quick_final.log 2>/dev/null"], capture_output=True, text=True).stdout)
