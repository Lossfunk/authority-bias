import subprocess, json
out = subprocess.run(["bash","-lc","cat /home/persona-vectors/neurips-results/exp19/ab_project_out_allcond_quick/gpt_oss/trivia_authority/steering_summary.json"], capture_output=True, text=True).stdout
data = json.loads(out)
print(f"Total entries: {len(data)}")
print()
print(f"{'cond':<10} {'alpha':<6} {'parsed':<8} {'acc':<8} {'wrong':<8} {'flip':<8}")
for e in data:
    flip = e.get('flip_rate')
    flip_str = f"{flip:.3f}" if flip is not None else "—"
    print(f"{e['condition']:<10} {e['alpha']:<6} {e['parsed']:<8} {e['accuracy']:<8.3f} {e['wrong_rate']:<8.3f} {flip_str:<8}")
