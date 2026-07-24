import os, subprocess
print("== /assistant-axis (top-level on box) ==")
print(subprocess.run(["bash","-lc","ls -la /assistant-axis 2>/dev/null && find /assistant-axis -maxdepth 6 -type f 2>/dev/null | head -100"], capture_output=True, text=True).stdout)

print("\n== /home/persona-vectors/external/assistant-axis ==")
print(subprocess.run(["bash","-lc","ls -la /home/persona-vectors/external/assistant-axis 2>/dev/null && find /home/persona-vectors/external/assistant-axis -maxdepth 6 -type f 2>/dev/null | head -100"], capture_output=True, text=True).stdout)

print("\n== anywhere else with gpt_oss + .pt or jsonl ==")
print(subprocess.run(["bash","-lc","find / -maxdepth 8 -type f \\( -name 'activations.pt' -o -name 'primary_direction.pt' -o -name 'directions.pt' -o -name 'metadata.jsonl' \\) -path '*gpt*' 2>/dev/null"], capture_output=True, text=True).stdout)

print("\n== sizes if any ==")
for p in ["/assistant-axis", "/home/persona-vectors/external/assistant-axis", "/deconfound-logs"]:
    if os.path.isdir(p):
        print(p)
        print(subprocess.run(["bash","-lc",f"du -ah --max-depth=2 {p} 2>/dev/null | head -50"], capture_output=True, text=True).stdout)
