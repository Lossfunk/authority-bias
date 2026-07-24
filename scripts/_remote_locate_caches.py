"""List small cached artifacts (directions, activations, masks) we'd need for the CPU shuffled-vector control."""
import os
print("== ps (job alive?) ==")
os.system("ps -p 1448 -o pid,etime,cmd >&2 || true")

# We need: per-model authority direction(s), assistant axis, masks, and a shard of cached residual activations
# at a single (layer, position) to project against.
candidates = []
for model_dir in ["olmo2", "olmo31"]:
    base = f"persona-vectors/neurips-results/{model_dir}/mechanism"
    for sub in os.listdir(base) if os.path.isdir(base) else []:
        candidates.append(f"{base}/{sub}")
print("\n== mechanism subdirs ==")
for c in candidates:
    print(c)

print("\n== detailed listing ==")
import json, subprocess
patterns = [
    "label_masks.json",
    "summary.json",
    "primary_direction.pt",
    "directions.pt",
    "assistant_axis.pt",
    "assistant_axis_layer.pt",
    "metadata.jsonl",
    "activations",
    "shard",
]
for c in candidates:
    if not os.path.isdir(c): continue
    for fn in os.listdir(c):
        full = os.path.join(c, fn)
        size = None
        try:
            if os.path.isfile(full): size = os.path.getsize(full)
        except Exception: pass
        if any(p in fn for p in patterns):
            print(size, full)
        elif os.path.isdir(full):
            inside = []
            for x in os.listdir(full)[:30]:
                xfull = os.path.join(full, x)
                xsize = os.path.getsize(xfull) if os.path.isfile(xfull) else None
                inside.append((xsize, xfull))
            for s, p in inside:
                print(s, p)

print("\n== where are extraction shards (token-level activation stores) ==")
out = subprocess.run(["bash","-lc","find persona-vectors/neurips-results/olmo31 -maxdepth 5 -type d -name 'extraction*' -o -name 'activation*' -o -name 'shards' 2>/dev/null | head -50"], capture_output=True, text=True).stdout
print(out)
out = subprocess.run(["bash","-lc","find persona-vectors/neurips-results/olmo2 -maxdepth 5 -type d -name 'extraction*' -o -name 'activation*' -o -name 'shards' 2>/dev/null | head -50"], capture_output=True, text=True).stdout
print(out)
out = subprocess.run(["bash","-lc","find persona-vectors -maxdepth 6 -type d -name 'olmo31_authority_activations*' 2>/dev/null | head -20"], capture_output=True, text=True).stdout
print(out)
out = subprocess.run(["bash","-lc","find persona-vectors -maxdepth 6 -type d -name '*authority_activations*' 2>/dev/null | head -20"], capture_output=True, text=True).stdout
print(out)
