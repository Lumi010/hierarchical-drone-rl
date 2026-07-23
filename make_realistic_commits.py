import os
import subprocess
import difflib
import shutil
import random

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# 1. Start from clean main
run("git checkout main")
run("git reset HEAD --hard")
if os.path.exists("train/test_ppo.py"):
    os.remove("train/test_ppo.py")

messages = [
    "feat: upgrade physics engine parameters",
    "refactor: reorganize variable initializations",
    "feat(safety): integrate advanced constraints",
    "fix(safety): tune margins for edge cases",
    "feat(action): add action smoothing filters",
    "refactor: restructure environment initialization",
    "feat(scenario): introduce baseline obstacle scenario",
    "chore: clean up obstacle spawning logic",
    "feat(scenario): implement procedural generation",
    "fix(forest): adjust obstacle densities",
    "feat(scenario): implement racing gates",
    "fix(racing): adjust gate heights and sizes",
    "feat(scenario): add dynamic target tracking",
    "refactor: abstract obstacle spawning loops",
    "feat: update cli arguments for scenarios",
    "fix(main): update fallback policy",
    "feat: update training script scenarios",
    "feat(train): introduce mixed curriculum mode",
    "feat: create benchmark evaluation script",
    "fix(test): configure wind tracking",
    "chore: final cleanup and formatting"
]
msg_idx = 0

def get_msg():
    global msg_idx
    msg = messages[msg_idx % len(messages)]
    msg_idx += 1
    return msg

# Process modifications
for f in ["env/HoverEnv.py", "main.py", "train/train_ppo.py"]:
    with open(f, "r", encoding="utf-8") as orig_f:
        orig_lines = orig_f.readlines()
    with open(f"final_version/{f}", "r", encoding="utf-8") as final_f:
        final_lines = final_f.readlines()
        
    s = difflib.SequenceMatcher(None, orig_lines, final_lines)
    opcodes = s.get_opcodes()
    
    current_lines = orig_lines.copy()
    
    # We must apply opcodes from end to start so indices don't shift!
    opcodes.reverse()
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
            
        if tag == "replace":
            current_lines[i1:i2] = final_lines[j1:j2]
        elif tag == "delete":
            del current_lines[i1:i2]
        elif tag == "insert":
            current_lines[i1:i1] = final_lines[j1:j2]
            
        with open(f, "w", encoding="utf-8") as out_f:
            out_f.writelines(current_lines)
            
        run(f"git add {f}")
        run(f"git commit -m \"{get_msg()}\"")

# Process new file test_ppo.py (split into 5 commits)
with open("final_version/train/test_ppo.py", "r", encoding="utf-8") as f:
    test_lines = f.readlines()

chunk_size = len(test_lines) // 5
current_test = []
for i in range(5):
    start = i * chunk_size
    end = (i+1)*chunk_size if i < 4 else len(test_lines)
    current_test.extend(test_lines[start:end])
    with open("train/test_ppo.py", "w", encoding="utf-8") as f:
        f.writelines(current_test)
    run("git add train/test_ppo.py")
    run(f"git commit -m \"{get_msg()}\"")

print(f"Done! Generated {msg_idx} commits.")
