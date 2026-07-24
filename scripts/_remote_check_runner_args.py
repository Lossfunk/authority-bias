import os, subprocess
src = open("persona-vectors/src/exp19/run_assistant_axis_causal_deconfound.py").read()
# print the chunk after _row_condition definition for context on conditions usage
i = src.find("trivia-conditions")
print("trivia-conditions context:")
print(src[i:i+1200])
print("---")
# look for how conditions string is parsed
import re
for m in re.finditer(r"conditions", src):
    pass
# print near 'conditions' to spot parsing
print("uses of 'conditions' near parse_csv or condition_codes:")
for m in re.finditer(r".{0,80}conditions.{0,200}", src):
    print(m.group())
print("---")
# check entry-point main and how it dispatches
print("end-of-file sketch:")
print(src[-3000:])
