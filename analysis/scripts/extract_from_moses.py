import sys

input_file = sys.argv[1]
out_file = input_file + '.text'

with open(input_file, 'r', encoding='utf8') as f:
    lines = f.readlines()

with open(out_file, 'w', encoding='utf8') as f:
    for line in lines:
        splits = line.strip().split("|||")
        txt = splits[1].strip()
        f.write(txt + '\n')
