import sys
import os

def parse(string):
    splits = string.strip().split('|||')
    idx, trg, score = splits
    idx = int(idx)
    score = float(score)
    return idx, trg, score

file_name = sys.argv[1]
output_name = file_name + '.text'

# output_file_name = file_name + '.top1'

with open(file_name, 'r', encoding='utf8') as inf:
    lines = inf.readlines()
    
with open(output_name, 'w', encoding='utf8') as f:
    for line in lines:
        parse_line = parse(line)
        text = parse_line[1].strip()
        f.write(text + '\n')