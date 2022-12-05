import sys
import os

def parse(string):
    splits = string.strip().split('|||')
    idx, trg, score = splits
    idx = int(idx)
    score = float(score)
    return idx, trg, score

file_name = sys.argv[1]
output_file_name = file_name + '.top1'

with open(file_name, 'r', encoding='utf8') as inf:
    lines = inf.readlines()
    
    # unique_idx = set([item[0] for item in lines])
    # nbest = len(lines) / len(unique_idx)
    # assert len(lines) % len(unique_idx) == 0

    # write outputs
    visited = set()
    with open(output_file_name, 'w', encoding='utf8') as outf:
        # lines = [parse(line) for line in lines]

        for line in lines:
            idx = parse(line)[0]
            if idx not in visited:
                outf.write(line)
            visited.add(idx)
