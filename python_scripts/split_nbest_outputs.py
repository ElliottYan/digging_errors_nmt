import sys
import os

def parse(string):
    splits = string.strip().split('|||')
    idx, trg, score = splits
    idx = int(idx)
    score = float(score)
    return idx, trg, score

file_name = sys.argv[1]
each_split = int(sys.argv[2])

# output_file_name = file_name + '.top1'

with open(file_name, 'r', encoding='utf8') as inf:
    lines = inf.readlines()
    
    parse_lines = [parse(line) for line in lines]
    unique_idx = set([item[0] for item in parse_lines])
    nbest = len(lines) // len(unique_idx)
    assert len(lines) % len(unique_idx) == 0
    
    import math
    n_splits = math.ceil(len(unique_idx) / each_split)
    
    for sp_idx in range(n_splits):
        # how much decoded outputs in each split
        each_sents = nbest * each_split
        cur_split_sents = parse_lines[sp_idx * each_sents: (sp_idx+1) * each_sents]
        cur_idx_shift = sp_idx * each_split
        with open(file_name + '.{}'.format(sp_idx), 'w', encoding='utf8') as outf:
            for item in cur_split_sents:
                idx, trg_words, score = item
                outf.write("|||".join([str(idx-cur_idx_shift), trg_words, str(score)]) + '\n')
