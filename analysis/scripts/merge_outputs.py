import sys
import math

in_file = sys.argv[1]
out_file = sys.argv[2]
beam = int(sys.argv[3])

# assert beam1 > beam2

with open(in_file, 'r', encoding='utf8') as inf:
    lines = inf.readlines()
assert len(lines) % beam == 0

with open(out_file, 'w', encoding='utf8') as outf:
    for i, line in enumerate(lines):
        splits = line.strip().split('|||')
        read_idx = int(splits[0])
        assert read_idx == i
        new_idx = str(math.floor(i / beam))
        new_splits = [new_idx] + splits[1:]
        new_line = "|||".join(new_splits)
        outf.write(new_line + '\n')
