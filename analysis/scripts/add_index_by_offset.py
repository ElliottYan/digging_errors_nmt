import sys

input_file = sys.argv[1]
bias = int(sys.argv[2])
out_file = input_file + '.bias'

with open(input_file, 'r', encoding='utf8') as f:
    lines = f.readlines()

with open(out_file, 'w', encoding='utf8') as f:
    for line in lines:
        splits = line.strip().split("|||")
        idx = int(splits[0])
        new_idx = idx + bias
        new_splits = [str(new_idx)] + splits[1:]
        txt = "|||".join(new_splits)
        f.write(txt + '\n')
