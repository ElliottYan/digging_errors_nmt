import sys
input_file = sys.argv[1]
k = int(sys.argv[2])
output_file = input_file + '.dup{}'.format(k)
with open(input_file, 'r', encoding='utf8') as inf:
    with open(output_file, 'w', encoding='utf8') as outf:
        lines = inf.readlines()
        for line in lines:
            for _ in range(k):
                outf.write(line)