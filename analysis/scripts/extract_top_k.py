import sys

in_file = sys.argv[1]
out_file = sys.argv[2]
beam1 = int(sys.argv[3])
beam2 = int(sys.argv[4])

assert beam1 > beam2

with open(in_file, 'r', encoding='utf8') as inf:
    with open(out_file, 'w', encoding='utf8') as outf:
        cnt = 0
        while True:
            cnt += 1
            line = inf.readline()
            if not line:
                break
            if cnt > beam2:
                if cnt > beam1:
                    cnt = 0
                continue
            else:
                outf.write(line.strip() + '\n')
