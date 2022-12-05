import sys

def get_text(text):
    def strip_str(t):
        # trim </s>
        t = t.strip()
        if t.endswith('</s>'):
            t = t[:-4].strip()
        return t
    if isinstance(text, list):
        return [strip_str(t) for t in text]
    else:
        return strip_str(text)

with open(sys.argv[1], 'r', encoding='utf8') as inf:
    with open(sys.argv[2], 'w', encoding='utf8') as outf:
        while True:
            line = inf.readline()
            if not line:
                break
            new_line = get_text(line)
            outf.write(new_line + '\n')
