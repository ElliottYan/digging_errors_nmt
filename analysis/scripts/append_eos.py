import sys

def append_eos(text, eos_token='</s>'):
    def _append_eos(t):
        # trim </s>
        t = t.strip()
        if not t.endswith(eos_token):
            if t != "":
                t = "{} {}".format(t, eos_token)
            else:
                t = eos_token
        return t
    
    if isinstance(text, list):
        return [_append_eos(t) for t in text]
    else:
        return _append_eos(text)

with open(sys.argv[1], 'r', encoding='utf8') as inf:
    with open(sys.argv[2], 'w', encoding='utf8') as outf:
        while True:
            line = inf.readline()
            if not line:
                break
            new_line = append_eos(line)
            outf.write(new_line + '\n')
