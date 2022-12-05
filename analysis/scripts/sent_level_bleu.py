import argparse
import sys

import subprocess
import copy
import tempfile
import subprocess
import re
import random
import multiprocessing as mp

import numpy as np

from collections import defaultdict
from itertools import repeat
from functools import partial

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Convert thumt output to nbest format suitable for moses.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input_file", required=True, type=str,
                        help="path of hypothesis")
    parser.add_argument("--ref_file", required=True, type=str,
                        help="path of references")
    parser.add_argument("--output", required=True, type=str,
                        help="path of output scores")
    parser.add_argument("--n_processes", type=int, default=10,
                        help="Number of workers to use")

    return parser.parse_args(args)

def score_func(two_lines):
    """Rerank and score certain hypothesis set with give weights

    Arguments:
        weights {dict} -- dict of weights for each feature, containing the length penalty
        hypo_dict {dict} -- dict of list of hypothesis, key: ID
    """
    hypo_line, ref_line = two_lines
    # create temp file
    f_temp = tempfile.NamedTemporaryFile(mode='w', encoding='utf8', delete=False)
    f_ref = tempfile.NamedTemporaryFile(mode='w', encoding='utf8', delete=False)
    hypo_name = f_temp.name
    ref_name = f_ref.name
    # write to temp file
    f_temp.write(hypo_line.strip()+'\n')
    f_temp.close()
    
    f_ref.write(ref_line.strip()+'\n')
    f_ref.close()

    bleu_script = '/apdcephfs/share_47076/elliottyan/beam-search/beam-search-decoding/analysis/scripts/call_multi_bleu.sh'
    bleu_cmd = ["bash", bleu_script, ref_name, hypo_name]
    try:
        bleu_output = subprocess.check_output(bleu_cmd, stderr=subprocess.STDOUT)
        bleu_score = float(re.search(r"BLEU = (.+?),", str(bleu_output)).group(1))
        print("BLEU score: {}".format(bleu_score))
    except:
        bleu_score = 0.0

    # print("BLEU Score: {} \n Params Set: {}".format(bleu_score))
    return bleu_score
    
def main(args):
    all_hypos = []
    file_name = args.input_file
    with open(file_name, 'r', encoding='utf8') as f:
        hypo_lines = f.readlines()
    with open(args.ref_file, 'r', encoding='utf8') as f:
        ref_lines = f.readlines()
    
    if args.n_processes > 1:
        p = mp.Pool(processes=args.n_processes)
        
        pool_ret = p.imap(score_func, zip(hypo_lines, ref_lines))
        bleu_scores = np.array(list(pool_ret))
        np.savetxt(args.output, bleu_scores)

    else:
        scores = score_func(hypo_lines, ref_lines)

if __name__ == "__main__":
    args = parse_args()
    main(args)
