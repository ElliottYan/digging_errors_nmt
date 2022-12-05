import sys
import os
import sacrebleu
import argparse
import numpy as np
from collections import defaultdict
from functools import partial
import pickle

from utils import *

def main(args):
    dfs_path = args.input_dir
    beam_size  = args.beam
    base_name = os.path.basename(dfs_path)
    dir_name = os.path.dirname(dfs_path)
    
    # read in references
    print('Extracting outputs!')
    dfs_outputs = read_split_files(dfs_path, beam_size)
    dfs_top1 = get_top1(dfs_outputs)
    cnts = defaultdict(int)
    total = defaultdict(int)
    for key in dfs_outputs:
        cur_output = dfs_outputs[key]
        idx = 0
        for score, sent in cur_output:
            total[idx] += 1
            sent_txt = get_text(sent)
            if sent_txt.strip() == "":
                cnts[idx] += 1
            idx += 1
    
    ordered_keys = sorted(list(cnts.keys()))
    for key in ordered_keys:
        print("Percentage of empty sentence at rank {} is {}.".format(key, cnts[key] / total[key]))
    # print("Model errors with avg ndcg is {}.".format(avg_ndcg))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    # parser.add_argument("--reference_file", type=str, default="")
    # parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)