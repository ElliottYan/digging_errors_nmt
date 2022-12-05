import sys
import os
import sacrebleu
import argparse
import numpy as np
from collections import defaultdict
from functools import partial
import pickle
import pdb

from utils import *

def main(args):
    dfs_path = args.input_dir
    beam_size  = args.beam
    base_name = os.path.basename(dfs_path)
    dir_name = os.path.dirname(dfs_path)
    cached_path = os.path.join(dir_name, base_name + '.outs')
    
    # read in references
    ref_file = args.reference_file
    refs = read(ref_file)

    if not os.path.exists(cached_path) or args.disable_cache is True:
        print('Extracting outputs!')
        dfs_outputs = read_split_files(dfs_path, beam_size)

        ### delbpe && detok for texts, evaluate bleu scores
        funct = partial(call_delbpe_and_detok, script_path=args.script_path)
        dbpe_detok_dfstopk_outputs = process_text_in_moses_format(dfs_outputs, funct)
        with open(cached_path, 'wb') as f:
            pickle.dump(dbpe_detok_dfstopk_outputs, f)
    else:
        print("Reading cached outputs from:")
        print("{}".format(cached_path))
        with open(cached_path, 'rb') as f:
            dbpe_detok_dfstopk_outputs = pickle.load(f)
            
    print('Output dict length: {}'.format(len(dbpe_detok_dfstopk_outputs)))

    dfstopk_scores = score_all_outputs(dbpe_detok_dfstopk_outputs, refs)

    inv_ratio = compute_inversions(dfstopk_scores)
    
    print("Computed input directory is {}.".format(dfs_path))
    print("Model errors with inversion ratios is {}.".format(inv_ratio))

def compute_inversions(all_scores):
    inv_cnt = 0
    cnt = 0
    for cur_scores in all_scores:
        # cur_scores = all_scores[key]
        for i in range(len(cur_scores)):
            for j in range(i+1, len(cur_scores)):
                cnt += 1
                if cur_scores[i] < cur_scores[j]:
                    inv_cnt += 1
    return inv_cnt / cnt

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")
    parser.add_argument("--disable_cache", action='store_true')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)