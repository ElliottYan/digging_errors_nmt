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
    # base_name = os.path.basename(dfs_path)
    # dir_name = os.path.dirname(dfs_path)
    
    # read in references
    print('Extracting outputs!')
    dfs_outputs = read_split_files(dfs_path, beam_size)
    # funct = partial(call_delbpe_and_detok, script_path=args.script_path)
    # dbpe_detok_dfstopk_outputs = process_text_in_moses_format(dfs_outputs, funct)
    # print('Size of output dict: {}'.format(len(dbpe_detok_dfstopk_outputs)))
 
    dfs_top1 = get_top1(dfs_outputs)
    with open(dfs_path + '.outs.top1', 'w', encoding='utf8') as f:
        # for key, val in dfs_top1:
        for i in range(len(dfs_top1)):
            val = get_text(dfs_top1[i])
            f.write(val + '\n')

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    # parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)