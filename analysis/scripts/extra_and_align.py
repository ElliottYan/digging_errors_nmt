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

def read_and_cache(path, beam, script_path):
    beam_size  = args.beam

    print('Extracting outputs!')
    outputs = read_split_files(path, beam_size)
    return outputs

def main(args):
    dfs_path = args.dfs_input_dir
    beam_size  = args.beam
    
    # # read in references
    src_file = args.source_file
    srcs = read(src_file)

    print("Reading dfs outputs!")
    dfs_outputs = read_and_cache(args.dfs_input_dir, beam_size, args.script_path)
    write_align_output(dfs_outputs, srcs, args.output_prefix)

def write_align_output(output_dict, srcs, output_prefix):
    # assert len(output_dict) == len(srcs)
    assert output_prefix != ""
    print('Writing outputs to prefix {}!'.format(output_prefix))
    src_f = open(output_prefix + '.src', 'w', encoding='utf8')
    tgt_f = open(output_prefix + '.tgt', 'w', encoding='utf8')

    # for i in range(len(srcs)):
    keys = sorted(list(output_dict.keys()))
    for i in keys:
        cur_exact = output_dict[i]
        # assume srcs is complete 
        src = srcs[i]
        for score, cand in cur_exact:
            cur_cand = cand.strip()
            cur_cand = append_eos(cur_cand, '</s>')
            tgt_f.write(cur_cand + '\n')
            src_f.write(src.strip() + '\n')
    src_f.close()
    tgt_f.close()

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--dfs_input_dir", type=str, default="")
    parser.add_argument("--source_file", type=str, default="")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)