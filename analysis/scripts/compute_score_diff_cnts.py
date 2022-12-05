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
    base_name = os.path.basename(path)
    dir_name = os.path.dirname(path)
    cached_path = os.path.join(dir_name, base_name + '.outs')

    if not os.path.exists(cached_path):
        print('Extracting outputs!')
        outputs = read_split_files(path, beam_size)

        ### delbpe && detok for texts, evaluate bleu scores
        funct = partial(call_delbpe_and_detok, script_path=args.script_path)
        dbpe_dtok_outputs = process_text_in_moses_format(outputs, funct)
        with open(cached_path, 'wb') as f:
            pickle.dump(dbpe_dtok_outputs, f)
    else:
        print("Reading cached outputs from:")
        print("{}".format(cached_path))
        with open(cached_path, 'rb') as f:
            dbpe_dtok_outputs = pickle.load(f)
            
    print('Output dict length: {}'.format(len(dbpe_dtok_outputs)))
    return dbpe_dtok_outputs

def main(args):
    # dfs_path = args.beam_input_dir_1
    beam_size  = args.beam
    
    # # read in references
    ref_file = args.reference_file
    refs = read(ref_file)

    print("Reading beam1 outputs!")
    dfs_outputs = read_and_cache(args.beam_input_dir_1, beam_size, args.script_path)
    print("Reading beam2 outputs!")
    beam_outputs = read_and_cache(args.beam_input_dir_2, beam_size, args.script_path)

    b1_score = score_all_outputs_dict(dfs_outputs, refs)
    b2_score = score_all_outputs_dict(beam_outputs, refs)
    pos_cnt = 0
    neg_cnt = 0
    keys = sorted(list(b1_score.keys()))
    out_list = []
    for key in b1_score:
        if key not in b2_score:
            continue
        if b1_score[key][0] > b2_score[key][0]:
            out_list.append(1)
        elif b1_score[key][0] < b2_score[key][0]:
            out_list.append(-1)
        else:
            out_list.append(0)
    with open(os.path.dirname(args.beam_input_dir_1) + '/tmp.txt', 'w') as f:
        for i, item in enumerate(out_list):
            f.write('{}: {}\n'.format(i, item))
    # print("Pos: {}, Neg: {}".format(pos_cnt, neg_cnt))
    # print("Mean of ranks: {}".format(ret.mean()))
    # np.savetxt(args.beam_input_dir + '.match_ranks.txt', ret)

def compute_score_diff(dfs_outputs, beam_outputs):
    ret = []
    for key in dfs_outputs:
        if key not in beam_outputs:
            continue
        # use top1 beam outputs
        beam_text = get_text(beam_outputs[key][0][1])
        i = 0
        for score, dfs_text in dfs_outputs[key]:
            dfs_text = get_text(dfs_text)
            if dfs_text == beam_text:
                break
            i += 1
        ret.append(i)
    return ret

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--beam_input_dir_1", type=str, default="")
    parser.add_argument("--beam_input_dir_2", type=str, default="")
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)