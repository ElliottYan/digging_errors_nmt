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
    dfs_path = args.dfs_input_dir
    beam_size  = args.beam
    
    # read in references
    ref_file = args.reference_file
    refs = read(ref_file)

    print("Reading dfs outputs!")
    dfs_outputs = read_and_cache(args.dfs_input_dir, beam_size, args.script_path)
    print("Reading beam outputs!")
    beam_outputs = read_and_cache(args.beam_input_dir, beam_size, args.script_path)

    # search bleu
    # First, extract dfs texts as search refs format (list)
    search_refs = dict()
    for i in dfs_outputs:
        cur_refs = []
        for j in range(len(dfs_outputs[i])):
            # cur_refs.append(get_text(dfs_outputs[i][j][1]))
            cur_refs.append(append_eos(dfs_outputs[i][j][1], eos_token='</s>'))
        search_refs[i] = cur_refs
    # add argument, check se uses bleu or pvl
    bleu_scores = score_all_outputs(beam_outputs, search_refs, preprocess_func=lambda x: x)
    pvl_scores = compute_all_pvl(beam_outputs, search_refs)
    ret = compute_search_error(pvl_scores, bleu_scores)
    avg_se = mean(ret)
    print("Search errors is {}".format(avg_se))
    
def compute_search_error(pvl_scores, bleu_scores, search_score_type='bleu', only_top1=True):
    """
    SE=1-MRR*(prefix/length + (1-prefix/length)*BLEU(s, ref))
    """
    if search_score_type == 'bleu':
        search_scores = bleu_scores
    elif search_score_type == 'pvl':
        search_scores = pvl_scores
    else:
        raise ValueError("Wrong search score type.")
    assert len(bleu_scores) == len(pvl_scores)

    if not only_top1:
        raise NotImplementedError('Currently, we only consider top1 output of beam search.')
        
    # find top1
    se_list = []
    for i in range(len(search_scores)):
        # only consider the top1 outputs in beam search
        top_search_scores = search_scores[i][0]
        top_pvl_scores = pvl_scores[i][0]
        top_bleu_scores = bleu_scores[i][0]
        top_rank = -1
        top_score = 0.0
        for j in range(len(top_search_scores)):
            cur_score = top_search_scores[j]
            if cur_score > top_score:
                top_rank = j
                top_score = cur_score
                
        if top_rank == -1 or top_score == 0.0:
            cur_mrr = 0.0
        else:
            cur_mrr = 1 / (top_rank+1)
        # top pvl
        cur_pvl = top_pvl_scores[top_rank]
        cur_bleu = top_bleu_scores[top_rank]
        # bleu \in [0, 1]
        cur_bleu = cur_bleu / 100
        cur_se = 1 - cur_mrr * (cur_pvl + (1-cur_pvl) * cur_bleu)
        se_list.append(cur_se)
    return se_list


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--dfs_input_dir", type=str, default="")
    parser.add_argument("--beam_input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)