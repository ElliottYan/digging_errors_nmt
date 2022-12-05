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
    dfs_file = args.input_path
    beam_size  = args.beam
    base_name = os.path.basename(dfs_path)
    dir_name = os.path.dirname(dfs_path)
    cached_path = os.path.join(dir_name, base_name + '.outs')
    print("Computed input directory is {}.".format(dfs_path))
    
    # read in references
    ref_file = args.reference_file
    refs = read(ref_file)

    if not os.path.exists(cached_path) or args.disable_cache is True:
        print('Extracting outputs!')
        if dfs_file == "":
            dfs_outputs = read_split_files(dfs_path, beam_size)
        else:
            dfs_outputs = read_one_file(dfs_file)

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

    np_dfs_scores = [np.array(item) for item in dfstopk_scores]
    dfstopk_logprobs = extra_logprobs(dbpe_detok_dfstopk_outputs)
    np_dfs_logprob = [np.array(item) for item in dfstopk_logprobs]

    for i in range(len(np_dfs_scores)):
        try:
            assert np_dfs_logprob[i].shape == np_dfs_scores[i].shape
        except:
            import pdb; pdb.set_trace()
    
    # should compute over bleu score over 0-1
    perc_np_dfs_score = [item / 100 for item in np_dfs_scores]

    ndcg_lists, dcg_lists, norm_dcg_lists = compute_ndcg_over_list(perc_np_dfs_score, np_dfs_logprob)
    avg_ndcg = mean(ndcg_lists)
    avg_dcg = mean(dcg_lists)
    avg_norm_dcg = mean(norm_dcg_lists)
    print("Model errors with avg relevance-based ndcg is {} !".format(avg_ndcg))
    print("Model errors with avg dcg is {} !".format(avg_dcg))
    print("Model errors with avg normed dcg is {} !".format(avg_norm_dcg))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_model_errors.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")
    parser.add_argument("--disable_cache", action='store_true')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)