import argparse
import sys

import subprocess
import copy
import tempfile
import subprocess
import re
import random
import multiprocessing as mp

from collections import defaultdict
from itertools import repeat
from functools import partial

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Convert thumt output to nbest format suitable for moses.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--input", required=True, type=str,
                        help="path of hypothesis")
    parser.add_argument("--output", required=True, type=str,
                        help="path of output scores")
    parser.add_argument("--num_trials", type=int, default=2000,
                        help="Number of trials")
    parser.add_argument("--n_processes", type=int, default=10,
                        help="Number of workers to use")


    return parser.parse_args(args)

def score_with_weights(weights, hypo_dict):
    """Rerank and score certain hypothesis set with give weights

    Arguments:
        weights {dict} -- dict of weights for each feature, containing the length penalty
        hypo_dict {dict} -- dict of list of hypothesis, key: ID
    """
    len_pen = weights['length_penalty']
    ori_weights = weights
    weights = copy.copy(weights)
    weights.pop('length_penalty')
    ret = []
    for idx in hypo_dict:
        group = hypo_dict[idx]
        best_score = float('-inf')
        best_cand = ''
        for each_item in group:
            hypo = each_item['hypo']
            # compute length penalty factor
            length = len(hypo.strip().split()) + 1
            len_pen_factor = ((5.0+length)/6.0) ** len_pen

            feats = {feat for feat in each_item.keys() if feat != 'hypo'}
            score = 0.0
            for feat in feats:
                score += each_item[feat] * weights[feat]
            # compute score with length penalty
            score = score * length / len_pen_factor
            # check whether is the best score.
            if score > best_score:
                best_score = score
                best_cand = hypo
        ret.append(best_cand)
    # create temp file
    f_temp = tempfile.NamedTemporaryFile(mode='w', encoding='utf8', delete=False)
    temp_name = f_temp.name
    # write to temp file
    for line in ret:
        f_temp.write(line+'\n')
    f_temp.close()
    bleu_script = '/apdcephfs/share_47076/elliottyan/sec_fake_ch2en_secft_secbt_xianfbt_l2r/data/eval_test19_zhen_no_save.sh'
    bleu_cmd = ["bash", bleu_script, temp_name]
    bleu_output = subprocess.check_output(bleu_cmd, stderr=subprocess.STDOUT)
    bleu_score = float(re.search(r"BLEU score = (.+?) ", str(bleu_output)).group(1))
    print("BLEU Score: {} \n Params Set: {}".format(bleu_score, ori_weights))
    return bleu_score
    
def generate_interleave_params(feats):
    assert len(feats) == 2
    key1 = list(feats)[0]
    key2 = list(feats)[1]
    ret = []
    for i in range(100):
        tmp_weight = {}
        tmp_weight['length_penalty'] = 1.4
        tmp_weight[key1] = float(i) / 100
        tmp_weight[key2] = 1.0 - tmp_weight[key1]
        ret.append(tmp_weight)

    return ret

def main(args):
    with open(args.input, 'r', encoding='utf8') as f:
        hypo_lines = f.readlines()
    
    SCORE_FIELD = 3
    FEAT_FIELD = 2
    HYPO_FIELD = 1
    ID_FILED = 0

    fields = [f.strip() for f in hypo_lines[0].split('|||')]
    feats = fields[FEAT_FIELD].split()

    feat_keys = []
    for i in range(len(feats)):
        if feats[i].endswith('='):
            key = feats[i][:-1]
            feat_keys.append(key)
    feats_weights = {key: 0.0 for key in feat_keys}
    feats_weights['length_penalty'] = 0.0

    hypo_dict = defaultdict(list)
    for line in hypo_lines:
        splits = [f.strip() for f in line.strip().split('|||')]
        hypothesis = splits[HYPO_FIELD]
        feats = splits[FEAT_FIELD].split()
        hypo_id = splits[ID_FILED]
        key = ''
        tmp_dict = {
            'hypo': hypothesis,
        }
        for feat in feats:
            if feat.endswith('='):
                key = feat[:-1]
            else:
                tmp_dict[key] = float(feat)
        hypo_dict[hypo_id].append(tmp_dict)
    
    seed = 10357
    random.seed(seed)
    
    # get lower bounds and upper bounds
    lower_bounds = {key: 0.0 for key in feats_weights}
    upper_bounds = {key: 3.0 if key != 'length_penalty' else 2.0 for key in feats_weights}

    if len(feat_keys) == 2:
        random_params = generate_interleave_params(feat_keys)
    else:
        random_params = [
            {key: random.uniform(lower_bounds[key], upper_bounds[key]) for key in feats_weights}
            for k in range(args.num_trials)
        ]
    best_score = 0.0
    best_param = None

    if args.n_processes > 1:
        score_fn = partial(score_with_weights, hypo_dict=hypo_dict)
        # call_bleu(lines_1, lines_2) 
        p = mp.Pool(processes=args.n_processes)
        
        pool_ret = p.imap(score_fn, random_params)
        pool_ret = sorted(list(zip(pool_ret, range(len(random_params)))))[::-1]
        best_score = pool_ret[0][0]
        best_param = random_params[pool_ret[0][1]]

    else:
        for param in random_params:
            score = score_with_weights(hypo_dict, param)
            if score > best_score:
                best_score = score
                best_param = param
    print('Best BLEU: {}'.format(best_score))
    print('Best Param: {}'.format(best_param))

    fout = open(args.output, 'w', encoding='utf8')
    for key, weight in best_param.items():
        fout.write('{}={}\n'.format(key, str(weight)))
    fout.close()
    return best_param

if __name__ == "__main__":
    args = parse_args()
    main(args)
