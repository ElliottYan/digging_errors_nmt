import sys
import os
from collections import defaultdict
from numpy.lib.type_check import _real_if_close_dispatcher
import sacrebleu
import copy
import subprocess

import sklearn
import numpy as np
import tqdm

def get_text(text):
    def strip_str(t):
        # trim </s>
        t = t.strip()
        if t.endswith('</s>'):
            t = t[:-4].strip()
        return t
    if isinstance(text, list):
        return [strip_str(t) for t in text]
    else:
        return strip_str(text)

def read_split_files(file_path, beam_size):
    """Read split files with a given beam size 

    Args:
        file_path (str): String contains the input directory
        beam_size (int): Beam size for each split file

    Returns:
        Dict: Give a output dict with each output indexing with its true index. Note: some index may be missing ([]) due to missing files in input directory.
    """
    all_lines = dict()
    for file in os.listdir(file_path):
        cur_path = os.path.join(file_path, file)
        with open(cur_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        num_suffix = file.split('.')[-1]
        all_lines[int(num_suffix)] = lines

    ret = defaultdict(list)
    for i in all_lines:
        offset = beam_size * i
        for j, line in enumerate(all_lines[i]):
            try:
                splits = line.strip().split('|||')
                assert len(splits) == 3 or len(splits) == 4
                if len(splits) == 4:
                    splits = (splits[0], splits[1], splits[3])
            except:
                print(i)
                print(line)
                print(splits)
            idx, sent, score = splits
            assert int(idx) < beam_size
            true_idx = offset + int(idx)
            ret[true_idx].append((float(score), sent.strip()))
    return ret

def read_one_file(file_path):
    """Read split files with a given beam size 

    Args:
        file_path (str): String contains the input directory
        beam_size (int): Beam size for each split file

    Returns:
        Dict: Give a output dict with each output indexing with its true index. Note: some index may be missing ([]) due to missing files in input directory.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    ret = defaultdict(list)
    for i in range(len(lines)):
        line = lines[i]
        try:
            splits = line.strip().split('|||')
            assert len(splits) == 3 or len(splits) == 4
            if len(splits) == 4:
                splits = (splits[0], splits[1], splits[3])
        except:
            print(i)
            print(line)
            print(splits)

        idx, sent, score = splits
        true_idx = int(idx)
        ret[true_idx].append((float(score), sent.strip()))
    return ret


def read(file):
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def process_text_in_moses_format(output_dict, func):
    all_texts = []
    num_keys = output_dict.keys()
    for i in num_keys:
        for j in range(len(output_dict[i])):
            all_texts.append(output_dict[i][j][1])
        
    import tempfile
    # write to temp file
    f_temp = tempfile.NamedTemporaryFile(mode='w', encoding='utf8', delete=False)
    temp_name = f_temp.name
    for line in all_texts:
        f_temp.write(line+'\n')
    f_temp.close()
    ret_file_name = func(temp_name)
    with open(ret_file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
    ret_texts = [line.strip() for line in lines]
    
    ret = copy.deepcopy(output_dict)
    k = 0
    # put back
    for i in num_keys:
        for j in range(len(ret[i])):
            ret[i][j] = (ret[i][j][0], ret_texts[k])
            k += 1
    assert k == len(ret_texts)
    return ret

def call_delbpe(file_name, script_path):
    subprocess.run(['bash', os.path.join(script_path, 'delbpe.sh'), file_name])
    return file_name + '.delbpe'

def call_detok(file_name, script_path):
    out_path = file_name + '.detok'
    f1 = open(file_name, 'r', encoding='utf8')
    f2 = open(out_path, 'w', encoding='utf8')
    subprocess.run(['perl', os.path.join(script_path, 'detokenizer.perl')], stdin=f1, stdout=f2)
    return out_path

def call_delbpe_and_detok(file_name, script_path):
    f1 = call_delbpe(file_name, script_path)
    f2 = call_detok(f1, script_path)
    return f2

def get_top1(outputs):
    ret = dict()
    output_keys = sorted(list(outputs.keys()))
    for key in outputs:
        ret[key] = outputs[key][0][1]
    return ret

def write_file(sents, file):
    with open(file, 'w', encoding='utf8') as f:
        for sent in sents:
            f.write(sent + '\n')

def score_all_outputs(output_dict, refs, preprocess_func=get_text):
    # NOTE: scores only contains scores for non-empty outputs 
    scores = []
    output_dict_keys = sorted(list(output_dict.keys()))
    for idx in tqdm.tqdm(output_dict_keys):
        # try:
        cur_ref = preprocess_func(refs[idx])
        # except:
        #     continue
            
        candidates = output_dict[idx]
        candidates = [preprocess_func(item[1]) for item in candidates]
        if not isinstance(cur_ref, list):
            cur_scores = [sacrebleu.sentence_bleu(item, [cur_ref]).score for item in candidates]
        else:
            cur_scores = []
            for each_cand in candidates:
                # each candidate over multiple references.
                cur_scores.append([sacrebleu.sentence_bleu(each_cand, [each_ref]).score if each_ref != "" else 0 for each_ref in cur_ref])
        scores.append(cur_scores)
    return scores

def score_all_outputs_dict(output_dict, refs, preprocess_func=get_text):
    # NOTE: scores only contains scores for non-empty outputs 
    scores = {}
    output_dict_keys = sorted(list(output_dict.keys()))
    for idx in tqdm.tqdm(output_dict_keys):
        try:
            cur_ref = preprocess_func(refs[idx])
        except:
            continue
            
        candidates = output_dict[idx]
        candidates = [preprocess_func(item[1]) for item in candidates]
        if not isinstance(cur_ref, list):
            cur_scores = [sacrebleu.sentence_bleu(item, [cur_ref]).score for item in candidates]
        else:
            cur_scores = []
            for each_cand in candidates:
                # each candidate over multiple references.
                cur_scores.append([sacrebleu.sentence_bleu(each_cand, [each_ref]).score if each_ref != "" else 0 for each_ref in cur_ref])
        scores[idx] = cur_scores
    return scores


def compute_all_pvl(output_dict, refs):
    print('Computing pvl for all outputs.')
    scores = []
    output_dict_keys = sorted(list(output_dict.keys()))
    for idx in tqdm.tqdm(output_dict_keys):
        try:
            cur_ref = refs[idx]
        except:
            continue
        candidates = output_dict[idx]
        candidates = [item[1] for item in candidates]
        cur_scores = []
        for each_cand in candidates:
            # each candidate over multiple references.
            cur_scores.append([compute_prefix_vs_length(each_cand, each_ref) for each_ref in cur_ref])
        scores.append(cur_scores)
    return scores

def compute_prefix_vs_length(candidate, reference):
    # correct prefix divided by reference length
    cand_splits = candidate.split()
    ref_splits = reference.split()
    if len(ref_splits) == 0:
        return 0
    min_range = min(len(cand_splits), len(ref_splits))
    max_range = max(len(cand_splits), len(ref_splits))
    cor = 0
    for i in range(min_range):
        if cand_splits[i] == ref_splits[i]:
            cor += 1
        else:
            break
    pvl = cor / max_range
    return pvl

def extra_logprobs(output_dict):
    scores = []
    output_dict_keys = sorted(list(output_dict.keys()))
    for idx in output_dict_keys:
        candidates = output_dict[idx]
        cur_scores = [item[0] for item in candidates]
        scores.append(cur_scores)
    return scores

def mean(l):
    return sum(l) / len(l)

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def optimal_dcg_score(y_true, k=10, gains="exponential"):
    # order = np.argsort(y_score)[::-1]
    y_true = np.ones(y_true.shape)[:k]

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    if best == 0:
        ret = 0, actual
    else:
        ret = actual / best, actual
    return ret

def ndcg_score_relevance(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    # in this function, we sort the scores and replace with relevance score.
    y_rel = copy.deepcopy(y_true)

    y_rel[np.argsort(y_rel)] = np.arange(0, len(y_rel))
    
    best = dcg_score(y_rel, y_rel, k, gains)
    actual = dcg_score(y_rel, y_score, k, gains)
    if best == 0:
        ret = 0
    else:
        ret = actual / best
    return ret

def compute_ndcg_over_list(score_list, logprob_list):
    ndcgs = []
    dcgs = []
    norm_dcgs = []
    for i in range(len(score_list)):
        cur_score = score_list[i]
        cur_logprob = logprob_list[i]
        # compute normed dcg
        _, dcg = ndcg_score(cur_score, cur_logprob)
        opt_dcg = optimal_dcg_score(cur_score)
        # compute relevance-based ndcg score
        rel_ndcg = ndcg_score_relevance(cur_score, cur_logprob)
        ndcgs.append(rel_ndcg)
        norm_dcgs.append(dcg / opt_dcg)
        dcgs.append(dcg)
    return ndcgs, dcgs, norm_dcgs

def append_eos(text, eos_token):
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
