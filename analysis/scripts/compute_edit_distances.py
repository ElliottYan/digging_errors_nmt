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
from pprint import pprint
import scipy.special as special

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def estimate_edit_distribution(len_target, V=60, T=0.9):
    v = 60  # Vocabulary size
    T = .9  # Temperature
    max_edits = len_target
    x = np.zeros(max_edits)
    for n_edits in range(max_edits):
        total_n_edits = 0  # total edits with n_edits edits without v^n_edits term
        for n_substitutes in range(min(len_target, n_edits)+1):
            # print(n_substitutes)
            n_insert = n_edits - n_substitutes
            current_edits = special.comb(len_target, n_substitutes, exact=False) * \
                            special.comb(len_target+n_insert-n_substitutes, n_insert, exact=False)
            total_n_edits += current_edits
        x[n_edits] = np.log(total_n_edits) + n_edits * np.log(v)
        # log(tot_edits * v^n_edits)
        x[n_edits] = x[n_edits] -n_edits / T * np.log(v) -n_edits / T
        # log(tot_edits * v^n_edits * exp(-n_edits / T) * v^(-n_edits / T))

    p = np.exp(x)
    p /= np.sum(p)
    return p

def minDis(s1, s2, n, m, dp) :
    # If any string is empty,
    # return the remaining characters of other string         
    if(n == 0) :
        return m       
    if(m == 0) :
        return n
                      
    # To check if the recursive tree
    # for given n & m has already been executed
    if(dp[n][m] != -1)  :
        return dp[n][m];
                    
    # If characters are equal, execute
    # recursive function for n-1, m-1   
    if(s1[n - 1] == s2[m - 1]) :          
        if(dp[n - 1][m - 1] == -1) :
            dp[n][m] = minDis(s1, s2, n - 1, m - 1, dp)
            return dp[n][m]                  
        else :
            dp[n][m] = dp[n - 1][m - 1]
            return dp[n][m]
          
    # If characters are nt equal, we need to          
    # find the minimum cost out of all 3 operations.        
    else :           
        if(dp[n - 1][m] != -1) :  
            m1 = dp[n - 1][m]     
        else :
            m1 = minDis(s1, s2, n - 1, m, dp)
                
        if(dp[n][m - 1] != -1) :               
            m2 = dp[n][m - 1]           
        else :
            m2 = minDis(s1, s2, n, m - 1, dp)  
        if(dp[n - 1][m - 1] != -1) :   
            m3 = dp[n - 1][m - 1]   
        else :
            m3 = minDis(s1, s2, n - 1, m - 1, dp)
      
        dp[n][m] = 1 + min(m1, min(m2, m3))
        return dp[n][m]

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
    return dbpe_dtok_outputs

def main(args):
    beam_size  = args.beam
    
    # read in references
    ref_file = args.reference_file
    refs = read(ref_file)
    
    dfs_outputs = read_and_cache(args.dfs_input_dir, beam_size, args.script_path)
    # beam_outputs = read_and_cache(args.beam_input_dir, beam_size, args.script_path)
    
    ed_distributions = {}
    ed_results = [[] for i in range(10)]
    ed_results_reorder = [[] for i in range(10)]
    
    debug = False 
    if debug is True:
        dfs_outputs = {
            key:val for key, val in dfs_outputs.items() if key < 20
        }

    for i in tqdm.tqdm(range(len(dfs_outputs))):
    # for i in range(390, 394):
        str1 = refs[i]
        tmp_res = []
        n = len(str1)
        is_overflow = False
        
        if len(dfs_outputs[i]) != 10: continue

        for j in range(len(dfs_outputs[i])): # for top-k
            str2 = dfs_outputs[i][j][1]
            m = len(str2)
            len_target = max(n, m)
            if len_target > 442:
                is_overflow = True
                break
        
        if is_overflow: continue

        for j in range(len(dfs_outputs[i])): # for top-k
            str2 = dfs_outputs[i][j][1]
            m = len(str2)
            len_target = max(n, m)

            if len_target not in ed_distributions.keys():
                ed_distributions[len_target] = estimate_edit_distribution(len_target)
    
            dp = [[-1 for i in range(m + 1)] for j in range(n + 1)]
            edit_dist = minDis(str1, str2, n, m, dp)
            edit_dist_prime = np.sum(ed_distributions[len_target][:edit_dist])
            # edit_dist_prime = edit_dist / len_target
            ed_results[j].append(edit_dist_prime)
            tmp_res.append(edit_dist_prime)
        
        tmp_res = sorted(tmp_res)
        for j in range(len(dfs_outputs[i])):
            ed_results_reorder[j].append(tmp_res[j])
    
    print("Edit distances by edit-distance-based reordering")
    print(np.asarray(ed_results_reorder, dtype=float).mean(axis=-1))
    ed_result_cat = np.asarray(ed_results[0])
    i = 0
    def paint(x, out_path):
        plt.ioff()
        fig = plt.figure()
        plt.hist(x, density=True, bins=30)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.savefig(out_path)
        plt.close(fig)
        
    while True:
        if i != 0:
            ed_result_cat = np.concatenate((ed_result_cat, ed_results[i]), axis=-1)
        print("Edit distances@k, where k = ", i+1)
        h = np.histogram(ed_result_cat, bins=10)
        print("Distribution: ", h[0] / ed_result_cat.shape[0])
        print("Counts: ", h[0])
        print("Bins: ", h[1])
        if args.paint is True:
            # paint(h[0] / ed_result_cat.shape[0], out_path=args.dfs_input_dir + '.ed_dist.top_{}.png'.format(i+1))
            paint(ed_result_cat, out_path=args.dfs_input_dir + '.ed_dist.top_{}.png'.format(i+1))
        else:
            if i == 9:
                np.savetxt(args.dfs_input_dir + '.ed_dist.top_{}.txt'.format(i+1), ed_result_cat)
        i += 1
        if i == 10: break

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_edit_distances.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--dfs_input_dir", type=str, default="")
    parser.add_argument("--beam_input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")
    parser.add_argument("--paint", action='store_true')


    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)