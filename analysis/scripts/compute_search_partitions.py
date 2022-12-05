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
import matplotlib.pyplot as plt
import numpy as np

def paint(xs, keys, filename, xlabel="BLEU Score"):
#     plt.style.use('seaborn-bright')
    plt.style.use('fivethirtyeight')
    fig = plt.figure()

#     for key, x in zip(keys, xs):
#     for x in xs:
#         print(key)
    plt.hist(xs, label=keys, density=False, bins=20)  # density=False would make counts
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    plt.legend(loc='upper left')
    plt.savefig(filename+'.pdf', format='pdf', bbox_inches = 'tight')


def read_and_cache(path, beam, script_path):
    beam_size  = args.beam
    base_name = os.path.basename(path)
    dir_name = os.path.dirname(path)
    cached_path = os.path.join(dir_name, base_name + '.outs')

    if True or not os.path.exists(cached_path):
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
    
    # # read in references
    # ref_file = args.reference_file
    # refs = read(ref_file)

    print("Reading dfs outputs!")
    dfs_outputs = read_and_cache(args.dfs_input_dir, beam_size, args.script_path)
    print("Reading beam outputs!")
    beam_outputs = read_and_cache(args.beam_input_dir, beam_size, args.script_path)
    # read in references
    ref_file = args.reference_file
    refs = read(ref_file)

    beam_bleus = score_all_outputs_dict(beam_outputs, refs)
    dfs_bleus = score_all_outputs_dict(dfs_outputs, refs)

    print(len(beam_bleus), len(dfs_bleus), len(refs),)
    # Check if the keys are matched
    compute_partitions(dfs_path, dfs_outputs, beam_outputs, dfs_bleus, beam_bleus, args.metric)
    
    print("Script done")


def compute_partitions(dfs_path, dfs_outputs, beam_outputs, dfs_bleus, beam_bleus, metric):
    _part_dfs, _part_intsec, _part_beam = [], [], []
    assert(metric in ["bleu", "prob"])
    sel_metric = 0 if metric == "bleu" else 1

    for key in dfs_outputs:
        if key not in beam_outputs:
            continue
        part_dfs, part_intsec, part_beam = [], [], []
        sen_intsec = []
        dfs_bleu = dfs_bleus[key]
        beam_bleu = beam_bleus[key]
        n_sen_dfs = len(dfs_outputs[key])
        n_sen_beam = len(beam_outputs[key])
        sen_dfs = [get_text(dfs_outputs[key][i][1]) for i in range(n_sen_dfs)]
        sen_beam = [get_text(beam_outputs[key][i][1]) for i in range(n_sen_beam)]
        print(len(sen_dfs), len(sen_beam))
        for i, s in enumerate(sen_dfs):
            if s in sen_beam:
                if sel_metric:
                    part_intsec.append(dfs_outputs[key][i][0])
                else:
                    part_intsec.append(dfs_bleu[i])
                sen_intsec.append(s)
            else:
                if sel_metric:
                    part_dfs.append(dfs_outputs[key][i][0])
                else:
                    part_dfs.append(dfs_bleu[i])

        # part_beam = [beam_bleu[i] for i, s in enumerate(sen_beam) if s not in part_intsec]
        for i, s in enumerate(sen_beam): 
            if s not in sen_intsec:
                if sel_metric:
                    part_beam.append(beam_outputs[key][i][0])
                else:
                    part_beam.append(beam_bleu[i])
    
        print(len(part_dfs), len(part_intsec), len(part_beam))
        _part_dfs.extend(part_dfs)
        _part_intsec.extend(part_intsec)
        _part_beam.extend(part_beam)
    paint([_part_dfs, _part_intsec, _part_beam], ['top-k', 'intersection', 'beam search'], os.path.basename(dfs_path), xlabel=metric)
        
    return 1.0

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="compute_search_partitions.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--dfs_input_dir", type=str, default="")
    parser.add_argument("--beam_input_dir", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--script_path", type=str, default="")
    parser.add_argument("--metric", type=str, default="bleu", help="bleu or prob")

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)