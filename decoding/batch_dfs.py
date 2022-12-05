# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the dfs search strategy """

import copy
import logging
import operator
import math
import numpy as np
import pdb
import heapq
import time

from numpy.lib.function_base import iterable

import utils

from collections import defaultdict
from datastructures.min_max_queue import MinMaxHeap
from datastructures.pointer_queue import PointerQueue
from decoding.core import Decoder, PartialHypothesis, EPS_P, Hypothesis

import torch
from joblib import Parallel, delayed

NEG_INF = float("-inf")

class BatchDFSTopkDecoder(Decoder):
    """This is a stripped down version of the DFS decoder which is
    designed to explore the entire search space. SimpleDFS is
    intended to be used with a `score_lower_bounds_file` from a
    previous beam search run which already contains good lower
    bounds. SimpleDFS verifies whether the lower bound is actually
    the global best score.

    SimpleDFS can only be used with a single predictor.

    SimpleDFS does not support max_expansions or max_len_factor.
    early_stopping cannot be disabled.
    """
    
    name = 'batchdfstopk'
    def __init__(self, decoder_args):
        """Creates new SimpleDFSTopk decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BatchDFSTopkDecoder, self).__init__(decoder_args)
        #self._min_length_ratio = 0.25  # TODO: Make configurable
        self._min_length_ratio = -0.1
        self._min_length = -100
        self.k = decoder_args.simpledfs_topk # default=10
        self.heap = []
        self.batch_size = decoder_args.dfstopk_batchsize # default=10
        self.approx_method = decoder_args.approx_method
        self.approx_k = decoder_args.approx_topk
        self.approx_p = decoder_args.approx_topp
        self.stacked_iters = 0
        
        def sanity_check_approx():
            assert self.approx_method in {"", "topk", "topk"}, "Wrong value for dfs approximation."
            if self.approx_method == 'topk':
                assert self.approx_k != -1, "You need to fill in approx_topk argument with non-negative values."
            if self.approx_method == 'topp':
                assert self.approx_p != -1, "You need to fill in approx_topp argument with non-negative values."
        sanity_check_approx()
        
        # read from score_file
        def parse(string):
            splits = string.strip().split("|||")
            try:
                # nbest lower bounds format.
                assert len(splits) == 4
                idx, trg_word, words, score = splits
                idx, score = int(idx), float(score)
                trg_word = trg_word.strip()
                words = words.strip()
                assert words.endswith("2")
                words = words.split()
                words = [int(item) for item in words]
                return idx, trg_word, words, score
            except:
                raise ValueError("Wrong format of lower bound file")

        if decoder_args.score_lower_bounds_file != "":
            # init lower_bounds
            # simple version here
            # TODO: save the trg_words in heap.
            # TODO: check if there are enough lower-bound sentences
            with open(decoder_args.score_lower_bounds_file) as f:
                lines = f.readlines()
                # Contains all lower bound values
                # mins = []
                from collections import defaultdict
                mins = defaultdict(list)
                for line in lines:
                    idx, trg_words, allwords, score = parse(line)
                    mins[idx].append((score, allwords))
                self.lower_bounds = []
                self.heap_init = []
                for i in range(len(mins)):
                    if len(mins[i]) >= self.k:
                        mins[i] = mins[i][:self.k]
                        self.heap_init.append(None)
                        self.lower_bounds.append(min([item[0] for item in mins[i]]))
                    else:
                        self.heap_init.append(mins[i])
                        self.lower_bounds.append(NEG_INF)
                # self.lower_bounds = [mins[i] for i in range(len(mins))]
    
    @torch.no_grad()
    def _dfs(self, partial_hypos, decoder_inputs=None, decoder_states=[{}], debug=False, iteration=0):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        if iteration >= self.max_len:
            return
        t = []
        if not isinstance(partial_hypos, list):
            partial_hypos = [partial_hypos]
        # assert len(partial_hypos) in [1, self.batch_size]
        if debug is True:
            pdb.set_trace()

        if decoder_inputs is None:
            # first step
            assert len(partial_hypos) == 1
            # decoder_inputs = partial_hypos[0].consumed
            decoder_inputs = self.dfs_predictor.consumed
            decoder_inputs = torch.LongTensor(decoder_inputs).cuda().view(1, -1)

        decoder_order = torch.arange(len(partial_hypos)).cuda()
        unfinished = torch.BoolTensor(len(partial_hypos)).cuda().fill_(True)

        # decoder_inputs = torch.nn.utils.rnn.pad_sequence()
        for idx, partial_hypo in enumerate(partial_hypos):

            if partial_hypo.get_last_word() == utils.EOS_ID:
                if len(partial_hypo.trgt_sentence) >= self._min_length:
                    full_sent = partial_hypo.generate_full_hypothesis()
                    self.add_full_hypo(full_sent)
                    if partial_hypo.score > self.lower_bound_value:
                        sent_str = partial_hypo.strings()
                        if sent_str not in self.heap_set:
                            # add in
                            heapq.heappush(self.heap, (partial_hypo.score, full_sent))
                        else:
                            # ignore instead
                            pass
                        
                        if len(self.heap) > self.k:
                            heapq.heappop(self.heap)
                        # logging.info("Size of heap: %s" % len(self.heap))
                        # logging.info("New in heap: score: %f sentence: %s :" %
                        #     (partial_hypo.score,
                        #     partial_hypo.trgt_sentence))

                        # update lower-bound only if the heap is full
                        if len(self.heap) == self.k:
                            self.lower_bound_value = self.heap[0][0]
                            logging.info("New best: score: %f exp: %d sentence: %s" %
                                (self.lower_bound_value,
                                self.apply_predictor_count,
                                partial_hypo.trgt_sentence))
                    unfinished[idx] = False
            elif partial_hypo.score <= self.lower_bound_value:
                unfinished[idx] = False

        # trim
        decoder_inputs = decoder_inputs[unfinished]
        decoder_order = decoder_order[unfinished]
        partial_hypos = [partial_hypos[i] for i in range(len(partial_hypos)) if unfinished[i].item() is True]
                
        # all is eos.
        if len(decoder_inputs) == 0:
            return

        bsz_indices, bsz_scores, bsz_iters = self._predict_and_sort(decoder_inputs, \
                                                                    decoder_states, \
                                                                    decoder_order, \
                                                                    partial_hypos)
        self._dfs_iteration(bsz_indices, \
                            bsz_scores, \
                            bsz_iters, \
                            decoder_states, \
                            decoder_inputs, \
                            partial_hypos,
                            debug=debug,
                            iteration=iteration)

    def _predict_and_sort(self, decoder_inputs, decoder_states, decoder_order, partial_hypos):

        self.dfs_predictor.reorder_incremental_states(decoder_states, decoder_order)
        self.apply_predictor_count += 1
        # pdb.set_trace()
        posteriors = self.dfs_predictor.batch_predict_next(decoder_inputs, decoder_states)
        # logging.debug("Expand: lower_bound_value: %f exp: %d partial_score: "
        #               "%f sentence: %s" %
        #               (self.lower_bound_value,
        #                self.apply_predictor_count,
        #                partial_hypo.score,
        #                partial_hypo.trgt_sentence))
        
        iters_added_score = posteriors + torch.Tensor([[ph.score] for ph in partial_hypos]).to(posteriors.device)
        score_eos = iters_added_score[:, utils.EOS_ID].clone()
        # In case that pytorch sort is not stable
        iters_added_score[:, utils.EOS_ID] = torch.arange(iters_added_score.shape[0], 0, -1, device=iters_added_score.device)
        vocab_size = iters_added_score.shape[-1]
        const_bsz = iters_added_score.shape[0]

        if self.approx_method == 'topk':
            """
            # Approximate top-k method 1, ignore the batch relationship
            # flatten
            flatten_score = iters_added_score.reshape(-1)
            flatten_mask = flatten_score > self.lower_bound_value

            # reduce the number of elements to sort
            masked_flat_score = flatten_score[flatten_mask]
            masked_flat_index = torch.arange(flatten_score.shape[0], device=iters_added_score.device)[flatten_mask] 
            # _, offset_iters = torch.sort(masked_flat_score, descending=True)
            _, offset_iters = torch.topk(masked_flat_score, min(masked_flat_score.shape[0], const_bsz*self.approx_k))
            iters = masked_flat_index[offset_iters]
            # End of Approximate top-k method 1
            """
            # """
            # Approximate top-k method 2, make sure that every batch has at least 'k' candidates
            const_offset = iters_added_score.shape[1]
            ext_scores, iters = torch.topk(iters_added_score, self.approx_k, sorted=False)
            bsz_offset = torch.arange(0, const_offset*const_bsz, const_offset, device=iters_added_score.device).unsqueeze(dim=-1)
            ext_scores = torch.flatten(ext_scores)
            ext_mask = ext_scores > self.lower_bound_value
            iters = torch.flatten(iters + bsz_offset)[ext_mask]
            # Extract the scores and sort them
            _, offset_iters = torch.sort(iters_added_score.reshape(-1)[iters], descending=True)
            iters = iters[offset_iters]
            # End of Approximate top-k method 2
            # """
        elif self.approx_method == 'topp':
            pass
        else:
            # flatten
            flatten_score = iters_added_score.reshape(-1)
            flatten_mask = flatten_score > self.lower_bound_value
            # reduce the number of elements to sort
            masked_flat_score = flatten_score[flatten_mask]
            masked_flat_index = torch.arange(flatten_score.shape[0], device=iters_added_score.device)[flatten_mask] 
            _, offset_iters = torch.sort(masked_flat_score, descending=True)
            iters = masked_flat_index[offset_iters]

        # Refill
        iters_added_score[:, utils.EOS_ID] = score_eos
        ranked_scores = iters_added_score.reshape(-1)[iters]
    
        bsz_iters = iters % vocab_size
        bsz_indices = iters // vocab_size
        bsz_scores = ranked_scores
        
        partial_hypo_mask = torch.ones(ranked_scores.shape[0], dtype=torch.bool, device=iters_added_score.device)
        partial_hypo_mask[:const_bsz] = ranked_scores[:const_bsz] > self.lower_bound_value
        
        bsz_iters = torch.masked_select(bsz_iters, partial_hypo_mask)
        bsz_scores = torch.masked_select(bsz_scores, partial_hypo_mask)
        bsz_indices = torch.masked_select(bsz_indices, partial_hypo_mask)
        
        return bsz_indices, bsz_scores, bsz_iters

    def _dfs_iteration(self, bsz_indices, bsz_scores, bsz_iters, decoder_states, decoder_inputs, partial_hypos, debug=False, iteration=0):
        max_index = bsz_indices.shape[0]
        adpative_bsz = self.batch_size if self.stacked_iters < 1e5 else 1
        # print(self.stacked_iters, adpative_bsz)
        num_iters = max_index // adpative_bsz + 1
        for i in range(num_iters):
            st = i * adpative_bsz
            nd = (i+1) * adpative_bsz
            if nd > max_index: nd = max_index
            assert(st <= nd)
            cur_scores = bsz_scores[st:nd]
            cur_mask = cur_scores > self.lower_bound_value
            if not torch.any(cur_mask).item(): continue
            cur_range = nd-st    
            cur_scores = cur_scores[cur_mask]
            bsz_idx = bsz_indices[st:nd][cur_mask]
            cur_words = bsz_iters[st:nd][cur_mask].view(-1, 1)
            # reorder incremental states
            cur_incremental_states = copy.deepcopy(decoder_states)
            # reord by bsz idx
            self.dfs_predictor.reorder_incremental_states(cur_incremental_states, bsz_idx)
            cur_history = decoder_inputs[bsz_idx]
            # print(cur_history.shape, cur_words.shape)
            cur_inputs = torch.cat([cur_history, cur_words], dim=-1)
            new_partial_hypos = [partial_hypos[bsz_idx[j].item()].expand(cur_words[j].item(), None, \
                score=cur_scores[j].item()) for j in range(cur_scores.shape[0])]
            
            self.stacked_iters += cur_range
            self._dfs(new_partial_hypos,
                        decoder_inputs=cur_inputs, 
                        decoder_states=cur_incremental_states,
                        debug=debug, iteration=iteration+1)
            self.stacked_iters -= cur_range

    def decode(self, src_sentence, retry=False):
        """Decodes a single source sentence exhaustively using depth 
        first search.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        import torch
        import gc
        cnt = 0

        # re-intialize heap
        self.dfs_predictor = self.predictor
        if self._min_length_ratio > 0.0:
            self._min_length = int(math.ceil(
              self._min_length_ratio * len(src_sentence))) + 1
        # from here, sent_id >= 0
        self.initialize_predictor(src_sentence, retry=retry)
        self.init_heap()
        self.lower_bound_value = self.get_lower_score_bound()
        debug_flag = False
        # if self.current_sen_id == 1:
        #     pdb.set_trace()
        self._dfs(PartialHypothesis(), debug=debug_flag, decoder_inputs=None, decoder_states=[{}])
        
        # try:
        #     # NOTE: without manually set decoder input and
        #     # decoder state to None, it will somehow cache... 
        #     self._dfs(PartialHypothesis(), debug=debug_flag, decoder_inputs=None, decoder_states=[{}])
        #     if 'out of memory' in str(e):
        #         pdb.set_trace()
        #         torch.cuda.empty_cache()
        #         self.batch_size = max(int(self.batch_size * 0.5), 1)
        #         logging.warning('| WARNING: Ran out of memory, retrying sentence!')
        #         logging.info('| INFO: Reset batch size to {}!'.format(self.batch_size))
        #         self.decode(src_sentence, retry=True)
        b = [item[1] for item in sorted(self.heap)][::-1]
        return b
    
    def init_heap(self):
        if self.heap_init[self.current_sen_id] is None:
            self.heap = []
            self.heap_set = set()
        else:
            self.heap = self.heap_init[self.current_sen_id]
            self.heap = [(item[0], Hypothesis(item[1], item[0])) for item in self.heap]
            self.heap_set = set([item[1].strings() for item in self.heap])