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

import utils

from collections import defaultdict
from datastructures.min_max_queue import MinMaxHeap
from datastructures.pointer_queue import PointerQueue
from decoding.core import Decoder, PartialHypothesis, EPS_P

import torch
NEG_INF = float("-inf")

class SimpleDFSDecoder(Decoder):
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
    
    name = 'simpledfs'
    def __init__(self, decoder_args):
        """Creates new SimpleDFS decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SimpleDFSDecoder, self).__init__(decoder_args)
        #self._min_length_ratio = 0.25  # TODO: Make configurable
        self._min_length_ratio = -0.1
        self._min_length = -100
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
                for i in range(len(mins)):
                    self.lower_bounds.append(max([item[0] for item in mins[i]]))
                # self.lower_bounds = [mins[i] for i in range(len(mins))]
 
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        # logging.info("New partial: score: %f -> sentence: %s" %
        #     (partial_hypo.score,
        #     partial_hypo.trgt_sentence))

        if partial_hypo.get_last_word() == utils.EOS_ID:
            if len(partial_hypo.trgt_sentence) >= self._min_length:
                self.add_full_hypo(partial_hypo.generate_full_hypothesis())
                if partial_hypo.score > self.best_score:
                    self.best_score = partial_hypo.score
                    logging.info("New best: score: %f exp: %d sentence: %s" %
                          (self.best_score,
                           self.apply_predictor_count,
                           partial_hypo.trgt_sentence))
            return
        self.apply_predictor_count += 1
        posterior = self.dfs_predictor.predict_next()
        # debug
        logging.debug("Expand: best_score: %f exp: %d partial_score: "
                      "%f sentence: %s" %
                      (self.best_score,
                       self.apply_predictor_count,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        first_expansion = True
        """
        # consider eos first.
        iters = list(range(len(posterior)))
        iters[utils.EOS_ID], iters[0] = iters[0], iters[utils.EOS_ID]
        # for trgt_word, score in utils.common_iterable(posterior):
        for i, trgt_word in enumerate(iters):
        """
        for trgt_word in range(len(posterior)):
            score = posterior[trgt_word]
            if score == float('-inf'):
                continue
            if partial_hypo.score + score > self.best_score:
                if first_expansion:
                    pred_states = copy.deepcopy(self.get_predictor_states())
                    first_expansion = False
                else:
                    # restore predictor states from previous dfs call.
                    self.set_predictor_states(copy.deepcopy(pred_states))
                self.consume(trgt_word)
                self._dfs(partial_hypo.expand(trgt_word,
                                              None, # Do not store states
                                              score=partial_hypo.score + score,
                                              score_breakdown=score))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence exhaustively using depth 
        first search.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        self.dfs_predictor = self.predictor
        if self._min_length_ratio > 0.0:
            self._min_length = int(math.ceil(
              self._min_length_ratio * len(src_sentence))) + 1
        self.initialize_predictor(src_sentence)
        self.best_score = self.get_lower_score_bound()
        self._dfs(PartialHypothesis())
        # if self.full_hypos == []:
        #     import pdb; pdb.set_trace()
        return self.get_full_hypos_sorted()

class SimpleDFSTopkDecoder(Decoder):
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
    
    name = 'simpledfstopk'
    def __init__(self, decoder_args):
        """Creates new SimpleDFSTopk decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SimpleDFSTopkDecoder, self).__init__(decoder_args)
        #self._min_length_ratio = 0.25  # TODO: Make configurable
        self._min_length_ratio = -0.1
        self._min_length = -100
        self.k = decoder_args.simpledfs_topk # default=10
        self.heap = []
                
        # read from score_file
        def parse(string):
            idx, trg_word, score = string.strip().split("|||")
            idx, score = int(idx), float(score)
            return idx, trg_word, score

        if decoder_args.score_lower_bounds_file != "":
            # init lower_bounds
            # simple version here
            # TODO: save the trg_words in heap.
            with open(decoder_args.score_lower_bounds_file) as f:
                lines = f.readlines()
                mins = dict()
                for line in lines:
                    idx, trg_words, score = parse(line)
                    if idx not in mins:
                        mins[idx] = score
                    else:
                        mins[idx] = min(score, mins[idx])
                self.lower_bounds = [mins[i] for i in range(len(mins))]
    
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        if partial_hypo.get_last_word() == utils.EOS_ID:
            if len(partial_hypo.trgt_sentence) >= self._min_length:
                # pdb.set_trace()
                full_sent = partial_hypo.generate_full_hypothesis()
                self.add_full_hypo(full_sent)
                if partial_hypo.score > self.lower_bound_value:
                    # self.lower_bound_value = partial_hypo.score
                    heapq.heappush(self.heap, (partial_hypo.score, full_sent))
                    
                    if len(self.heap) > self.k:
                        heapq.heappop(self.heap)
                    
                    # logging.info("Size of heap: %s" % len(self.heap))
                    # logging.info("New in heap: score: %f sentence: %s :" %
                    #         (partial_hypo.score,
                    #         partial_hypo.trgt_sentence))

                    # update lower-bound only if the heap is full
                    if len(self.heap) == self.k:
                        self.lower_bound_value = self.heap[0][0]
                        logging.info("New best: score: %f exp: %d sentence: %s" %
                            (self.lower_bound_value,
                            self.apply_predictor_count,
                            partial_hypo.trgt_sentence))
            return
        
        self.apply_predictor_count += 1
        posterior = self.dfs_predictor.predict_next()
        logging.debug("Expand: lower_bound_value: %f exp: %d partial_score: "
                      "%f sentence: %s" %
                      (self.lower_bound_value,
                       self.apply_predictor_count,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        first_expansion = True
        """
        score_eos = posterior[utils.EOS_ID]
        posterior[utils.EOS_ID] = 0
        # consider eos first.
        def argsort(seq):
            return sorted(range(len(seq)), key=seq.__getitem__, reverse=True) # Descending order
        
        iters = argsort(posterior)
        posterior[utils.EOS_ID] = score_eos
        # iters = list(range(len(posterior)))
        # iters[utils.EOS_ID], iters[0] = iters[0], iters[utils.EOS_ID]
        # for trgt_word, score in utils.common_iterable(posterior):
        for i, trgt_word in enumerate(iters):
        """
        for trgt_word, score in enumerate(posterior):
            score = posterior[trgt_word]
            # NOTE: As we sort the posterior, there's no need to further visit the nodes if bound is not satisfied or -inf
            if (trgt_word != utils.EOS_ID and partial_hypo.score + score <= self.lower_bound_value) or score == float('-inf'):
                break # continue 
            
            if partial_hypo.score + score > self.lower_bound_value:
                if first_expansion:
                    pred_states = copy.deepcopy(self.get_predictor_states())
                    first_expansion = False
                else:
                    # restore predictor states from previous dfs call.
                    self.set_predictor_states(copy.deepcopy(pred_states))
                self.consume(trgt_word)
                self._dfs(partial_hypo.expand(trgt_word,
                                                None, # Do not store states
                                                partial_hypo.score + score,
                                                [(score, 1.0)]))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence exhaustively using depth 
        first search.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        # re-intialize heap
        self.heap = []
        self.dfs_predictor = self.predictor
        if self._min_length_ratio > 0.0:
            self._min_length = int(math.ceil(
              self._min_length_ratio * len(src_sentence))) + 1
        self.initialize_predictor(src_sentence)
        self.lower_bound_value = self.get_lower_score_bound()
        self._dfs(PartialHypothesis())
        b = [item[1] for item in sorted(self.heap)][::-1]
        return b

class PrunedStateManager:
    def __init__(self, n_memory=40000, n_file=1e12):
        self.mem_states = []
        self.file_states = []
        self.path_states = []
        self.mem_num = n_memory
        self.file_num = n_file
        self.mem_score = -1e8
        self.file_score = -1e8
        self.file_id = 0

    def push_state(self, score, state, hypo):
        """ At the begining, push state to the mem states,
        "   if it is full, then save it as a file (ends with pt) using torch
        """
        # print(len(self.mem_states))
        # Add (score, state, hypo) to memory heap
        if len(self.mem_states) < self.mem_num or score > self.mem_score:
            rt = self.push_to_heap(score, state, hypo, self.mem_states, self.mem_num)
            if rt is not None: # if the memory states are full
                self.mem_score = rt[0]
                tmp_state, tmp_hypo = self.save_state_hypo_to_file(rt[1], rt[2])
                rt_1 = self.push_to_heap(rt[0], tmp_state, tmp_hypo, self.file_states, self.file_num)
                if rt_1 is not None:
                    self.file_score = rt_1[0]
                    self.push_to_heap(rt_1[0], rt_1[1], rt_1[2], self.path_states)
        # Add (score, state, hypo) to file heap
        elif len(self.file_states) < self.file_num or score > self.file_score:
            state, hypo = self.save_state_hypo_to_file(state, hypo)
            rt = self.push_to_heap(score, state, hypo, self.file_states, self.file_num)
            if rt is not None: # TODO: it is not implemented as of now.
                self.file_score = rt[0]
                self.push_to_heap(rt[0], rt[1], rt[2], self.path_states)
        # Add (score, state, hypo) to path heap
        else:
            self.push_to_heap(state, score, hypo, self.path_states)
        
    def save_state_hypo_to_file(self, state, hypo, force_delete=True):
        file_state_str = "/media/jast/Dev/TMP/{}.state.pt".format(self.file_id)
        file_hypo_str = "/media/jast/Dev/TMP/{}.hypo.pt".format(self.file_id)
        torch.save(state, file_state_str)
        torch.save(hypo, file_hypo_str)
        self.file_id += 1
        if force_delete:
            del state, hypo # delete pred_state and partial_hypo instancely
        return file_state_str, file_hypo_str

    def push_to_heap(self, score, state, hypo, heap, heap_max=None):
        heapq.heappush(heap, (-score, state, hypo))
        try:
            if heap_max!=None and len(heap) > heap_max:
                return heapq.heappop(heap)
            else:
                return None 
        except:
            print(len(self.mem_states), len(self.file_states))
            quit(0)

    def get_top_node(self, lower_bound):
        if len(self.mem_states) != 0:
            top_node = heapq.heappop(self.mem_states)
        elif len(self.file_states) != 0:
            top_node = heapq.heappop(self.file_states)
        elif len(self.path_states) != 0:
            top_node = heapq.heappop(self.path_states)
        else:
            return None
        if top_node[0] < lower_bound:
            return None
        return top_node

class MaxHeap():
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapq.heapify( self.h)
        
    def add(self, element):
        if len(self.h) < self.length:
            heapq.heappush(self.h, element)
        else:
            heapq.heappushpop(self.h, element)
            
    def getSorted(self):
        return heapq.nlargest(self.length, self.h)
    
    def getK(self, k):
        return heapq.nlargest(k+1, self.h)[-1]

    def getMin(self):
        return self.h[0]

    def clear(self):
        self.h = []
        heapq.heapify(self.h)

class SimpleDFSTopkFasterDecoder(Decoder):
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
    
    name = 'simpledfstopk_faster'
    def __init__(self, decoder_args):
        """Creates new SimpleDFSTopk decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SimpleDFSTopkFasterDecoder, self).__init__(decoder_args)
        #self._min_length_ratio = 0.25  # TODO: Make configurable
        self._min_length_ratio = -0.1
        self._min_length = -100
        self.k = decoder_args.simpledfs_topk # default=10
        self.current_k = 0
        self.heap = []
        self.pruned_states_mng = PrunedStateManager()
        self.all_mins = dict()
        self.max_heap = MaxHeap(self.k)

        # read from score_file
        def parse(string):
            idx, trg_word, score = string.strip().split("|||")
            idx, score = int(idx), float(score)
            return idx, trg_word, score

        if decoder_args.score_lower_bounds_file != "":
            # init lower_bounds
            # simple version here
            # TODO: save the trg_words in heap.
            with open(decoder_args.score_lower_bounds_file) as f:
                lines = f.readlines()
                for line in lines:
                    idx, trg_words, score = parse(line)
                    if idx not in self.all_mins:
                        self.all_mins[idx] = [score,]
                    else:
                        self.all_mins[idx].append(score)
                for idx in self.all_mins:
                    self.all_mins[idx] = sorted(self.all_mins[idx], reverse=True)
                self.lower_bounds = [self.all_mins[i][0] for i in range(len(self.all_mins))]

    def _dfs_single(self, partial_hypo, k):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        # logging.info("New partial: score: %f -> sentence: %s" %
        #     (partial_hypo.score,
        #     partial_hypo.trgt_sentence))

        if partial_hypo.get_last_word() == utils.EOS_ID:
            if partial_hypo.score > self.max_heap.getMin():
                self.max_heap.add(partial_hypo.score)
                logging.info("New global lower bound %f" % self.max_heap.getMin())
            if str(partial_hypo.trgt_sentence) in self.saved_topk:
                return
            if len(partial_hypo.trgt_sentence) >= self._min_length:
                self.add_full_hypo(partial_hypo.generate_full_hypothesis())
                # update lower bound value
                if partial_hypo.score > self.lower_bound_value:
                    self.lower_bound_value = partial_hypo.score
                    logging.info("New best score for top %d: %f exp: %d sentence: %s" %
                          (k, self.lower_bound_value,
                           self.apply_predictor_count,
                           partial_hypo.trgt_sentence))
                # maintain a max heap
                heapq.heappush(self.heap, (-partial_hypo.score, partial_hypo))
            return
        
        self.apply_predictor_count += 1
        posterior = self.dfs_predictor.predict_next()
        # debug
        logging.debug("Expand: lower_bound_value: %f exp: %d partial_score: "
                      "%f sentence: %s" %
                      (self.lower_bound_value,
                       self.apply_predictor_count,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        first_expansion = True
        # consider eos first.
        iters = list(range(len(posterior)))
        iters[utils.EOS_ID], iters[0] = iters[0], iters[utils.EOS_ID]

        # TODO: Sort iters, do DFS on better nodes in priorty
        

        # for trgt_word, score in utils.common_iterable(posterior):
        for i, trgt_word in enumerate(iters):
            score = posterior[trgt_word]
            if score == float('-inf'):
                continue
            # relax constraints for endding sequences.
            if partial_hypo.score + score > self.lower_bound_value or trgt_word == utils.EOS_ID:
                if first_expansion:
                    pred_states = copy.deepcopy(self.get_predictor_states())
                    first_expansion = False
                else:
                    # restore predictor states from previous dfs call.
                    self.set_predictor_states(copy.deepcopy(pred_states))
                self.consume(trgt_word)
                self._dfs_single(partial_hypo.expand(trgt_word,
                                              None, # Do not store states
                                              partial_hypo.score + score,
                                              [(score, 1.0)]), k)
            # elif partial_hypo.score + score > self.max_heap.getMin() and trgt_word != utils.EOS_ID:
            #     pred_states = copy.deepcopy(self.get_predictor_states())
            #     self.consume(trgt_word)
            #     self.pruned_states_mng.push_state(
            #          partial_hypo.score+score, # score
            #          copy.deepcopy(self.get_predictor_states()), # state
            #          copy.deepcopy(partial_hypo) #hypo
            #          )
            #     self.set_predictor_states(copy.deepcopy(pred_states))

    def search_from_node(self):
        """ restore a state and re-search using a relaxed lower-bound
        "
        """
        top_node = self.pruned_states_mng.get_top_node(self.lower_bound_value)
        pred_state = top_node[1][0]
        partial_hypo = top_node[1][1]
        
        # Step 1: lower the bound score
        self.lower_bound_value = self.get_lower_score_bound()
        self.set_predictor_states(copy.deepcopy(pred_state))

        # Step 2: DFS from the node
        self._dfs_single(partial_hypo, k)
        

    def decode(self, src_sentence):
        """Decodes a single source sentence exhaustively using depth 
        first search.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        # re-intialize heap
        # heap saves current best 
        self.heap = []
        # self.truncated = []
        self.dfs_predictor = self.predictor
        if self._min_length_ratio > 0.0:
            self._min_length = int(math.ceil(
              self._min_length_ratio * len(src_sentence))) + 1
        self.initialize_predictor(src_sentence)

        # initialize max heap for bounds
        self.initialize_global_bounds()

        # get top1 score from beam search.
        self.saved_topk = set()
        self.lower_bound_value = self.get_lower_score_bound()
        
        k = 0
        while len(self.saved_topk) < self.k:
            if k > 0:
                # re-init predictor states
                self.predictor.initialize(src_sentence)

            self.current_k = k
            logging.info('Staring iter {}'.format(k))
            # update lower_bound_value
            
            # re-init heap
            self.heap = []
            print(self.lower_bound_value)

            # if k == 0:
            self._dfs_single(PartialHypothesis(), k)


            for cand in self.max_heap.getSorted():
                if cand < self.lower_bound_value:
                    self.lower_bound_value = cand
                    break

            print(self.lower_bound_value)
            k += 1
            # while k > 0:
            #     top_node = self.pruned_states_mng.get_top_node(self.lower_bound_value)
            #     if top_node is not None:
            #         pred_states = top_node[1][0]
            #         partial_hypo = top_node[1][1]
            #         self.set_predictor_states(copy.deepcopy(pred_states))
            #         self._dfs_single(partial_hypo, k)
            #     else:
            #         break # break while, need a lower bound score for the next round of searching
            
            while len(self.heap) > 0 and len(self.saved_topk) < self.k:
                self.saved_topk.add(str(heapq.heappop(self.heap)[1].trgt_sentence))
        
        return self.get_full_hypos_sorted()

    def add_full_hypo(self, hypo):
        """Adds a new full hypothesis to ``full_hypos``. This can be
        used by implementing subclasses to add a new hypothesis to the
        result set. 
        
        Args:
            hypo (Hypothesis): New complete hypothesis
        """
        if len(self.full_hypos) == 0 or hypo.total_score > self.cur_best.total_score:
            self.cur_best = hypo
        self.full_hypos.append(hypo)
    
    def initialize_global_bounds(self):
        """ Get the lowerest global bound
        """
        self.max_heap.clear()
        for bnd in self.all_mins[self.current_sen_id]:
            self.max_heap.add(bnd-EPS_P)
        logging.info('Current lower bound (global): {}'.format(self.max_heap.getMin()))
    

    def get_lower_score_bound(self):
        """Intended to be called by implementing subclasses. Returns a
        lower bound on the best score of the current sentence. This is
        either read from the lower bounds file (if provided) or set to
        negative infinity.
        
        Returns:
            float. Lower bound on the best score for current sentence
        """ 
        if self.current_sen_id < len(self.lower_bounds) and self.current_k == 0:
            lower_bound_value = self.lower_bounds[self.current_sen_id] - EPS_P
        else:
            cand = float("-inf")
            if len(self.heap) > 0:
                cand = max(-1 * self.heap[0][0] - EPS_P, cand)
            # pick the corresponding beam scores 
            if self.all_mins != {}:
                cand = max(self.all_mins[self.current_sen_id][self.current_k] - EPS_P, cand)
            lower_bound_value = cand
        logging.info('Current lower bound: {}'.format(lower_bound_value))
        return lower_bound_value
    