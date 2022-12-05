import copy
import utils
from decoding.core import Decoder, PartialHypothesis
import numpy as np
import heapq


class BeamDecoder(Decoder):
    """This decoder implements standard beam search and several
    variants of it such as diversity promoting beam search and beam
    search with heuristic future cost estimates. This implementation
    supports risk-free pruning.
    """
    name = 'beam'
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. The following values
        """
        super(BeamDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.beam_size = decoder_args.beam if not self.gumbel else self.nbest
        self.stop_criterion = self._best_eos if decoder_args.early_stopping else self._all_eos
    
    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        if len(self.full_hypos) == 0:
            return False
        cur_scores = [self.get_max_pos_score(hypo) for hypo in hypos]
        return all([c < self.cur_best.total_score for c in cur_scores])
            
    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])

    def _get_next_hypos(self, all_hypos, all_scores):
        """Get hypos for the next iteration. """

        inds = utils.argmax_n(all_scores, self.beam_size)
        return [all_hypos[ind] for ind in inds]
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [PartialHypothesis(self.get_predictor_states(), self.calculate_stats)]
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictor(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        while not self.stop_criterion(hypos) and it < self.max_len:
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.beam_size):
                    next_hypos.append(next_hypo)
                    next_scores.append(self.get_adjusted_score(next_hypo))
            hypos = self._get_next_hypos(next_hypos, next_scores)
            
        return self.get_full_hypos_sorted(hypos)


class DiverseBeamDecoder(BeamDecoder):
    """This decoder implements diversity promoting beam search Vijayakumar et. al. (2016).
    """
    name = 'diverse_beam'
    def __init__(self, decoder_args):
        
        super(DiverseBeamDecoder, self).__init__(decoder_args)
        assert not self.gumbel

        self.beam_size = decoder_args.beam
        self.num_groups = decoder_args.diversity_groups
        self.lmbda = decoder_args.diversity_reward
        self.group_sizes = [self.beam_size//self.num_groups]*self.num_groups
        for i in range(self.beam_size - self.group_sizes[0]*self.num_groups):
            self.group_sizes[i] += 1
        assert sum(self.group_sizes) == self.beam_size
        
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [[PartialHypothesis(copy.deepcopy(self.get_predictor_states()), 
                                    self.calculate_stats)] for i in range(self.num_groups)]

    def _get_next_hypos(self, all_hypos, size, other_groups=None):
        """Get hypos for the next iteration. """
        all_scores = np.array([self.get_adjusted_score(hypo) for hypo in all_hypos])
        if other_groups:
            all_scores = all_scores + self.lmbda*self.hamming_distance_penalty(all_hypos, 
                                                            utils.flattened(other_groups))
        inds = utils.argmax_n(all_scores, size)
        return [all_hypos[ind] for ind in inds]

    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictor(src_sentence)
        hypos = self._get_initial_hypos()
        it = 1
        while not self.stop_criterion(utils.flattened(hypos)) and it < self.max_len:
            it = it + 1
            next_hypos = []
            for i, group in enumerate(hypos):
                next_group = []
                for hypo in group:
                    if hypo.get_last_word() == utils.EOS_ID:
                        next_group.append(hypo)
                        continue 
                    for next_hypo in self._expand_hypo(hypo):
                        next_group.append(next_hypo)
                next_hypos.append(self._get_next_hypos(next_group, self.group_sizes[i], next_hypos))
            hypos = next_hypos

        return self.get_full_hypos_sorted(utils.flattened(hypos))
                        
    @staticmethod
    def hamming_distance_penalty(set1, set2):
        longest_hypo = len(max(set1 + set2, key=len))
        hypos = utils.as_ndarray(set1, min_length=longest_hypo)
        other_hypos = utils.as_ndarray(set2, min_length=longest_hypo)
        return np.apply_along_axis(lambda x: utils.hamming_distance(x, other_hypos), 1, hypos)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--diversity_groups", default=1, type=int,
                       help="If this is greater than one, promote diversity "
                       "between groups of hypotheses as in Vijayakumar et. "
                       "al. (2016). Only compatible with 'diverse_beam' decoder. "
                       "They found diversity_groups = beam size to be most "
                       "effective.")
        parser.add_argument("--diversity_reward", default=0.5, type=float,
                           help="If this is greater than zero, add reward for diversity "
                           "between groups as in Vijayakumar et. al. (2016). Only "
                           "compatible with 'diverse_beam' decoder. Setting value "
                           "equal to 0 recovers standard beam search.")


class MinHeapBeamDecoder(BeamDecoder):
    name = 'min_heap_beam'
    def __init__(self, decoder_args):
        
        super(MinHeapBeamDecoder, self).__init__(decoder_args)
        assert not self.gumbel
        self.heap = []
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.heap = []
        self.initialize_predictor(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        while not self.stop_criterion(hypos) and it < self.max_len:
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                exp_hypos = self._expand_hypo(hypo, self.beam_size)
                if len(exp_hypos) != self.beam_size:
                    eos_hypo = exp_hypos[-1]
                    heapq.heappush(self.heap, (self.get_adjusted_score(eos_hypo), eos_hypo))
                    if len(self.heap) > self.beam_size:
                        heapq.heappop(self.heap)

                for next_hypo in exp_hypos[:self.beam_size]:
                    if next_hypo.get_last_word() == utils.EOS_ID:
                        heapq.heappush(self.heap, (self.get_adjusted_score(next_hypo), next_hypo))
                        if len(self.heap) > self.beam_size:
                            heapq.heappop(self.heap)
    
                    next_hypos.append(next_hypo)
                    next_scores.append(self.get_adjusted_score(next_hypo))
            hypos = self._get_next_hypos(next_hypos, next_scores)
        
        hypos = [hypo for _, hypo in self.heap]
        return self.get_full_hypos_sorted(hypos)

    
    def _combine_posteriors_simple(self,
                                      non_zero_words,
                                      posterior,
                                      unk_prob,
                                      top_n=0,
                                      original_posterior=None):
        """        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if top_n > 0:
            non_zero_words = utils.argmax_n(posterior, top_n)
            if utils.EOS_ID not in non_zero_words:
                non_zero_words = np.append(non_zero_words, [utils.EOS_ID])

        scores_func = np.vectorize(lambda x: utils.common_get(posterior, x, unk_prob))
        scores = scores_func(non_zero_words)

        orig_scores = None
        if original_posterior is not None:
            scores_func = np.vectorize(lambda x: utils.common_get(original_posterior, x, unk_prob))
            orig_scores = scores_func(non_zero_words)

        return non_zero_words, scores, orig_scores    