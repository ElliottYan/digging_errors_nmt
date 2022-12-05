import utils
from decoding.core import Decoder, PartialHypothesis


class ReferenceDecoder(Decoder):
    
    name = "reference"
    def __init__(self, decoder_args):
        super(ReferenceDecoder, self).__init__(decoder_args)
        
    def decode(self, src_sentence, trgt_sentence):
        self.trgt_sentence = trgt_sentence + [utils.EOS_ID]
        self.initialize_predictor(src_sentence)

        hypo = PartialHypothesis(self.get_predictor_states(), self.calculate_stats)
        t = 0
        while hypo.get_last_word() != utils.EOS_ID:
            hypo = self._expand_hypo(hypo, debug=True)
            t += 1
        
        hypo.score = self.get_adjusted_score(hypo)
        self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()

    def _expand_hypo(self,hypo, debug=False):
        self.set_predictor_states(hypo.predictor_states)
        next_word = self.trgt_sentence[len(hypo.trgt_sentence)]
        posterior= self.apply_predictor(disable_combine=True)
        # ind = utils.binary_search(ids, k)

        max_score = utils.max_(posterior)
        # hypo.predictor_states = self.get_predictor_states()
        # hypo.score += posterior[next_word]
        # hypo.score_breakdown.append(posterior[next_word])
        # hypo.trgt_sentence += [next_word]
        new_hypo = hypo.expand(
                        next_word,
                        self.get_predictor_states(),
                        posterior[next_word] + hypo.score,
                        score_breakdown=posterior[next_word],
                        base_score=hypo.base_score,
                        cur_max=max_score,
                        )
        self.consume(next_word)
        return new_hypo