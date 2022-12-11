
import torch
from torch.nn.utils.rnn import pad_sequence


class BeamSearchScorer(object):
    """ standard beam search decoding. """
    def __init__(self, batch_size, beam_size, device, len_penalty=1.0, do_early_stopping=False,
                 num_beam_hyps_to_keep=1):
        """
        Args:
            batch_size (int): batch size of `input_ids`, beam search decoding is run in parallel.
            beam_size (int): number of beams for beam search.
            device (torch.device): 'cpu' or 'cuda'
            len_penalty (float): defaults to 1.0, exponential penalty to the length. 1.0 means no
                penalty. Values < 1.0 in order to encourage the model to generate shorter sequences
            do_early_stopping (bool): defaults to `False`, whether to stop the beam search when
                at least `beam_size` sentences are finished per batch or not.
            num_beam_hyps_to_keep (int): defaults to 1, number of beam hypotheses that shall be
                returned upon calling
        """
        self.beam_size = beam_size
        self.len_penalty = len_penalty
        self.device = device
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self.beam_hyps = [
            BeamHypotheses(self.beam_size, self.len_penalty, self.do_early_stopping)
            for _ in range(batch_size)
        ]
        self.done = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

    @property
    def is_done(self) -> bool:
        return self.done.all()

    def process(self, input_idxs, next_scores, next_tokens, next_indices, pad_idx, eos_idx):
        """
        Args:
            input_idxs (LongTensor): input sequence of decoder, `[batch * beam_size, curr_len]`
            next_scores (FloatTensor): curr scores of the top `2 * beam_size` non-finished
                beam hypotheses, `[batch, 2 * beam_size]`
            next_tokens (LongTensor): word idxs of the tokens over vocab corresponding to
                the top `2 * num_beams` non-finished beam hypotheses, `[batch, 2 * beam_size]`
            next_indices (LongTensor): beam idxs indicating to which beam hypothesis the
                `next_tokens` correspond, `[batch, 2 * beam_size]`
            pad_idx (int): `PAD`
            eos_idx (int): `EOS`
        Returns:
            dict:
            -
        """
        cur_len = input_idxs.size(-1)  # 1
        batch_size = len(self.beam_hyps)
        device = input_idxs.device

        data_shape = (batch_size, self.beam_size)
        next_beam_scores = torch.zeros(data_shape, dtype=torch.float, device=device)
        next_beam_tokens = torch.zeros(data_shape, dtype=torch.long, device=device)
        next_beam_indices = torch.zeros(data_shape, dtype=torch.long, device=device)

        for batch_idx, beam_hyp in enumerate(self.beam_hyps):
            if self.done[batch_idx]:  # this example already done
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_idx
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.beam_size + next_index

                if next_token.item() == eos_idx:  # add to generated hypotheses if end of sentence
                    if beam_token_rank >= self.beam_size:
                        # if beam_token does not belong to top num_beams tokens,
                        # it should not be added
                        continue
                    beam_hyp.add(input_idxs[batch_beam_idx].clone(), next_score.item())
                else:  # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1  # count of token added

                if beam_idx >= self.beam_size:
                    # once the beam for next step is full, don't add more tokens to it.
                    break

            # check if we are done so that we can save a pad step if all(done)
            self.done[batch_idx] = self.done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
        }

    def finalize(self, final_seq, final_beam_scores, pad_idx, eos_idx):
        """
        Args:
            final_seq (LongTensor): idxs of generated seq, `[batch * beam_size, curr_len]`
            final_beam_scores (FloatTensor): `[batch * beam_size]`
            pad_idx (int):
            eos_idx (int):
        Returns:
            (LongTensor, FloatTensor):
            - decoded (LongTensor): `[batch, local_max_seq_len]`
            - best_scores (FloatTensor): `[batch]`
        """
        batch_size = len(self.beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self.beam_hyps):
            if self.done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            for beam_id in range(self.beam_size):
                batch_beam_idx = batch_idx * self.beam_size + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = final_seq[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = final_seq.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep,
                                  device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self.beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                res_idx = self.num_beam_hyps_to_keep * i + j
                sent_lengths[res_idx] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[res_idx] = best_score

        decoded = pad_sequence(best, batch_first=True, padding_value=pad_idx)
        max_len = decoded.size(-1)
        for idx, sent_len in enumerate(sent_lengths):
            if sent_len < max_len:
                decoded[idx, sent_len] = eos_idx

        return decoded


class BeamHypotheses(object):
    def __init__(self, beam_size: int, len_penalty: float, early_stopping: bool):
        """ init n-best list of hypotheses. """
        self.len_penalty = len_penalty
        self.early_stopping = early_stopping
        self.beam_size = beam_size
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """ number of hypotheses in the list. """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_log_probs: float):
        """ add a new hypothesis to the list. """
        score = sum_log_probs / (hyp.size(-1) ** self.len_penalty)

        # add, if the beams is not full or has a better score
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.beam_size:  # add the new one and remove the worst one
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_log_probs: float, cur_len: int) -> bool:
        """ If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we are
        done with this sentence.
        """
        if len(self) < self.beam_size:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_log_probs / (cur_len ** self.len_penalty)
            ret = self.worst_score >= cur_score
            return ret
