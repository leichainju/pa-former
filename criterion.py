""" criterion for seq generation and some auxiliary tasks """

from torch import nn

from utils import UNK


def build_criterion(config):
    if config.model.copy_attn:
        criterion = CopyGeneratorCriterion(config.data.tgt_vocab_size,
                                           config.criterion.force_copy)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    aux_criterion = None
    
    return criterion, aux_criterion


class CopyGeneratorCriterion:
    """ Copy generator criterion """
    def __init__(self, vocab_size, force_copy, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size

    def __call__(self, scores, dy_tgt, tgt):
        """
        Args:
            scores (FloatTensor): `[batch, tgt_len - 1, vocab_size]`
            dy_tgt (LongTensor): `[batch, tgt_len - 1]`
            tgt (LongTensor): `[batch, tgt_len - 1]`
        Returns:
            `[batch]`
        """
        dy_tgt = dy_tgt.view(-1)
        tgt = tgt.view(-1)
        scores = scores.view(-1, scores.size(2))

        # Compute unks in align and target for readability
        dy_tgt_unk = dy_tgt.eq(UNK).float()
        dy_tgt_not_unk = dy_tgt.ne(UNK).float()
        tgt_unk = tgt.eq(UNK).float()
        tgt_not_unk = tgt.ne(UNK).float()

        # copy probability of tokens in source, `[batch * (tgt_len - 1)]`
        copy_scores = scores.gather(1, dy_tgt.view(-1, 1) + self.offset).view(-1)
        # et scores for unk to 0 and add eps
        copy_scores = copy_scores * dy_tgt_not_unk + self.eps

        # get scores for tokens in target
        ori_scores = scores.gather(1, tgt.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            final_scores = copy_scores + ori_scores * tgt_not_unk
            # Add score for when word is unk in both align and tgt
            final_scores = final_scores + ori_scores * dy_tgt_unk * tgt_unk
        else:  # make nonsense
            # Forced copy. Add only probability for not-copied tokens
            final_scores = copy_scores + ori_scores * dy_tgt_unk

        loss = - final_scores.log()
        return loss
