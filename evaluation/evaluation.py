import json
import math
from collections import OrderedDict, Counter

from .bleu import corpus_bleu
from .rouge import Rouge

from utils import AverageMeter


def calc_metrics(hypotheses, references, sources=None, filename=None, mode='train'):
    """ An unofficial evaluation helper.
     Args:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        sources: Map of id --> input text sequence.
        filename:
        mode (str):
            - 'dev': ignore meteor for efficiency
            - 'test': calc meteor
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = calc_accs(hypotheses[key][0], references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            pred_i = hypotheses[key]
            log_obj = OrderedDict()
            log_obj['id'] = key
            if sources is not None:
                log_obj['code'] = sources[key]
            log_obj['predictions'] = pred_i
            log_obj['references'] = references[key]
            log_obj['bleu'] = ind_bleu[key]
            log_obj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(log_obj) + '\n')

    if fw:
        fw.close()

    metrics = {
        'bleu': bleu * 100,
        'rouge_l': rouge_l * 100,
        'meteor': meteor * 100,
        'precision': precision.avg * 100,
        'recall': recall.avg * 100,
        'f1': f1.avg * 100
    }

    return metrics


def calc_accs(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = calc_acc(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def calc_acc(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_string(prediction).split()
        ground_truth_tokens = normalize_string(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def calc_perplexity(loss_per_token):
    loss_per_token = 10 if loss_per_token > 10 else loss_per_token
    perplexity = math.exp(loss_per_token)
    return perplexity


def normalize_string(s):
    """Lower text and remove extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))
