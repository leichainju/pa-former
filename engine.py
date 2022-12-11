

from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from utils import tens2sent, PAD, BOS, EOS, UNK
from evaluation import calc_perplexity
from models.modules import BeamSearchScorer
from models import build_model


class Engine:
    """ Core module which maintains the training and evaluating process. The engine holds the 
    states of `model`, `optimizer`, `lr_scheduler` for training and we also need to pass the 
    `criterion` and `vocabs`(a dict for supporting various vocabs) to it. Besides, we can assign
    auxiliary tasks by passing `aux_criterion` to engine. 
    Args:
        config (CfgNode): all configs for our experiment.
        logger (Logging.Logger): which `print`s and backups the runtime state of our experiments.
        model (X2Seq.BaseModel): your full model. Up to now (2022-03), `Code2Seq` supports all 
            `X2Seq` models (with `aux-task` heads attached to the end of encoder), `X` means
            every thing (code, image, text, ...). What's more, the `model` must overwrite the
            methods: `encode` and `step_wise_decode`, which are the key for the feature of
            *highly customizable*.
        optimizer (torch.optim.Optimizer): ...
        lr_scheduler (torch.optim.lr_scheduler): ...
        criterion (X2Seq.Criterion): ...
        aux_criterion (X2Seq.Criterion): ...
        vocabs (dict): various vocabs
            key (str): name of vocab
            value (X2Seq.Vocabulary): ... 
    Usage:
    >>> engine = Engine(...)
    >>> engine.update(ex)  # update model one step, ex is batched traning data.
    >>> engine.predict(ex, policy)  # predict for a batch using specified policy (`greedy`/`beam`)
    >>> engine.
    """
    def __init__(self, config, logger, model=None, optimizer=None, lr_scheduler=None,
            criterion=None, vocabs=None):
        self.config = config
        
        # runtime settings
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.vocabs = vocabs
        self.parallel = False

        # criterion
        if isinstance(criterion, Tuple):
            self.criterion, self.aux_criterion = criterion
        else:
            self.criterion, self.aux_criterion = criterion, None
        
        # warmup settings
        self.warmup_steps = self.config.train.warmup_steps
        self.warmup = self.warmup_steps > 100
        if self.warmup:
            self.warmup_factor = self.config.optim.lr / self.warmup_steps

        # data flow settings
        self.meta_info = config.model.meta_info  # stok_seq_rep or memory_bank

        # set vocabs
        if vocabs is not None:
            for k, v in vocabs.items():
                setattr(self, k, v)
        else:
            for vocab_k in config.data.vocabs:
                setattr(self, vocab_k, None)
        assert hasattr(self, 'tgt_vocab')

        # state
        self.best_eval_result = None
        self.best_epoch = -1
        self.best_metric = -1
        self.no_improvement_epochs = -1
        self.update_steps = 0
        self.update_epochs = 0

        # parameter statistics
        if not self.config.resume:
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"number of params: {n_parameters}")

    def update(self, ex):
        """ update mode using a batch of examples"""
        self.model.train()

        # warmup, for stable training of Transformer
        if self.warmup and self.update_steps <= self.warmup_steps:
            curr_lr = self.warmup_factor * self.update_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

        # extract forward data
        forward_params = self.pack_params(self.config.model.forward_params, ex)

        tgt_seq_dy_rep = None
        if self.config.model.copy_attn:
            # mapping each src word to its idx in the "dynamic" vocab
            src_map = F.one_hot(ex['src_map']).float()  # `[batch, src_len, extra_words]`
            forward_params['src_map'] = src_map.cuda(non_blocking=True)
            # `[batch, tgt_len]`
            tgt_seq_dy_rep = ex['tgt_seq_dy_rep'].cuda(non_blocking=True)

        # forward
        scores = self.model(**forward_params)  # -> decoder_out or (decoder_out, aux_out) 
        aux_out = None
        if isinstance(scores, Sequence):
            scores, aux_out = scores[0], scores[1]

        # loss, `[batch * (tgt_len - 1)]`
        target = forward_params['tgt_seq_rep'][:, 1:].contiguous()  # ignore BOS
        tgt_seq_len = ex['tgt_seq_len'].cuda(non_blocking=True)       # `[batch]`
        if self.config.model.copy_attn:
            loss = self.criterion(scores, tgt_seq_dy_rep[:, 1:].contiguous(), target)
        else:
            loss = self.criterion(scores.view(-1, scores.size(2)), target.view(-1))

        # seq generation loss
        loss = loss.view(*scores.shape[:-1])
        loss = loss.mul(target.ne(PAD).float())  # mask pad tok
        loss = loss.sum(1)  # `[batch]`
        loss_per_token = loss.div((tgt_seq_len - 1).float()).mean().item()
        loss = loss.mean()

        # auxiliary task loss
        if aux_out is not None:
            w = self.config.model.aux_weight
            aux_params = self.pack_params(self.config.model.aux_params)
            aux_loss = self.aux_criterion(**aux_out, **aux_params)
            loss = loss + w * aux_loss

        # update the parameters of model
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.optim.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_steps += 1

        loss = loss.item()
        perplexity = calc_perplexity(loss_per_token)

        return loss, perplexity, grad_norm

    def predict(self, ex, policy='greedy'):
        """ inference a batch of examples """
        self.model.eval()

        # predict data
        predict_params = self.pack_params(self.config.model.predict_params, ex)

        src_map, shared_idxs = None, None
        if self.config.model.copy_attn:  # collect source map and alignment info
            src_map = F.one_hot(ex['src_map']).float()
            # `[batch, src_len, max_local_vocab_size]`
            src_map = src_map.cuda(non_blocking=True)
            shared_idxs = [(torch.LongTensor(dy_idxs).cuda(non_blocking=True), 
                            torch.LongTensor(tgt_idxs).cuda(non_blocking=True))
                           for (dy_idxs, tgt_idxs) in ex['shared_idxs']]

        # encode
        enc_out = self.model.encode(**predict_params, return_dict=True)

        # auto-regression decode
        if policy == 'greedy':
            dec_preds = self.greedy_decode(
                src_map=src_map,
                shared_idxs=shared_idxs,
                **enc_out
            )  # `[batch, ~tgt_len]`
        elif policy == 'beam':
            dec_preds = self.beam_search_decode(
                src_map=src_map,
                shared_idxs=shared_idxs,
                **enc_out
            )  # `[batch, ~tgt_len]`
        else:
            raise RuntimeError(f'Unsupported decoding strategy '
                               f'{self.config.test.decoding_strategy}')

        # generate predictions
        tgt_vocab = getattr(self, 'tgt_vocab')
        predictions = tens2sent(dec_preds, tgt_vocab, ex.get('dynamic_vocabs', None))
        targets = list(ex['tgt_text'])

        return predictions, targets

    def greedy_decode(self, src_map, shared_idxs, hidden=None, **kwargs):
        """ Auto regression decoding with greedy decoding strategy.
        Args:
            src_map (LongTensor): encoded src_seq by local src_vocab, shape as
                `[batch, src_len, max_local_vocab_size]`, which maps the `copy scores` over
                the src_seq into the corresponding dynamic vocab.
            shared_idxs (List[(LongTensor, LongTensor)]):
                - dy_idxs, `[num_shared_words_{i}]`
                - tgt_idxs, `[num_shared_words_{i}]`
            hidden (Tuple): `[num_layers, src_len, dim]`, for rnn
        Returns:
            LongTensor:
            - the predicted tgt word idx, `[batch, max_tgt_len + 2]`
        """
        # inference
        batch_size = src_map.size(0) if src_map is not None else kwargs[self.meta_info].size(0)
        device = src_map.device if src_map is not None else kwargs[self.meta_info].device
        tgt_vocab_size = len(getattr(self, 'tgt_vocab'))

        # auto-regression decoding
        tgt_rep = torch.full([batch_size], BOS, device=device)
        acc_gen_seq = tgt_rep.unsqueeze(1).clone()

        # some specific params need to be updated step-by-step
        cache = {}
        params = {
            'hidden': hidden,  # for rnn
        }

        for idx in range(self.config.model.max_tgt_len + 1):
            # step-wise-decode
            prediction, params = self.model.step_wise_decode(
                tgt_rep=tgt_rep,
                src_map=src_map,
                shared_idxs=shared_idxs,
                cache=cache,
                step=idx,
                **params,
                **kwargs
            )  # `[batch, (extended)_tgt_vocab_size]`

            # greedy selection, -> `[batch]`, `[batch]`
            _, tgt_rep = torch.max(prediction, dim=1)
            # gather the step-wise outputs, -> `[batch, seq_len=idx + 2]`
            acc_gen_seq = torch.cat((acc_gen_seq, tgt_rep.unsqueeze(1)), dim=1)
            # set the word from extended vocab as UNK
            tgt_rep[tgt_rep >= tgt_vocab_size] = UNK

        return acc_gen_seq  # `[batch, max_tgt_len + 2]`

    def beam_search_decode(self, src_map, shared_idxs, hidden=None, **kwargs):
        """ generates sequences for LM using beam search.
        Args:
            src_map (LongTensor): encoded src_seq by local src_vocab, shape as
                `[batch, src_len, max_local_vocab_size]`, which maps the `copy scores` over
                the src_seq into the corresponding dynamic vocab.
            shared_idxs (List[(LongTensor, LongTensor)]):
                - dy_idxs, `[num_shared_words_{i}]`
                - tgt_idxs, `[num_shared_words_{i}]`
            hidden (Tuple): `[num_layers, batch, dim]`, for rnn
            src_pad_mask (LongTensor): `[batch, src_len]`
        Returns:
            LongTensor:
            - the predicted tgt word idx, `[batch, max_tgt_len + 2]`
        """
        batch_size = src_map.size(0) if src_map is not None else kwargs[self.meta_info].size(0)
        device = src_map.device if src_map is not None else kwargs[self.meta_info].device
        tgt_vocab_size = len(getattr(self, 'tgt_vocab'))
        beam_size = self.config.test.beam_size
        batch_beam_size = batch_size * beam_size

        # how beam hypotheses are constructed, stored and sorted during generation.
        beam_scorer = BeamSearchScorer(batch_size, beam_size, device)
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e5
        beam_scores = beam_scores.view(-1)

        # expand the `memory_bank`, `src_pad_mask` and `src_map` to `[batch_beam, ...]`
        for _, iter_params in self.config.model.beam_params.items():
            data = kwargs.get(iter_params.data)
            mask = kwargs.get(iter_params.mask)
            seq_len = None
            
            # TODO: add data_repeat
            if isinstance(data, tuple):  # for rnn
                dim = data[0].size(-1)
                data_1 = data[0].unsqueeze(2).repeat(1, 1, beam_size, 1)
                data_1 = data_1.view(-1, batch_beam_size, dim)
                data_2 = data[1].unsqueeze(2).repeat(1, 1, beam_size, 1)
                data_2 = data_2.view(-1, batch_beam_size, dim)
                data = (data_1, data_2)
            else:
                seq_len = data.size(1)
                data = data.unsqueeze(1).repeat(1, beam_size, 1, 1)
                data = data.view(batch_beam_size, seq_len, -1)

            if mask is not None:
                mask = mask.unsqueeze(1).repeat(1, beam_size, 1)
                mask = mask.view(batch_beam_size, seq_len)

            kwargs[iter_params.data] = data
            kwargs[iter_params.mask] = mask

        if self.config.model.copy_attn:
            src_len = src_map.size(1)
            src_map = src_map.unsqueeze(1).repeat(1, beam_size, 1, 1)
            src_map = src_map.view(batch_beam_size, src_len, -1)
        if hidden is not None:
            if isinstance(hidden, tuple):
                hidden_expand = []
                num_layers = hidden[0].size(0)
                for h in hidden:
                    h_ = h.unsqueeze(2).repeat(1, 1, beam_size, 1)
                    h_ = h_.view(num_layers, batch_beam_size, -1)
                    hidden_expand.append(h_)
                hidden = tuple(hidden_expand)
            else:
                num_layers = hidden.size(0)
                hidden = hidden.unsqueeze(2).repeat(1, 1, beam_size, 1)
                hidden = hidden.view(num_layers, batch_beam_size, -1)
        cache = {}

        # some specific params need to be updated step-by-step
        iter_params = {
            'hidden': hidden
        }

        # auto-regression decoding
        tgt_rep = torch.full([batch_beam_size], BOS, dtype=torch.long, device=device)
        acc_gen_rep = tgt_rep.unsqueeze(1).clone()
        for idx in range(self.config.model.max_tgt_len + 1):  # BOS + `tok` * max_tgt_len
            # step-wise-decode
            prediction, iter_params = self.model.step_wise_decode(
                tgt_rep=tgt_rep,
                src_map=src_map,
                shared_idxs=shared_idxs,
                beam_size=beam_size,
                cache=cache,
                step=idx,
                **iter_params,
                **kwargs
            )  # `[batch, (extended)_tgt_vocab_size]`

            # `[batch_beam, (extended)_vocab_size]`
            next_token_scores = prediction.log() + beam_scores[:, None]

            # reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )  # `[batch, 2 * beam_size]`, `[batch, 2 * beam_size]` (idxs)
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size

            # update the beam scorer
            beam_outputs = beam_scorer.process(
                acc_gen_rep,        # `[batch * beam_size, curr_len = idx + 1]`
                next_token_scores,  # `[batch, 2 * beam_size]`, values of topK
                next_tokens,        # `[batch, 2 * beam_size]`, idx of topK over vocab
                next_indices,       # `[batch, 2 * beam_size]`, idx of topK over beam
                pad_idx=PAD,
                eos_idx=EOS,
            )

            beam_scores = beam_outputs['next_beam_scores']  # `[batch_beam]`
            tgt_rep = beam_outputs['next_beam_tokens']  # `[batch_beam]`
            beam_idx = beam_outputs['next_beam_indices']    # `[batch_beam]`

            # update specific params
            iter_params['beam_idx'] = beam_idx

            # merge the outputs
            acc_gen_rep = torch.cat((acc_gen_rep[beam_idx], tgt_rep.unsqueeze(1)), dim=-1)
            tgt_rep[tgt_rep >= tgt_vocab_size] = UNK

        pred_seqs = beam_scorer.finalize(
            acc_gen_rep,
            beam_scores,
            PAD, EOS
        )

        return pred_seqs

    def reset_state(self, update_steps=0, update_epochs=0, best_eval_result=None):
        self.update_steps = update_steps
        self.update_epochs = update_epochs

        if best_eval_result is None:
            self.best_eval_result = None
            self.best_epoch = -1
            self.best_metric = -1
            self.no_improvement_epochs = -1
        else:
            self.best_metric = best_eval_result[self.config.test.main_metric]
            self.no_improvement_epochs = 0
            self.best_eval_result = best_eval_result
            self.best_epoch = best_eval_result['epoch']

    def update_state(self, eval_result):
        self.update_epochs += 1
        curr_metric = eval_result[self.config.test.main_metric]

        if curr_metric > self.best_metric:
            self.best_metric = curr_metric
            self.no_improvement_epochs = 0
            self.best_eval_result = eval_result
            self.best_epoch = eval_result['epoch']
        else:
            self.no_improvement_epochs += 1

    def save_checkpoint(self, filename, epoch):
        params = {
            'model':            self.model.state_dict(),
            'optimizer':        self.optimizer.state_dict(),
            'lr_scheduler':     self.lr_scheduler.state_dict(),
            'vocabs':           self.pack_vocabs(),
            'epoch':            epoch,
            'update_steps':     self.update_steps,
            'config':           self.config,
            'best_eval_result': self.best_eval_result
        }

        try:
            torch.save(params, filename)
        except IOError:
            self.logger.warning('Saving checkpoint failed ... continuing anyway.')

    def pack_vocabs(self):
        vocabs = {}
        for vocab_tag in self.config.data.vocabs:
            vocab = getattr(self, vocab_tag)
            vocabs[vocab_tag] = vocab
        return vocabs

    @staticmethod
    def pack_params(params, ex):
        packed_params = {}
        for fn_key, data_key in params.items():
            if ex[data_key] is None:
                packed_params[fn_key] = None
            else:
                if isinstance(ex[data_key], torch.Tensor):
                    packed_params[fn_key] = ex[data_key].cuda(non_blocking=True)
                else: # for dgl.Graph
                    packed_params[fn_key] = ex[data_key].to('cuda')
        return packed_params

    def load_checkpoint(self, ckpt_file):
        self.logger.info(f'Loading checkpoint from {ckpt_file}')
        checkpoint = torch.load(ckpt_file, map_location='cpu')

        # load vocabs
        vocabs = checkpoint['vocabs']
        self.config.defrost()
        for vocab_tag in self.config.data.vocabs:
            vocab = vocabs[vocab_tag]
            setattr(self, vocab_tag, vocab)
            self.config.model.embedding.update({f'{vocab_tag}_size': len(vocab)})
            self.logger.info(f'reload vocabs: {vocab_tag} size: {len(vocab)}')
        self.config.freeze()

        # build model and load weights from checkpoint
        self.model = build_model(self.config.model)
        self.model.cuda()
        self.logger.info('build model w.r.t. the reloaded vocabs')
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        self.logger.info(msg)

        # load optimizer and lr_scheduler
        if self.config.mode == 'train':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.config.defrost()
            self.config.train.start_epoch = checkpoint['epoch'] + 1
            self.config.freeze()

        if self.config.resume:
            self.reset_state(
                update_steps=checkpoint.get('update_steps'),
                update_epochs=checkpoint.get('epoch')
            )

        # self.best_eval_result = checkpoint['best_eval_result']
        self.logger.info(f"Loaded successfully {ckpt_file}(epoch={checkpoint['epoch']})")
        del checkpoint
        torch.cuda.empty_cache()
