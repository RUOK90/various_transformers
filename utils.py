import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets
import time
import numpy as np
from config import *
import math


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class _MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size):
                self.batches.append(sorted(b, key=self.sort_key))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class BeamSearchNode(object):
    def __init__(self, prev_node, word_idx, cum_log_prob, length, stop):
        self.prev_node = prev_node
        self.word_idx = word_idx
        self.cum_log_prob = cum_log_prob
        self.length = length
        self.stop = stop
        # self.score = score

    # def eval(self, alpha=1.0):
    #     reward = 0
    #     # Add here a function for shaping a reward
    #
    #     return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def get_word_idx_seqs_from_beam_search_nodes(beam_search_nodes):
    batch_list = []
    batch_size = len(beam_search_nodes)
    beam_size = len(beam_search_nodes[0])

    for sample_idx in range(batch_size):
        beam_list = []
        for beam_idx in range(beam_size):
            word_idx_seq = []
            cur_node = beam_search_nodes[sample_idx][beam_idx]
            word_idx_seq.append(cur_node.word_idx)
            while cur_node.prev_node is not None:
                cur_node = cur_node.prev_node
                word_idx_seq.insert(0, cur_node.word_idx)
            beam_list.append(word_idx_seq)
        batch_list.append(beam_list)

    return torch.Tensor(batch_list)


def get_cum_log_probs_from_beam_search_nodes(beam_search_nodes):
    batch_list = []
    batch_size = len(beam_search_nodes)
    beam_size = len(beam_search_nodes[0])

    for sample_idx in range(batch_size):
        beam_list = []
        for beam_idx in range(beam_size):
            cur_node = beam_search_nodes[sample_idx][beam_idx]
            beam_list.append(cur_node.cum_log_prob)
        batch_list.append(beam_list)

    return torch.Tensor(batch_list)


def get_length_from_beam_search_nodes(beam_search_nodes):
    batch_list = []
    batch_size = len(beam_search_nodes)
    beam_size = len(beam_search_nodes[0])

    for sample_idx in range(batch_size):
        beam_list = []
        for beam_idx in range(beam_size):
            cur_node = beam_search_nodes[sample_idx][beam_idx]
            beam_list.append(cur_node.length)
        batch_list.append(beam_list)

    return torch.Tensor(batch_list)


def get_length_penalty(length, alpha):
    return ((5 + length) ** alpha) / ((5 + 1) ** alpha)


def beam_search_decode(model, src, src_vocab, tgt_vocab):
    src_mask = (src != src_vocab.stoi[BLANK_WORD]).unsqueeze(-2)
    start_symbol = tgt_vocab.stoi[BOS_WORD]
    batch_size, src_max_len = src.size()
    decode_max_len = src_max_len + ARGS.decode_max_len_add

    memory = model.encode(src, src_mask)
    beam_search_nodes = [[BeamSearchNode(None, start_symbol, 0, 0, False) for j in range(ARGS.beam_size)] for i in range(batch_size)]

    for len_idx in range(decode_max_len):
        cum_log_probs_cand = []
        word_idxs_cand = []
        prev_cum_log_probs = get_cum_log_probs_from_beam_search_nodes(beam_search_nodes).to(ARGS.device)
        length = (get_length_from_beam_search_nodes(beam_search_nodes) + 1).to(ARGS.device)
        word_idx_seqs = get_word_idx_seqs_from_beam_search_nodes(beam_search_nodes).type_as(src.data)

        for beam_idx in range(ARGS.beam_size):
            out = model.decode(memory, src_mask, word_idx_seqs[:, beam_idx], subsequent_mask(word_idx_seqs.size(-1)).type_as(src.data))
            log_probs = model.generator(out[:, -1])
            topk_log_probs, topk_words = log_probs.topk(ARGS.beam_size, dim=-1)

            if len_idx == 0:
                stop_cnt = 0
                for sample_idx in range(batch_size):
                    for beam_idx in range(ARGS.beam_size):
                        prev_node = beam_search_nodes[sample_idx][beam_idx]
                        word_idx = topk_words[sample_idx][beam_idx]
                        if word_idx == tgt_vocab.stoi[EOS_WORD] or prev_node.stop:
                            stop = True
                            stop_cnt += 1
                            word_idx = tgt_vocab.stoi[EOS_WORD]
                            length = prev_node.length
                        else:
                            stop = False
                            length = prev_node.length + 1
                        cum_log_prob = topk_log_probs[sample_idx, beam_idx]
                        new_node = BeamSearchNode(prev_node, word_idx, cum_log_prob, length, stop)
                        beam_search_nodes[sample_idx][beam_idx] = new_node
                break
            else:
                cum_log_probs = topk_log_probs + prev_cum_log_probs
                cum_log_probs_cand.append(cum_log_probs)
                word_idxs_cand.append(topk_words)

        if len_idx != 0:
            cum_log_probs_cand = torch.cat(cum_log_probs_cand, dim=-1)
            word_idxs_cand = torch.cat(word_idxs_cand, dim=-1)
            modifier = get_length_penalty(length, ARGS.length_penalty)
            scores_cand = cum_log_probs_cand / modifier
            topk_scores, topk_scores_idxs = scores_cand.topk(ARGS.beam_size, dim=-1)
            beam_mapping_idxs = (topk_scores_idxs // ARGS.beam_size).long()

            stop_cnt = 0
            node_cache = []
            for sample_idx in range(batch_size):
                for beam_idx in range(ARGS.beam_size):
                    prev_node = beam_search_nodes[sample_idx][beam_mapping_idxs[sample_idx][beam_idx]]
                    word_idx = word_idxs_cand[sample_idx][topk_scores_idxs[sample_idx][beam_idx]]
                    if word_idx == tgt_vocab.stoi[EOS_WORD] or prev_node.stop:
                        stop = True
                        stop_cnt += 1
                        word_idx = tgt_vocab.stoi[EOS_WORD]
                        length = prev_node.length
                    else:
                        stop = False
                        length = prev_node.length + 1
                    score = topk_scores[sample_idx][beam_idx]
                    new_node = BeamSearchNode(prev_node, word_idx, cum_log_prob, length, stop)
                    node_cache.append([sample_idx, beam_idx, new_node])

            for sample_idx, beam_idx, new_node in node_cache:
                beam_search_nodes[sample_idx][beam_idx] = new_node

        if stop_cnt == batch_size * ARGS.beam_size:
            break

    # choose the best one
    output_seqs = get_word_idx_seqs_from_beam_search_nodes(beam_search_nodes)
    max_score = -math.inf
    for sample_idx in range(batch_size):
        for beam_idx in range(ARGS.beam_size):
            node = beam_search_nodes[sample_idx][beam_idx]
            length = node.length
            log_prob = node.cum_log_prob
            # length penalty
            modifier = (((5 + length) ** ARGS.length_penalty) / ((5 + 1) ** ARGS.length_penalty))
            score = log_prob / modifier

            if score > max_score:
                max_score = score
                output_seq = output_seqs[sample_idx][beam_idx]

    return result






