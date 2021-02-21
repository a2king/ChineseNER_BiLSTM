#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2021/2/19 14:05
# @Author: CaoYugang
import json
import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM, LR, TRAIN_FILE_PATH, TAG_IO_IX, MODEL_PATH, EPOCH, \
    LOG_PRINT_INDEX, MODEL_SAVE_INDEX, SPACE_WORD


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class BiLSTMDataset(Dataset):
    def __init__(self, FilePath, labeldict, should_invert=True):
        self.worddict = {"": 0}
        self.labeldict = labeldict
        self.dataset = []
        with open(FilePath, 'r', encoding='utf-8') as f:
            self.SenList = [s.strip() for s in f.readlines() if s.strip()]
        for line in self.SenList:
            seg = []
            tag = []
            label = SPACE_WORD
            for s in line:
                if s in labeldict:
                    if label == SPACE_WORD:
                        label = s
                    else:
                        label = SPACE_WORD
                else:
                    tag.append(self.labeldict[label])
                    if s not in self.worddict:
                        self.worddict[s] = len(self.worddict.items())
                    seg.append(self.worddict[s])
            self.dataset.append((seg, tag))
        self.should_invert = should_invert

    def __getitem__(self, index):
        sen_tag = self.dataset[index]
        return torch.tensor(sen_tag[0], dtype=torch.long), torch.tensor(sen_tag[1], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def get_worddict(self):
        return self.worddict


if __name__ == '__main__':
    training_data_set = BiLSTMDataset(
        TRAIN_FILE_PATH,
        TAG_IO_IX)
    word_to_ix = training_data_set.get_worddict()
    with open(os.path.join(MODEL_PATH, 'word_to_ix.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(word_to_ix, ensure_ascii=False))

    model = BiLSTM_CRF(len(word_to_ix), TAG_IO_IX, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    iteration_number = 0
    counter = []
    loss_history = []
    for epoch in range(EPOCH):
        for i in range(len(training_data_set)):
            sentence_in, targets = training_data_set.__getitem__(i)
            model.zero_grad()
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            if (epoch * len(training_data_set) + i) % LOG_PRINT_INDEX == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                avg = numpy.mean(loss_history)
                print("Epoch number: {},{}/{} , Current loss: {:.4f}, Avg loss: {:.4f}".format(
                    epoch, i, training_data_set.__len__(), loss.item(), avg)
                )
            if (epoch * len(training_data_set) + i) % MODEL_SAVE_INDEX == 0 and (i > 0 or epoch > 0):
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'bi-lstm-crf-{}-{}.pkl'.format(epoch, i)))
