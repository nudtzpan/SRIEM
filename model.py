#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.batch_size = opt.batchSize
        self.n_node = n_node
        self.hidden_size = opt.hiddenSize
        self.out_size = opt.outSize

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.ie_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ie_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def importance_extraction(self, hidden, mask, mask_repeat):
        query, key = self.ie_query(hidden), self.ie_key(hidden)
        similarity = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        similarity = torch.sigmoid(similarity)

        similarity_mask = similarity * mask_repeat
        similarity_each = torch.sum(similarity_mask, 2)
        length = torch.sum(mask, 1) - 1
        length[length == 0] = 1
        similarity_each /= length.unsqueeze(-1).float()

        similarity_each = torch.exp(similarity_each)
        importance = similarity_each * mask.float()
        importance /= (torch.sum(importance, dim=1, keepdim=True))
        alpha = importance.view(hidden.size(0), -1, 1)
        return alpha

    def forward(self, inputs, mask, mask_repeat):
        hidden = self.embedding(inputs)
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask.long(), 1) - 1]

        alpha = self.importance_extraction(hidden, mask, mask_repeat)

        a = torch.sum(alpha * hidden, 1)
        a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]

        scores = torch.matmul(a, b.transpose(1, 0))
        return scores


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    items, mask, targets = data.get_slice(i)
    items = trans_to_cuda(torch.Tensor(items).long())
    mask = trans_to_cuda(torch.Tensor(mask).float())
    batchsize, node_num = mask.size(0), mask.size(1)

    mask_repeat = mask.repeat(1, node_num).view(mask.size(0), node_num, node_num)#.float()
    no_mask = torch.eye(node_num)
    no_mask_repeat = no_mask.repeat(batchsize, 1).view(batchsize, node_num, node_num)#.float()
    no_mask_repeat = trans_to_cuda(torch.Tensor(no_mask_repeat).float())    
    mask_repeat = mask_repeat - no_mask_repeat
    scores = model(items, mask, mask_repeat)
    return targets, scores

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
