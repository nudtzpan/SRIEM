#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data():
    def __init__(self, data, shuffle=False, graph=None, opt=None):
        inputs = data[0]        
        cut_inputs = []
        for input_ in inputs:
            cut_inputs.append(input_[-opt.cutnum:])
        inputs, mask, len_max = data_masks(cut_inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, index):
        items, item_num = [], []
        for u_input in self.inputs[index]:
            item_num.append(len(u_input))
        max_item_num = np.max(item_num)
        for u_input in self.inputs[index]:
            items.append(u_input.tolist() + (max_item_num - len(u_input)) * [0])
        return items, self.mask[index], self.targets[index]
