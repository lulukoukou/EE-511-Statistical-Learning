#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import data
import tensorflow as tf
import numpy as np

MAX_LENGTH = 160
BATCH_SIZE = 128

class data(object):
    def __init__(self, filename):
        self.filename = filename
        self.epoches_completed = 0
        self.index_in_epoch = 0

        self.character_ids = []
        self.language_ids = []
        self.lang = []
        self.num_samples = 0

    def preprocess(self):
        sentence_start = '<S>'
        sentence_end = '</S>'
        unk = 'unk'
        voca = {sentence_start: 0, sentence_end: 0}

        with open(self.filename, encoding="utf_8") as fout:
            reader = csv.reader(fout, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                voca[sentence_start] += 1
                voca[sentence_end] += 1
                for char in row[1]:
                    if char in voca.keys():
                        voca[char] += 1
                    else:
                        voca[char] = 1

        temp = []
        voca_size = 0
        out_of_voca_size = 0
        voca_type = 0
        for key, value in voca.items():
            if value >= 10:
                voca_size += value
                voca_type += 1
            else:
                out_of_voca_size += value
                temp.append(key)

        for key in temp:
            del voca[key]
        voca[unk] = out_of_voca_size



        num_character = 1
        num_language = 0
        character_mapping = {}
        for key in voca.keys():
            character_mapping[key] = num_character
            num_character += 1
        language_mapping = {}

        with open(self.filename, encoding="utf_8") as fout:
            reader = csv.reader(fout, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                if line[0] not in language_mapping:
                    language_mapping[line[0]] = num_language
                    num_language += 1
                self.language_ids.append([language_mapping[line[0]]]*MAX_LENGTH)
                self.lang.append([language_mapping[line[0]]])

                n_padding = MAX_LENGTH - len(line[1]) - 2
                seq = [character_mapping[sentence_start]]
                for char in line[1]:
                    if char in temp:
                        seq.append(character_mapping[unk])
                    else:
                        seq.append(character_mapping[char])
                seq.append(character_mapping[sentence_end])
                seq += [0] * n_padding
                self.character_ids.append(seq)
        self.character_ids = np.array(self.character_ids)
        self.language_ids = np.array(self.language_ids)
        self.lang = np.array(self.lang)
        self.lang = self.lang.ravel()
        self.num_samples = self.character_ids.shape[0]
        num_samp = self.num_samples
        return num_character, num_language, num_samp

    @property
    def character_ids_(self):
        return self.character_ids

    @property
    def language_ids_(self):
        return self.language_ids
    
    @property
    def lang_(self):
        return self.lang
    
    # cite from HW#3 binary_data.py
    def next_batch(self, batch_size, shuffle=True):
        """Return the next batch_size sized samples."""
        start = self.index_in_epoch
        if start == 0 and shuffle:
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)
            self.character_ids = self.character_ids_[perm]
            self.language_ids = self.language_ids_[perm]
            self.lang = self.lang_[perm]
        # Extracts the next batch data.
        if start + batch_size > self.num_samples:
            # One epoch is done.
            self.epoches_completed += 1
            self.index_in_epoch = 0
            start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch += batch_size
        x_c = np.array(self.character_ids[start:end])
        x_l = np.array(self.language_ids[start:end])
        y_l = np.array(self.lang[start:end])
        y_l = y_l.ravel()
##        print (x_c.shape)
##        print (x_l.shape)
##        print (y_l.shape)
        return x_c, x_l, y_l
