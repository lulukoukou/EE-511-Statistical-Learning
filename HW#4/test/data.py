#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import collections
import numpy as np
import csv

class DataSet():
    
    def __init__(self, path):
        self.path = path
        self.max_seq_len = 64
        self.batch_size = 32
        self._char_ids = []
        self._lang_ids = []
        self._epoches_completed = 0
        self._index_in_epoch = 0
        self.char_size = 0
        self.lang_size = 0
        

    
    def preprocessing(self):
        with open(self.path, encoding="utf8") as tsvfile:
            dataset = pd.read_csv(tsvfile, sep = '\t', header = None, quoting=csv.QUOTE_NONE)
        
        self.num_samples = dataset.shape[0]
        
        ## Word_ids
        
        text = dataset[1].tolist()
        
        text_spl = [list(x) for x in text] 
        
        all_char = [y for x in text_spl for y in x]
        
        c = collections.Counter(all_char) 
        
        # Build vocabulary
        vocabulary = {}
        UNK_num = 0 #low frequency words
        total_num = 0 # Number of total words
        
        for k, v in c.items():
            total_num += v
            if v < 10:
                UNK_num += v
            else:
                vocabulary.update({k:v})
       
        vocabulary.update({'UNK':UNK_num}) #add low frequency word to vocabulary as a whole
        vocabulary.update({'<S>':len(dataset)}) #add start token to vocabulary
        vocabulary.update({'</S>':len(dataset)}) #add end token to vocabulary
        
        preprocessed_list = []
        list_ori_len = []
        
        
        # Split data in to charaters
        for i in text_spl:
            seq = i

            seq = ['<S>'] + i
            ori_len = len(seq)
        
            if ori_len <= self.max_seq_len - 1:
                list_to_append = ['</S>']*(self.max_seq_len-ori_len)
                seq = seq + list_to_append
                
            else:
                seq = seq[:self.max_seq_len-1]
                seq.append('</S>')
            
            preprocessed_list.append([seq])
        
            list_ori_len.append(ori_len+1)
            
        preprocessed_list_df = pd.DataFrame(preprocessed_list, columns = ['chars'])
        
        label_df = pd.DataFrame(dataset[0].tolist(), columns = ['labels'])
        
        preprocessed_data_df = pd.concat([preprocessed_list_df, pd.DataFrame(list_ori_len, columns = ['len'])], axis = 1)
        
        preprocessed_data_df = pd.concat([preprocessed_data_df, label_df], axis = 1)
        
        # Create maping    
        char_map = {}

        j = 0
        for i in vocabulary:
            char_map.update({i:j})
            j += 1
            
        self.char_size = len(char_map)
            
        word_ids = []

        for i in preprocessed_data_df['chars']:
            word_id_i = []
            
            for j in i:
                
                if j in vocabulary:
                    word_id_j = char_map[j]
                else:
                    word_id_j = char_map['UNK']
                    
                word_id_i.append(word_id_j)
                
            word_ids.append(word_id_i)
            
            
        #lang_ids
        label = dataset[0].tolist()

        c = collections.Counter(label) 
        
        label_map = {}
        j = 0
        
        for k,v in c.items():
            label_map.update({k:j})
            j += 1
            
        self.lang_size = len(label_map)
        
        label_ids = []
        for i in preprocessed_data_df['labels']:
            label_ids.append([label_map[i]]* self.max_seq_len)
        
        self._char_ids = np.array(word_ids)
        self._lang_ids = np.array(label_ids)
        self._list_ori_len = np.array(list_ori_len)###


    @property
    def char_ids(self):
        return self._char_ids

    @property
    def lang_ids(self):
        return self._lang_ids


    def next_batch(self, shuffle = True):
        
        """Return the next batch_size sized samples."""
        start = self._index_in_epoch
        
        if start == 0 and shuffle:
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)
            self._char_ids = self.char_ids[perm]
            self._lang_ids = self.lang_ids[perm]
            self._list_ori_len = self._list_ori_len[perm]###
            
        # Extracts the next batch data.
        if start + self.batch_size > self.num_samples:
            
            # One epoch is done.
            self.epoches_completed += 1
            self.index_in_epoch = 0
            start = self.index_in_epoch

        end = start + self.batch_size
        self._index_in_epoch += self.batch_size
            
        return self._char_ids[start:end], self._lang_ids[start:end], self._list_ori_len[start:end]
