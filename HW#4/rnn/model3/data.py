#!/usr/bin/env python3
import os
import csv
import numpy as np
import pickle as pkl

def read_lines(filename):
    with open(filename, 'r',encoding="utf_8") as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        return [ row[-1] for row in list(reader) ]

def index_(lines):
    vocab = list(set('\n'.join(lines)))
    ch2idx = { k:v for v,k in enumerate(vocab) }
    return vocab, ch2idx

def to_array(lines, seqlen, ch2idx):
    raw_data = '\n'.join(lines)
    num_chars = len(raw_data)
    data_len = num_chars//seqlen
    X = np.zeros([data_len, seqlen])
    Y = np.zeros([data_len, seqlen])
    for i in range(0, data_len):
        X[i] = np.array([ ch2idx[ch] for ch in raw_data[i*seqlen:(i+1)*seqlen] ])
        Y[i] = np.array([ ch2idx[ch] for ch in raw_data[(i*seqlen) + 1 : ((i+1)*seqlen) + 1] ])
    return X.astype(np.int32), Y.astype(np.int32)

def process_data(path, filename, seqlen=20):
    lines = read_lines(filename)
    for item in lines:
        print (item)
    idx2ch, ch2idx = index_(lines)
    X, Y = to_array(lines, seqlen, ch2idx)
    np.save(os.path.join(path+ 'idx_x.npy'), X)
    np.save(os.path.join(path+ 'idx_y.npy'), Y)
    with open(os.path.join(path+ 'metadata.pkl'), 'wb') as f:
        pkl.dump( {'idx2ch' : idx2ch, 'ch2idx' : ch2idx }, f )

if __name__ == '__main__':
    process_data(path = './train',filename = './lang_id_data/train.tsv')
    process_data(path='./valid', filename = './lang_id_data/val.tsv')

def load_data(path):
    with open(os.path.join(path + 'metadata.pkl'), 'rb') as f:
        metadata = pkl.load(f)
        X = np.load(os.path.join(path + 'idx_x.npy'))
        Y = np.load(os.path.join(path + 'idx_y.npy'))
        return X, Y, metadata['idx2ch'], metadata['ch2idx']
