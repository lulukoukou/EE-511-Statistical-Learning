#!/usr/bin/env python3
import os
import csv
import numpy as np
import pickle as pkl

def read_lines(filename):
    with open(filename, 'r',encoding="utf_8") as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        return [ row[-1] for row in list(reader) ]

def voca(lines):
    vocab = list(set('\n'.join(lines)))
    ch2id = { k:v for v,k in enumerate(vocab) }
    return vocab, ch2id

def embed_ch(lines, seq_len, ch2id):
    raw_data = '\n'.join(lines)
    num_chars = len(raw_data)
    data_len = num_chars//seq_len
    X = np.zeros([data_len, seq_len])
    Y = np.zeros([data_len, seq_len])
    for i in range(0, data_len):
        X[i] = np.array([ ch2id[ch] for ch in raw_data[i*seq_len:(i+1)*seq_len] ])
        Y[i] = np.array([ ch2id[ch] for ch in raw_data[(i*seq_len) + 1 : ((i+1)*seq_len) + 1] ])
    return X.astype(np.int32), Y.astype(np.int32)

def process_data(path, filename, seq_len=20):
    lines = read_lines(filename)
    id2ch, ch2id = voca(lines)
    X, Y = embed_ch(lines, seq_len, ch2id)
    np.save(os.path.join(path+ 'id_x.npy'), X)
    np.save(os.path.join(path+ 'id_y.npy'), Y)
    with open(os.path.join(path+ 'metadata.pkl'), 'wb') as f:
        pkl.dump( {'id2ch' : id2ch, 'ch2id' : ch2id }, f )

if __name__ == '__main__':
    process_data(path = './train',filename = './lang_id_data/train.tsv')
    process_data(path='./valid', filename = './lang_id_data/val.tsv')

def load_data(path):
    with open(os.path.join(path + 'metadata.pkl'), 'rb') as f:
        metadata = pkl.load(f)
        X = np.load(os.path.join(path + 'id_x.npy'))
        Y = np.load(os.path.join(path + 'id_y.npy'))
        return X, Y, metadata['id2ch'], metadata['ch2id']
