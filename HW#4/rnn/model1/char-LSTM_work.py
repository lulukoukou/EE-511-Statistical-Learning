#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import tensorflow as tf
import numpy as np

MAX_LENGTH = 160
BATCH_SIZE = 32

class data(object):
    def __init__(self, filename):
        self.filename = filename
        self.epoches_completed = 0
        self.index_in_epoch = 0

        self.char_ids = []
        self.lang_ids = []
        self.seq_length = []
        self.num_samples = 0

    def preprocess(self):
        sentence_start = '<S>'
        sentence_end = '</S>'
        unk = 'unk'
        voca = {sentence_start: 0, sentence_end: 0}

        with open('./train.tsv', encoding="utf_8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
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



        n_char = 1
        n_lang = 0
        char_map = {}
        for key in voca.keys():
            char_map[key] = n_char
            n_char += 1
        lang_map = {}

        with open(self.filename, encoding="utf_8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                if line[0] not in lang_map:
                    lang_map[line[0]] = n_lang
                    n_lang += 1
                self.lang_ids.append([lang_map[line[0]]]*MAX_LENGTH)

                n_padding = MAX_LENGTH - len(line[1]) - 2
                self.seq_length.append([1]*(len(line[1])+2) + [0]*(n_padding))
                seq = [char_map[sentence_start]]
                for char in line[1]:
                    if char in temp:
                        seq.append(char_map[unk])
                    else:
                        seq.append(char_map[char])
                seq.append(char_map[sentence_end])
                seq += [0] * n_padding
                self.char_ids.append(seq)
        self.char_ids = np.array(self.char_ids)
        self.lang_ids = np.array(self.lang_ids)
        self.seq_length = np.array(self.seq_length)
        self.num_samples = self.char_ids.shape[0]

        return n_char, n_lang

    @property
    def char_ids_(self):
        return self.char_ids

    @property
    def lang_ids_(self):
        return self.lang_ids
    @property
    def seq_length_(self):
        return self.seq_length
    # cite from HW#3 binary_data.py
    def next_batch(self, batch_size, shuffle=True):
        """Return the next batch_size sized samples."""
        start = self.index_in_epoch
        if start == 0 and shuffle:
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)
            self.char_ids = self.char_ids_[perm]
            self.lang_ids = self.lang_ids_[perm]
            self.seq_length = self.seq_length_[perm]
        # Extracts the next batch data.
        if start + batch_size > self.num_samples:
            # One epoch is done.
            self.epoches_completed += 1
            self.index_in_epoch = 0
            start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch += batch_size
        x_c = np.array(self.char_ids[start:end])
        x_l = np.array(self.lang_ids[start:end])
        x_s = np.array(self.seq_length[start:end])
        y_c = np.array(self.char_ids[start:end])
##        print (x_c.shape)
##        print (x_l.shape)
##        print (x_s.shape)
##        print (y_c.shape)
        return x_c, x_l, x_s, y_c

class rnn_lstm(object):
    def __init__(self, batch_size, num_steps, char_size, lang_size, char_embedding, lang_embedding, cell_size):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.char_size = char_size
        self.lang_size = lang_size
        self.char_embedding = char_embedding
        self.lang_embedding = lang_embedding
        self.input_size = char_embedding + lang_embedding
        self.cell_size = cell_size
        self.x = tf.zeros([self.batch_size, self.num_steps, self.input_size], dtype=tf.float32)

        with tf.name_scope('inputs'):
            self.x_char = tf.placeholder(tf.int32, shape=[None, self.num_steps])
            self.x_lang = tf.placeholder(tf.int32, shape=[None, self.num_steps])
            self.len = tf.placeholder(tf.int32, shape=[BATCH_SIZE])###
            self.y = tf.placeholder(tf.int32, shape=[None,self.num_steps])
        with tf.variable_scope('embedding'):
            self.embedding_layer()
        with tf.variable_scope('lstm'):
            self.lstm_layer()
        with tf.variable_scope('output'):
            self.output_layer()
        with tf.name_scope('loss'):
            self.compute_loss()
        with tf.name_scope('train_op'):
            self.train_op()

    def embedding_layer(self):
        char_embed= tf.get_variable("char_embedding", [self.char_size, self.char_embedding])
        lang_embed = tf.get_variable("lang_embedding", [self.lang_size, self.lang_embedding])

        embedding_char_id_x = tf.nn.embedding_lookup(char_embed, self.x_char)
        embedding_lang_id_x = tf.nn.embedding_lookup(lang_embed, self.x_lang)
        embedding= tf.concat([embedding_char_id_x, embedding_lang_id_x], 2)          # concatenate character-language pair
        
        self.x = embedding[:, :-1, :]
        self.y = embedding[:, 1:, :]
        
    def lstm_layer(self):
        #lstm_cell = tf.nn.rnn_cell.GRUCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_output, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.x, initial_state=self.cell_init_state)
        
    def output_layer(self):
##        self.cell_output = tf.transpose(self.cell_output, [1,0,2])
        output = tf.reshape(self.cell_output, [-1, self.cell_size])
        w = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        weight = tf.get_variable(shape=[self.cell_size, self.char_size], initializer=w, name='weight')
        b = tf.constant_initializer(0.1)
        bias = tf.get_variable(shape=[self.char_size], initializer=b, name='bias')

        self.logit = tf.matmul(output, weight) + bias

        self.prediction = tf.nn.softmax(self.logit)
##        self.prediction = tf.reshape(self.prediction, [self.batch_size, self.num_steps, self.char_size])
        
    def compute_loss(self):
        #self.y = tf.reshape(self.y[:,1:], [-1])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.y))

    def train_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
def evaluation(sess, data, model):
    
    # TODO: Fills in how you compute the accuracy.
    count = 0  # Counts the number of correct predictions.
    epoch = (data.num_samples//BATCH_SIZE)
    total = epoch * BATCH_SIZE
    for step in range(epoch):
        x_c, x_l, x_s, y_c = data.next_batch(BATCH_SIZE)
        if train_data.epoches_completed > epoch:
            epoch = train_data.epoches_completed
            print(epoch)
        if step == 0:
            feed_dict = {model.x_char: x_c,model.x_lang: x_l,model.y: y_c} # create initial state model.weight4loss: x_s,
        else:
            feed_dict = {model.x_char: x_c,model.x_lang: x_l,model.y: y_c,model.cell_init_state: state} # use previous state as the current state model.weight4loss: x_s,
        _, loss, state, prediction = sess.run([model.train_op, model.loss, model.cell_final_state, model.prediction],feed_dict=feed_dict)
        count += loss  
    loss = count/total
    perplexity = math.pow(2,loss)
    print (perplexity)
    return perplexity
        
if __name__ == '__main__':
    train_data = data('./train.tsv')
    char_size, lang_size = train_data.preprocess()

    model = rnn_lstm(batch_size=BATCH_SIZE, num_steps=MAX_LENGTH, char_size=char_size,lang_size=lang_size, char_embedding=15, lang_embedding=6, cell_size=1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    if not os.path.exists('.\model'):
        os.mkdir('.\model')
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)

    epoch = 0

    for step in range(500):

        train_feed_dict = {model.x_char: x_c,model.x_lang: x_l,model.y: y_c} # create initial state model.weight4loss: x_s,
        _, loss_val = sess.run([model.train_op, model.loss],feed_dict=train_feed_dict)

        if step % 10 == 0:
            print('train perplexity at step %s is: %s '  %(step,evaluation(sess,train_data, model)))
            print('valid perplexity at step %s is: %s '  %(step,evaluation(sess,valid_data, model)))  
            print('======')
        checkpoint_file = os.path.join('.\model', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
##    restore_file = os.path.join('.\model', 'model.ckpt')
##    saver.restore(sess, restore_file)
##    valid_data = data('./valid.tsv')
##    char_size, lang_size = valid_data.preprocess()
##    

