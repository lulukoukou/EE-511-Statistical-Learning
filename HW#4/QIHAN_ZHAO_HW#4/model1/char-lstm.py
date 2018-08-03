#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import generator_stop

import os
import csv
import sys
import math
import data
import numpy as np
import pickle as pkl
import tensorflow as tf
from random import sample

BATCH_SIZE = 128

class rnn_lstm(object):
    def __init__(self, num_classes, cell_size):
        
        self.cell_size = cell_size
        self.num_classes = num_classes
        self.path = './'
        self.model_name = 'rnn_lstm'
        
        def __lstm__():
            tf.reset_default_graph()
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)

            embs = tf.get_variable('emb', [num_classes, cell_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            
            init_state = tf.placeholder(shape=[2, None, cell_size], dtype=tf.float32, name='initial_state')
            xav_init = tf.contrib.layers.xavier_initializer

            W = tf.get_variable(shape=[4, self.cell_size, self.cell_size], initializer=xav_init(), name='W')
            U = tf.get_variable(shape=[4, self.cell_size, self.cell_size], initializer=xav_init(), name='U')

            def inter(prev, x):
                st_1, ct_1 = tf.unstack(prev)
                i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(st_1,W[0]))
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(st_1,W[1]))
                o = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(st_1,W[2]))
                g = tf.tanh(tf.matmul(x,U[3]) + tf.matmul(st_1,W[3]))
                ct = ct_1*f + g*i
                st = tf.tanh(ct)*o
                return tf.stack([st, ct])
            states = tf.scan(inter,tf.transpose(rnn_inputs, [1,0,2]),initializer=init_state)
            V = tf.get_variable(shape=[cell_size, num_classes],initializer=xav_init(),name='V')
            bo = tf.get_variable(shape=[num_classes],initializer=tf.constant_initializer(0.),name='bo')
            last_state = states[-1]
            self.num = BATCH_SIZE/55
            states = tf.transpose(states, [1,2,0,3])[0]
            states_reshaped = tf.reshape(states, [-1, cell_size])
            logits = tf.matmul(states_reshaped, V) + bo
            predictions = tf.nn.softmax(logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = ys_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

            
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state
        __lstm__()
        

    def train(self,epoch=10):
        X, Y, _, _ = data.load_data('./train')
        X_v, Y_v, _, _ = data.load_data('./valid')
        def next_batch(x, y):
            ch2id = sample(list(np.arange(len(x))), BATCH_SIZE)
            yield x[ch2id], y[ch2id]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            valid_loss = 0
            step_per_epoch = (self.num_classes//BATCH_SIZE)
            step_per_epochs = step_per_epoch*self.num
            epochs = epoch*self.num
            #print ("step_per_epoch is %s" %(step_per_epoch)) 
            for i in range(epoch):

                for j in range(step_per_epoch):
                    train_set = next_batch(X, Y)
                    xs, ys = train_set.__next__()
                    valid_set = next_batch(X_v, Y_v)
                    xs_v, ys_v = valid_set.__next__()
                    if j == 0:
                        _, train_loss_, last= sess.run([self.train_op, self.loss,self.last_state],
                                                       feed_dict = {self.xs_ : xs,self.ys_ : ys.flatten(),self.init_state : np.zeros([2, BATCH_SIZE, self.cell_size])})
                    else:
                        _, train_loss_, last= sess.run([self.train_op, self.loss,self.last_state],
                                                       feed_dict = {self.xs_ : xs,self.ys_ : ys.flatten(),self.init_state : last})
                    train_loss += train_loss_
                train_loss = (train_loss/step_per_epochs)
                print('train loss is: %s' %(train_loss))
                perplexity = math.pow(2,train_loss)
                print ('train perplexity is: %s' %(perplexity))
                train_loss = 0

                if i == 0:
                    valid_loss_, last, prediction = sess.run([self.loss,self.last_state,self.predictions],
                                                             feed_dict = {self.xs_ : xs_v,self.ys_ : ys_v.flatten(),self.init_state : np.zeros([2, BATCH_SIZE, self.cell_size])})
                else:
                    valid_loss_, last, prediction = sess.run([self.loss,self.last_state,self.predictions],
                                                             feed_dict = {self.xs_ : xs_v,self.ys_ : ys_v.flatten(),self.init_state : last})
                valid_loss += valid_loss_
                
            valid_loss = (valid_loss/epochs)
            print('valid loss is: %s' %(valid_loss))
            perplexity = math.pow(2,valid_loss)
            print ('valid perplexity is: %s' %(perplexity))
            valid_loss = 0

            saver = tf.train.Saver()
            saver.save(sess, self.path + self.model_name, global_step=i)

if __name__ == '__main__':
    
    X, Y, id2ch, ch2id = data.load_data('./train')
    seq_len = X.shape[0]
    model = rnn_lstm(cell_size = 32, num_classes=len(id2ch))
    model.train()
    
