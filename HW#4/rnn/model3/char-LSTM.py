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

MAX_LENGTH = 160
BATCH_SIZE = 128

class rnn_lstm(object):
    def __init__(self, state_size, num_classes,ckpt_path='./',model_name='model3'):
        self.state_size = state_size
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        def __graph__():
            tf.reset_default_graph()
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)

            embs = tf.get_variable('emb', [num_classes, state_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            
            init_state = tf.placeholder(shape=[2, None, state_size], dtype=tf.float32, name='initial_state')
            xav_init = tf.contrib.layers.xavier_initializer

            W = tf.get_variable('W', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            U = tf.get_variable('U', shape=[4, self.state_size, self.state_size], initializer=xav_init())

            def step(prev, x):
                st_1, ct_1 = tf.unstack(prev)
                i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(st_1,W[0]))
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(st_1,W[1]))
                o = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(st_1,W[2]))
                g = tf.tanh(tf.matmul(x,U[3]) + tf.matmul(st_1,W[3]))
                ct = ct_1*f + g*i
                st = tf.tanh(ct)*o
                return tf.stack([st, ct])
            states = tf.scan(step,tf.transpose(rnn_inputs, [1,0,2]),initializer=init_state)
            V = tf.get_variable('V', shape=[state_size, num_classes],initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes],initializer=tf.constant_initializer(0.))
            last_state = states[-1]
            states = tf.transpose(states, [1,2,0,3])[0]
            states_reshaped = tf.reshape(states, [-1, state_size])
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
        __graph__()
        

    def train(self,epoch=10):
        X, Y, idx2w, w2idx = data.load_data('./train')
        X_v, Y_v, idx2w_v, w2idx_v = data.load_data('./valid')
        def next_batch(x, y, batch_size):
            sample_idx = sample(list(np.arange(len(x))), batch_size)
            yield x[sample_idx], y[sample_idx]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            valid_loss = 0
            step_per_epoch = (self.num_classes//BATCH_SIZE)
            #print ("step_per_epoch is %s" %(step_per_epoch)) 
            for i in range(epoch):

                for j in range(step_per_epoch):
                    train_set = next_batch(X, Y ,BATCH_SIZE)
                    xs, ys = train_set.__next__()
                    valid_set = next_batch(X_v, Y_v ,BATCH_SIZE)
                    xs_v, ys_v = valid_set.__next__()
                    if j == 0:
                        _, train_loss_, last= sess.run([self.train_op, self.loss,self.last_state],feed_dict = {self.xs_ : xs,self.ys_ : ys.flatten(),self.init_state : np.zeros([2, BATCH_SIZE, self.state_size])})
                    else:
                        _, train_loss_, last= sess.run([self.train_op, self.loss,self.last_state],feed_dict = {self.xs_ : xs,self.ys_ : ys.flatten(),self.init_state : last})
                    train_loss += train_loss_
                train_loss = train_loss/step_per_epoch
                print('train loss at step %s is: %s' %(i,train_loss))
                perplexity = math.pow(2,train_loss)
                print ('train perplexity is: %s' %(perplexity))
                train_loss = 0

                if i == 0:
                    valid_loss_, last= sess.run([self.loss,self.last_state],feed_dict = {self.xs_ : xs_v,self.ys_ : ys_v.flatten(),self.init_state : np.zeros([2, BATCH_SIZE, self.state_size])})
                else:
                    valid_loss_, last= sess.run([self.loss,self.last_state],feed_dict = {self.xs_ : xs_v,self.ys_ : ys_v.flatten(),self.init_state : last})
                valid_loss += valid_loss_
                

            valid_loss = valid_loss/epoch  
            print('valid loss is: %s' %(valid_loss))
            perplexity = math.pow(2,valid_loss)
            print ('valid perplexity is: %s' %(perplexity))
            valid_loss = 0

            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=i)

if __name__ == '__main__':
    
    X, Y, idx2w, w2idx = data.load_data('./train')
    seqlen = X.shape[0]
    model = rnn_lstm(state_size = 32, num_classes=len(idx2w))
    model.train()
    
