#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
from data import *
import tensorflow as tf
import numpy as np
import math

BATCH_SIZE = 1
class rnn_lstm(object):
    def __init__(self, batch_size, num_steps, char_size, lang_size, char_embedding, lang_embedding, cell_size):

        self.path = './'
        self.model_name = 'rnn_lstm'
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.char_size = char_size
        self.lang_size = lang_size
        self.char_embedding = char_embedding
        self.lang_embedding = lang_embedding
        self.input_size = char_embedding + lang_embedding
        self.cell_size = cell_size
        self.x = tf.zeros([self.batch_size, self.num_steps, self.input_size], dtype=tf.float32)
        self.num = 40
        with tf.name_scope('inputs'):
            self.x_char = tf.placeholder(tf.int32, shape=[None, self.num_steps])
            self.x_lang = tf.placeholder(tf.int32, shape=[None, self.num_steps])
            self.y = tf.placeholder(tf.int32, shape=[None])
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
        #print (embedding_lang_id_x)
        self.x = tf.concat([embedding_char_id_x, embedding_lang_id_x], 2)          # concatenate character-language pair
        
    def lstm_layer(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_output, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.x, initial_state=self.cell_init_state) # rnn lstm functions
        
    def output_layer(self):
        output = tf.reshape(self.cell_output, [-1, self.cell_size])
        w = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        weight = tf.get_variable(shape=[self.cell_size, self.lang_size], initializer=w, name='weight')
        b = tf.constant_initializer(0.1)
        bias = tf.get_variable(shape=[self.lang_size], initializer=b, name='bias')
        with tf.name_scope('projection_plus_bias'):
            self.logit = tf.matmul(output, weight) + bias
            l = self.logit
            self.logit = tf.reshape(self.logit, [self.batch_size, self.num_steps*self.lang_size])      # linear projection + bias for output from RNN layer
            l = tf.reshape(l, [self.batch_size*self.num_steps,self.lang_size])
            self.prediction = tf.nn.softmax(l)
        
    def compute_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.y)) # cross-entropy

    def train_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss) # minimize cross-entropy
        
    def train(self,epoch=10):
        train_data = data('./train.tsv')                                  # train model and validate model
        char_size, lang_size, num_samples = train_data.preprocess()

        valid_data = data('./val.tsv')
        char_size_v, lang_size_v, num_samples_v = valid_data.preprocess() # preprocess training and validation data
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            valid_loss = 0
            predict_id = []
            step_per_epoch = (num_samples//(10*BATCH_SIZE))
            epochs = epoch/self.num
            #print ("step_per_epoch is %s" %(step_per_epoch)) 
            language = ['es', 'en', 'pt', 'fr', 'ca', 'de', 'eu', 'it', 'gl']
            for i in range(epoch):
                valid_loss = []
                for j in range(step_per_epoch):                              # in each epoach iterate num_samples//BATCH_SIZE times to train
                    for k in range(9):
                        x_c, x_l, y_l = train_data.next_batch(BATCH_SIZE)
                        x_c_v, x_l_v, y_l_v = valid_data.next_batch(BATCH_SIZE)   # read by batch size
                        if j == 0:
                            _, train_loss_, last= sess.run([self.train_op, self.loss,self.cell_final_state],
                                                           feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l})
                                                           #feed_dict = {self.xs_ : xs,self.ys_ : ys.flatten(),self.init_state : np.zeros([2, BATCH_SIZE, self.cell_size])})
                            valid_loss_, last, prediction = sess.run([self.loss,self.cell_final_state,self.prediction],
                                                                 feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l})
                        else:
                            _, train_loss_, last= sess.run([self.train_op, self.loss,self.cell_final_state],
                                                           feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l, self.cell_init_state : last})
                            valid_loss_, last, prediction = sess.run([self.loss,self.cell_final_state,self.prediction],
                                                                 feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l, self.cell_init_state : last})

                        valid_loss.append(valid_loss_)
##                    train_loss += train_loss_
##                train_loss = (train_loss/step_per_epoch)
##                print('train loss is: %s' %(train_loss))
##                perplexity = math.pow(2,train_loss)
##                print ('train perplexity is: %s' %(perplexity))  # return perplexity in training data
##                train_loss = 0
                print (valid_loss)
                valid_id = valid_loss.index(max(valid_loss))
                predict_id.append(language[valid_id])
##                if i == 0:
##                    valid_loss_, last, prediction = sess.run([self.loss,self.cell_final_state,self.prediction],
##                                                             feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l})                                            
##                else:
##                    valid_loss_, last, prediction = sess.run([self.loss,self.cell_final_state,self.prediction],
##                                                             feed_dict = {model.x_char: x_c, model.x_lang: x_l, model.y: y_l, self.cell_init_state : last})
##                                                  
##                valid_loss += valid_loss_
##                
##                valid_loss = (valid_loss/epochs)
##                print('valid loss is: %s' %(valid_loss))
##                perplexity = math.pow(2,valid_loss)
##                print ('valid perplexity is: %s' %(perplexity))  # return perplexity in validation data
##                valid_loss = 0

            saver = tf.train.Saver()
            saver.save(sess, self.path + self.model_name, global_step=i)
        


if __name__ == '__main__':

    train_data = data('./train.tsv')
    char_size, lang_size, num_samples = train_data.preprocess()
    model = rnn_lstm(batch_size=BATCH_SIZE, num_steps=MAX_LENGTH, char_size=char_size,lang_size=lang_size, char_embedding=15, lang_embedding=6, cell_size=1)
    model.train() # train model and return perplexity in validation data
    
    
