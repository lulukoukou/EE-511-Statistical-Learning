#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:25:23 2018

"""
import data 
import tensorflow as tf


class LSTM():
    
    def __init__(self, char_size, lang_size, batch_size, max_seq_len, char_embedsize = 16, lang_embedsize = 4, cell_size = 27):
        self.char_size = char_size
        self.lang_size = lang_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.char_embedsize = char_embedsize
        self.lang_embedsize = lang_embedsize
        self.cell_size = cell_size
        self.learning_rate = 1e-1 ###
        self.max_iters = 100###
    
    def placeholder_inputs(self):
        
        
        """
        Creats the input placeholders .
    
        Args:
            batch_size: Integer for the batch size.
            feat_dim: Integer for the feature dimension.
    
        Returns:
            char_placeholder: charaters placeholder.
            lang_placeholder: language placeholder.
        """

        char_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len))
        lang_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len)) ####W###### size????
        ori_len_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))###
        return char_placeholder, lang_placeholder, ori_len_placeholder###
    
    
    def fill_feed_dict(self, data_set, char_ph, lang_ph, ori_len_ph):
        
        """
        Given the data for current step, fills both placeholders.
    
        Args:
            data_set: The DataSet object.
            batch_size: Integer for the batch size.
            char_ph: The image placeholder, from placeholder_inputs_feedfoward().
            lang_ph: The label placehodler, from placeholder_inputs_feedfoward().
    
        Returns:
            feed_dict: The feed dictionary maps from placeholders to values.
        """
        # TODO: Creates the feed dictionary.
        # Fill in your code below.
        [char_data, lang_data, ori_len_data] = data_set.next_batch(data_set.batch_size) # lang_data: 32 * 63 ###
        feed_dict={char_ph: char_data, lang_ph: lang_data, ori_len_ph: ori_len_data}
        
        return feed_dict


    def embedding_layer(self, char, lang):
        
        
        char_embedding = tf.get_variable("char_embedding", [self.char_size, self.char_embedsize])
        lang_embedding = tf.get_variable("lang_embedding", [self.lang_size, self.lang_embedsize])
    
        embedded_char_id = tf.nn.embedding_lookup(char_embedding, char)
        embedded_lang_id = tf.nn.embedding_lookup(lang_embedding, lang)
        
        embeddings = tf.concat([embedded_char_id, embedded_lang_id], 2)
        
        self.y = embeddings[:, 1:, :]
        self.x = embeddings[:, :-1, :]
        
        return self.x, self.y
        
        
    def recurrent_layer(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        

        self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(lstm_cell, self.x, initial_state=self.cell_init_state, time_major=False)

        
        return self.rnn_out, self.rnn_state
        
        #return reshape_rnn_out, self.rnn_state
        
    def output_layer(self):
        reshape_rnn_out = tf.reshape(self.rnn_out, [-1, self.cell_size]) # h:2016*27
        w_out = tf.get_variable('w_out', [self.cell_size, self.char_size] ) # output_weight: 27*509
        bias = tf.get_variable('bias', [self.char_size]) #1*509
        
        self.logits = tf.matmul(reshape_rnn_out, w_out) + bias #2016*509
        
        self.prediction = tf.nn.softmax(self.logits)
        return self.logits, self.prediction
        
    def compute_loss(self, lang_ph, ori_len_ph):
        
        labels = tf.reshape(lang_ph[:,1:], [-1]) # 2016*1 #
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = self.logits)
        
        ###
        #create a mask
        mask = tf.sequence_mask(ori_len_ph, self.max_seq_len - 1)
        #set loss in the paddings position to zero
        masked_loss = tf.where(tf.reshape(mask, [-1]), loss, tf.zeros_like(loss))
#        per_character_loss = tf.reshape(masked_loss, tf.shape(lang_ph[:,1:]))
#        per_sentence_loss = tf.reduce_sum(per_character_loss, 1) / tf.to_float(ori_len_ph)
        ave_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(tf.cast(mask, tf.float32))# average loss of all non-masked characters
        ###
        
        return ave_loss#per_character_loss#masked_loss#loss
    
    
        
        
        

