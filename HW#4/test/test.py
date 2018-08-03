#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import data 
import model
import tensorflow as tf
import numpy as np

EVAL_STEP = 10

dataset = data.DataSet('lang_id_data/train.tsv')

dataset.preprocessing()


def create_train_op(loss, learning_rate):###
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # This variable is for tracking purpose.
    global_step = tf.Variable(0, name='global_step')
    # Creates the minimization training op.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

with tf.Graph().as_default():
    
    LSTM_model = model.LSTM(dataset.char_size, dataset.lang_size, dataset.batch_size, dataset.max_seq_len)
    
    char_ph, lang_ph, ori_len_ph = LSTM_model.placeholder_inputs()###
    
    embedding_op = LSTM_model.embedding_layer(char_ph, lang_ph)

    rnn_op = LSTM_model.recurrent_layer()
    
    output_op = LSTM_model.output_layer()
    
    loss_op = LSTM_model.compute_loss(lang_ph, ori_len_ph)###
    
    # Creates a training operator.
    train_op = create_train_op(loss_op, LSTM_model.learning_rate)###
        
    #initialize variables
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
                sess.run(init_op)
                
                for step in range(LSTM_model.max_iters):
                
                    train_feed_dict = LSTM_model.fill_feed_dict(dataset, char_ph, lang_ph, ori_len_ph)
                    
                    (x, y), (r_out, r_state), (_,prediction), _, loss = sess.run([embedding_op,rnn_op, output_op, train_op, loss_op],feed_dict=train_feed_dict)###
                    print (prediction.shape)
                    print (max(prediction))
                    print (loss)
                    if step % EVAL_STEP == 0:###
                        print ('loss of iteration {0}'.format(step), np.mean(loss))###
