#!/usr/bin/env python3
"""This contains model functions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import math
import tensorflow as tf


class Config(object):
    """This is a wrapper for all configurable parameters for model.

    Attributes:
        batch_size: Integer for the batch size.
        learning_rate: Float for the learning rate.
        image_pixel_size: Integer for the flatten image size.
        hidden1_size: Integer for the 1st hidden layer size.
        hidden2_size: Integer for the 2nd hidden layer size.
        num_class: Integer for the number of label classes.
        max_iters: Integer for the number of training iterations.
        model_dir: String for the output model dir.
    """

    def __init__(self):
        self.batch_size = 100
        self.learning_rate = 1e-3
        # Each image is 28x28.
        self.image_pixel_size = 784
#You can modify this to change the network.
        self.hidden1_size = 128
        self.hidden2_size = 128
#You need to change this for multi-class.
        self.num_class = 10
#You need to change this.
        self.max_iters = 400
        self.model_dir = './model'
        # TODO: You can adds more configurable attributes here.


def placeholder_inputs_feedfoward(batch_size, feat_dim):
    """Creats the input placeholders for the feedfoward neural network.

    Args:
        batch_size: Integer for the batch size.
        feat_dim: Integer for the feature dimension.

    Returns:
        image_placeholder: Image placeholder.
        label_placeholder: Label placeholder.
    """
# TODO: Creates two placeholders.
    # API link (https://www.tensorflow.org/api_docs/python/tf/placeholder).
    # Fill in your code below.
    image_placeholder = tf.placeholder(tf.float32, shape=[None, feat_dim])
    label_placeholder = tf.placeholder(tf.int32)
    return image_placeholder, label_placeholder


def fill_feed_dict(data_set, batch_size, image_ph, label_ph):
    """Given the data for current step, fills both placeholders.

    Args:
        data_set: The DataSet object.
        batch_size: Integer for the batch size.
        image_ph: The image placeholder, from placeholder_inputs_feedfoward().
        label_ph: The label placehodler, from placeholder_inputs_feedfoward().

    Returns:
        feed_dict: The feed dictionary maps from placeholders to values.
    """
    # Creates the feed dictionary.
    x, y = data_set.next_batch(batch_size)
    feed_dict = {image_ph:x,label_ph:y}
    return feed_dict


def feed_forward_net(images, config):
    """Creates a feedforward neuralnetwork.

    Args:
        images: Image placeholder.
        config: The Config object contains model parameters.

    Returns:
        logits: Output tensor with logits.
    """
# Creates the 1st feed fully-connected layer with ReLU activation.
# Refer to https://www.tensorflow.org/versions/r1.2/get_started/mnist/beginners
# which inspirs to apply truncated_normal for randomly generated weights
    with tf.variable_scope('hidden_layer_1'):
        # Creates two variables:
        # 1) hidden1_weights with size [image_pixel_size, hidden1_size].
        # 2) hidden1_biases with size [hidden1_size].
        # Fill in your code below.
        # Performs feedforward on images using the two variables defined above.
        # Fill in your code below.
        hidden1_weights = tf.Variable(tf.truncated_normal([config.image_pixel_size, config.hidden1_size], stddev=1.0 / math.sqrt(float(config.image_pixel_size))))
        hidden1_biases = tf.Variable(tf.zeros([config.hidden1_size]))
        hidden_1_layer = {'weights':hidden1_weights,'biases':hidden1_biases}
        hidden1 = tf.nn.relu(tf.matmul(images, hidden_1_layer['weights'])+ hidden_1_layer['biases'])

# TODO: Creates the 2nd feed fully-connected layer with ReLU activation.
    with tf.variable_scope('hidden_layer_2'):
        # Creates two variables:
        # 1) hidden2_weights with size [hidden1_size, hidden2_size].
        # 2) hidden2_biases with size [hidden2_size].
        # Fill in your code below.
        # Performs feedforward on hidden1 using the two variables defined above.
        # Fill in your code below.
        hidden2_weights = tf.Variable(tf.truncated_normal([config.hidden1_size, config.hidden2_size], stddev=1.0 / math.sqrt(float(config.image_pixel_size))))
        hidden2_biases = tf.Variable(tf.zeros([config.hidden2_size]))
        hidden_2_layer = {'weights':hidden2_weights,'biases':hidden2_biases}
        hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden_2_layer['weights'])+ hidden_2_layer['biases'])

# TODO: Creates the pen-ultimate linear layer.
    with tf.variable_scope('logits_layer'):
        # Creates two variables:
        # 1) logits_weights with size [config.hidden2_size, config.num_class].
        # 2) logits_biases with size [config.num_class].
        # Fill in your code below.
        # Performs linear projection on hidden2 using the two variables above.
        # Fill in your code below.
        logits_weights = tf.Variable(tf.truncated_normal([config.hidden2_size, config.num_class], stddev=1.0 / math.sqrt(float(config.image_pixel_size))))
        logits_biases = tf.Variable(tf.zeros([config.num_class]))
        logits_layer = {'weights':logits_weights,'biases':logits_biases}
        logits = tf.matmul(hidden2,logits_layer['weights']) + logits_layer['biases']
    return logits


def compute_loss(logits, labels):
    """Computes the cross entropy loss between logits and labels.

    Args:
        logits: A [batch_size, num_class] sized float tensor.
        labels: A [batch_size] sized integer tensor.

    Returns:
        loss: Loss tensor.
    """
    # Computes the cross-entropy loss.
    # API (https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits).
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


def evaluation(sess, image_ph, label_ph, data_set, eval_op):
    """Runs one full evaluation and computes accuracy.

    Args:
        sess: The session object.
        image_ph: The image placeholder.
        label_ph: The label placeholder.
        data_set: The DataSet object.
        eval_op: The evaluation accuracy op.

    Returns:
        accuracy: Float scalar for the prediction accuracy.
    """
    # Fills in how you compute the accuracy.
    count = 0  # Counts the number of correct predictions and add them up per batch
    config = Config()
    batch_size = config.batch_size
    epoch = (data_set.num_samples//batch_size) # interger to interate
    total = epoch * batch_size 
    for step in range(epoch):
        feed_dict = fill_feed_dict(data_set,batch_size,image_ph,label_ph)
        count += sess.run(eval_op, feed_dict=feed_dict)
    accuracy = count/total # calculate the precision, how many labels are correctly predicted
    return accuracy
    
