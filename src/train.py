#!/usr/bin/python2

#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np

# load data model
from data_model import model

FLAGS = None

num_hidden = model.max_timesteps
num_epochs = 10
batch_size = 128
learning_rate = 1.e-3

graph = tf.Graph()

with graph.as_default():
    def get_inference(input_list):
        with tf.name_scope('forward'):
            weights_forward = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(1./num_hidden)), name='weights')

            biases_forward = tf.Variable(tf.zeros(num_hidden), name='biases')

            forward = rnn_cell.LSTMCell(num_hidden, use_peepholes=True)


        with tf.name_scope('backward'):
            weights_backward = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(1./num_hidden)), name='weights')

            biases_backward = tf.Variable(tf.zeros(num_hidden), name='biases')

            backward = rnn_cell.LSTMCell(num_hidden, use_peepholes=True)

        with tf.name_scope('bidirectional'):
            bidirectional_h1, _, _ = bidirectional_rnn(forward, backward, input_list, dtype=tf.float32)

        with tf.name_scope('logits'):
            weights_class = tf.Variable(tf.truncated_normal([num_hidden, model.num_classes], stddev=np.sqrt(1./num_hidden)), name='weights')
            biases_class = tf.Variable(tf.zeros([model.num_classes]))
            bidirectional_h1 = [(tf.reshape(t, [batch_size, 2, num_hidden]) for t in weights_class)]
            out_h1 = [tf.reduce_sum(tf.mul(t, weights_forward), reduction_indices=1) + biases_backward for t in bidirectional_h1]
            logits = [tf.matmul(t, weights_class) + biases_class for t in out_h1]

        logits3d = tf.pack(logits)

        return logits3d


    def get_loss(logits3d, target_y, seq_lens):
        return tf.reduce_mean(ctc.ctc_loss(logits3d, target_y, seq_lens))


    def get_optimizer(loss):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    def get_eval(logits3d, target_y, seq_lens):
        logits_test = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seq_lens[0], 1])
        predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seq_lens)[0][0])
        error = tf.reduce_sum(tf.edit_distance(predictions, target_y, normalize=False))/tf.to_float(tf.size(target_y.values))
        return error, logits_test


def train():
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():

        # Initialize model input
        with tf.name_scope('input'):
            # Reshape input_x
            input_x = tf.placeholder(tf.float32, shape=(model.max_timesteps, model.num_features, model.max_timesteps))
            input_x_rs = tf.reshape(input_x, [-1, model.num_features])
            input_list = tf.split(0, model.max_timesteps, input_x_rs)

            target_indxs = tf.placeholder(tf.int64)
            target_vals = tf.placeholder(tf.int32)
            seq_lengths = tf.placeholder(tf.int32, shape=(batch_size, ))
            target_shape = tf.placeholder(tf.int64)
            target_y = tf.SparseTensor(target_indxs, target_vals, target_shape)

        # Create model and optimizer
        logits = get_inference(input_list)
        loss = get_loss(logits, target_y, seq_lengths)
        optimizer = get_optimizer(loss)
        error_rate, logits_test = get_eval(logits, target_y, seq_lengths)

    sess.run(tf.initialize_all_variables())

    for epoch in range(num_epochs):
        num_batches = model.num_samples/batch_size
        random_batches = np.random.permuation(num_batches)

        batch_errors = np.zeros(batch_size)
        for batch, orig_idx in enumerate(random_batches):
            batch_x, target_sparse, seq_lengths = model.get_batch(orig_idx, batch_size)
            t_indxs, t_vals, t_shape = target_sparse
            feed_dict = {
                input_x: batch_x,
                target_indxs: t_indxs,
                target_vals: t_vals,
                target_shape: t_shape,
                seq_lengths: seq_lengths
            }
            _, l, er, lmt = sess.run([optimizer, loss, error_rate, logits_test], feed_dict=feed_dict)
            print(np.unique(lmt))
            if (batch % 5) == 0:
                print('batch: %d, original indx: %d\nloss: %s\nerror_rate %f. ' %(batch, orig_idx, str(l), er))
        epoch_er = batch_errors.sum() / num_batches
        print("Epoch: %d, error rate: %f", epoch+1, epoch_er)


def main():
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS = parser.parse_args()
    tf.app.run()