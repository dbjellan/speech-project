#!/usr/bin/python2

#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import argparse

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np

# load data model
import data_model

model = data_model.Model.load_model(os.path.join(data_model.project_dir, 'corpus', 'testset'))

FLAGS = None

num_hidden = 64
num_epochs = 10
batch_size = 10
learning_rate = 1.e-3
momentum = 0.9


def bidir_ltsm(x):
    with tf.name_scope('Weights'):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, model.num_features])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, model.max_timesteps, x)

        weights_out1 = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(1./num_hidden)), name='weights')
        biases_out1 = tf.Variable(tf.zeros(num_hidden), name='biases')
        weights_out2 = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(1./num_hidden)), name='weights')
        biases_out2 = tf.Variable(tf.zeros(num_hidden), name='biases')

    with tf.name_scope('LTSM'):
        forward = rnn_cell.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)
        backward = rnn_cell.LSTMCell(num_hidden, use_peepholes=True, forget_bias=1.0)

    with tf.name_scope('Bidirectionrnn'):
        bidirectional_h1, _, _ = bidirectional_rnn(forward, backward, x, dtype=tf.float32)
        bd_h1s = [tf.reshape(t, [batch_size, 2, num_hidden]) for t in bidirectional_h1]

    with tf.name_scope('logits'):
        weights_class = tf.Variable(tf.truncated_normal([num_hidden, model.num_classes], stddev=np.sqrt(1./num_hidden)), name='weights')
        biases_class = tf.Variable(tf.zeros([model.num_classes]))
        out_h1 = [tf.reduce_sum(tf.mul(t, weights_out1), reduction_indices=1) + biases_out1 for t in bd_h1s]
        logits = [tf.matmul(t, weights_class) + biases_class for t in out_h1]

        logits3d = tf.pack(logits)

    return logits3d


def get_loss(logits3d, target_y, seq_lens):
    return tf.reduce_mean(ctc.ctc_loss(logits3d, target_y, seq_lens))


def get_optimizer(loss):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)


def get_eval(logits3d, target_y, seq_lens):
    logits_test = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seq_lens[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seq_lens)[0][0])
    error = tf.reduce_sum(tf.edit_distance(predictions, target_y, normalize=False))/tf.to_float(tf.size(target_y.values))
    return error, logits_test

graph = tf.Graph()




def train_data():
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():

        # Initialize model input
        with tf.name_scope('input'):
            input_x = tf.placeholder(tf.float32, shape=(batch_size, model.max_timesteps, model.num_features))

            # converts target y into sparse tensor of shape (batch_size, max(seq_lengths))
            target_indxs = tf.placeholder(tf.int64)
            target_vals = tf.placeholder(tf.int32)
            seq_lengths_placeholder = tf.placeholder(tf.int32, shape=(batch_size, ))
            target_shape = tf.placeholder(tf.int64)
            target_y = tf.SparseTensor(target_indxs, target_vals, target_shape)

        # Create model and optimizer
        logits = bidir_ltsm(input_x)
        loss = get_loss(logits, target_y, seq_lengths_placeholder)
        optimizer = get_optimizer(loss)
        error_rate, logits_test = get_eval(logits, target_y, seq_lengths_placeholder)
    sess.run(tf.initialize_all_variables())

    num_batches =int(model.num_samples / batch_size)

    print('Initialization done')

    for epoch in range(num_epochs):
        random_batches = np.random.permutation(range(num_batches))
        batch_errors = np.zeros(batch_size)
        for batch, orig_idx in enumerate(random_batches):
            batch_x, target_sparse, seq_lengths = model.get_batch(orig_idx, batch_size)
            t_indxs, t_vals, t_shape = target_sparse
            feed_dict = {
                input_x: batch_x,
                target_indxs: t_indxs,
                target_vals: t_vals,
                target_shape: t_shape,

                seq_lengths_placeholder: seq_lengths
            }
            _, l, er, lmt = sess.run([optimizer, loss, error_rate, logits_test], feed_dict=feed_dict)
            print(np.unique(lmt))
            if (batch % 1) == 0:
                print('batch: %d, original indx: %d\nloss: %s\nerror_rate %f. ' %(batch, orig_idx, str(l), er))
        epoch_er = batch_errors.sum() / num_batches
        print("Epoch: %d, error rate: %f", epoch+1, epoch_er)

def train():
    print('Training %d samples in batches of size %d' % (model.num_samples, batch_size,))
    print('Data info: num classes: %d, num features: %d, num timesteps: %d' % (model.num_classes, model.num_features, model.max_timesteps, ))

    #sess,loss,optimizer,error_rate,logits_test = initialize_training_data()
    #print('Initialization complete')
    #train_data(sess,loss,optimizer,error_rate,logits_test)
    train_data()
    


def main():
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS = parser.parse_args()
    main()
