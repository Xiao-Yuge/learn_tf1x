# _*_coding:utf-8_*_

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Dropout
def mnist_test():
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    batch_size = 50
    num_batches = mnist.train.num_examples // batch_size
    init_lr = 1e-3

    x_input = tf.placeholder(tf.float32, [None, 784])
    y_input = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.Variable(init_lr, dtype=tf.float32)

    W_L1 = tf.Variable(tf.truncated_normal([784, 1024], stddev=0.1), dtype=tf.float32)
    b_L1 = tf.Variable(tf.zeros([1024])+0.1, dtype=tf.float32)
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(x_input, W_L1) + b_L1), keep_prob)
    W_L2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), dtype=tf.float32)
    b_L2 = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    predict = tf.nn.softmax(tf.matmul(hidden, W_L2) + b_L2)

    # loss = tf.reduce_mean(tf.square(y_input - predict))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                          (labels=y_input, logits=predict))
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_input, 1), tf.arg_max(predict, 1)), tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(41):
            sess.run(tf.assign(lr, lr * 0.98))
            for n_batch in range(num_batches):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x_input: x_batch, y_input: y_batch, keep_prob: 0.2})
            acc = sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_input: mnist.test.labels, keep_prob: 1})
            print("Epoch:{}  Test Accuracy:{}".format(epoch, acc))


if __name__ == "__main__":
    mnist_test()
