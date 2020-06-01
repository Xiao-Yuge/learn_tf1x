# _*_coding:utf-8_*_

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


# 线性回归
def linear_reg_test():
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.03, x_data.shape)
    y_data = np.square(x_data) + noise

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    W_L1 = tf.Variable(np.random.normal(size=(1, 10)).astype(np.float32))
    b_L1 = tf.Variable(np.zeros((1, 10)).astype(np.float32))
    activation_L1 = tf.nn.tanh(tf.matmul(x, W_L1) + b_L1)

    W_L2 = tf.Variable(np.random.normal(size=(10, 1)).astype(np.float32))
    b_L2 = tf.Variable(np.zeros((1, 1)).astype(np.float32))
    prediction = tf.nn.tanh(tf.matmul(activation_L1, W_L2) + b_L2)
    loss = tf.reduce_mean(tf.square(y - prediction))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
        prediction_value = sess.run(prediction, feed_dict={x: x_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()


def mnist_test():
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    batch_size = 50
    num_batches = mnist.train.num_examples // batch_size

    x_input = tf.placeholder(tf.float32, [None, 784])
    y_input = tf.placeholder(tf.float32, [None, 10])

    # W_L1 = tf.Variable(np.random.normal(size=(784, 1024)).astype(np.float32))
    # b_L1 = tf.Variable(np.zeros(1024).astype(np.float32))
    # hidden = tf.nn.relu(tf.matmul(x_input, W_L1) + b_L1)
    # W_L2 = tf.Variable(np.random.normal(size=(1024, 10)).astype(np.float32))
    # b_L2 = tf.Variable(np.zeros(10).astype(np.float32))
    # predict = tf.nn.softmax(tf.matmul(hidden, W_L2) + b_L2)

    W = tf.Variable(np.zeros((784, 10)).astype(np.float32))
    b = tf.Variable(np.zeros(10).astype(np.float32))
    predict = tf.nn.softmax(tf.matmul(x_input, W) + b)

    # loss = tf.reduce_mean(tf.square(y_input - predict))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=predict))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_input, 1), tf.arg_max(predict, 1)), tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(41):
            for n_batch in range(num_batches):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x_input: x_batch, y_input: y_batch})
            acc = sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_input: mnist.test.labels})
            print("Epoch:{}  Test Accuracy:{}".format(epoch, acc))


if __name__ == "__main__":
    mnist_test()
