# _*_coding:utf-8_*_

import tensorflow as tf
import numpy as np


# 创建图，启动图
def cal_graph_test():
    m1 = tf.constant([[3,3]])
    m2 = tf.constant([[2], [3]])
    op = tf.matmul(m1, m2)
    with tf.Session() as sess:
        print(sess.run(op))


# 变量
def variable_test():
    x = tf.Variable([1, 2])
    a = tf.constant([3, 3])
    sub = tf.subtract(x, a)
    add = tf.add(x, sub)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(sub))
        print(sess.run(add))


def variable_test_01():
    state = tf.Variable(0, name="Counter")
    new_value = tf.add(state, 1)
    update = tf.assign(state, new_value)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))
        for _ in range(6):
            sess.run(update)
        print(sess.run(state))


# Fetch & Feed
def fetch_test():
    input1 = tf.constant(3.)
    input2 = tf.constant(2.)
    input3 = tf.constant(5.)

    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)
    with tf.Session() as sess:
        print(sess.run([add, mul]))


def feed_test():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [3.], input2: [5.]}))


def sample():
    x_data = np.random.random(100)
    y_data = 0.3 * x_data + 0.7

    k = tf.Variable(0.)
    b = tf.Variable(0.)

    y = k * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(201):
            sess.run(train)
            if i % 20 == 0:
                print(i, sess.run([k, b]))


if __name__ == "__main__":
    sample()
