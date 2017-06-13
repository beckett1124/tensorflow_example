#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#%matplotlib inline

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print("Train data shape:", train.shape)
#print("Test data shape:", test.shape)

#print("****************************")
#print train['SalePrice'].describe()

#print("****************************")
#print train.head()

train = train[train['LotArea'] < 12000 ]

train_X = train['LotArea'].values.reshape(-1,1)
train_Y = train['SalePrice'].values.reshape(-1,1)

n_samples = train_X.shape[0]
print n_samples

learning_rate=2

training_epochs=1000

display_step = 50

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = "weight", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name = "bias", dtype=tf.float32)

pred = tf.add(tf.mul(W,X), b)

cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X, train_Y):
			sess.run(optimizer, feed_dict={X:x,Y:y})
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X,Y:train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    plt.plot(train_X,train_Y,'ro',label="original data")
    plt.plot(train_X,sess_run(W) * train_X+ sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
