# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:50:19 2018

@author: 罗骏
"""
import tensorflow as tf
import numpy as np
import pretty_midi
#from Data import CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH
from Data import CLASS_NUM, INPUT_LENGTH
from tensorflow.contrib import data
from tensorflow.python.ops import init_ops
from Self_lib import impl
# total_row = 2400
total_batch = 250
gen_hidden_dim = 2048
disc_hidden_dim = 2048
noise_dim = 512
batch_size = 20
LAMBDA = 10 # Gradient penalty lambda hyperparameter
learning_rate = 0.0002
image_dim = CLASS_NUM * INPUT_LENGTH  #
num_hidden = 256
h_depth = num_hidden

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

def lrelu(x, leak=0.3, name="lrelu"):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, CLASS_NUM, INPUT_LENGTH], name='disc_input')
y = tf.transpose(disc_input, perm=[0, 2, 1])
x = tf.unstack(y, axis = 1)
inputs = x[0]
input_depth = inputs.get_shape().as_list()[1]

weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([num_hidden, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
    'lstm_weight': tf.get_variable('lstm_weight',shape=[input_depth + h_depth, 4 * num_hidden]),
    'gru_weight1': tf.get_variable('gru_weight1',shape=[input_depth + h_depth, 2 * num_hidden]),
    'gru_weight2': tf.get_variable('gru_weight2', shape=[input_depth + h_depth, num_hidden])
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
    'lstm_bias': tf.get_variable('lstm_bias',shape=[4 * num_hidden],
                initializer=init_ops.zeros_initializer(dtype=tf.int32))
}

def generator(x):
    hidden_layer = tf.add(tf.matmul(x, weights['gen_hidden1']), biases['gen_hidden1'])
    hidden_layer = lrelu(hidden_layer)
    out_layer = tf.add(tf.matmul(hidden_layer, weights['gen_out']), biases['gen_out'])
    return out_layer

def discriminator(x, reuse=False):
    # input as [-1, CLASS_NUM, INPUT_LENGTH]:[-1, 72, 500]
    global num_hidden
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
    y = tf.transpose(x, perm=[0, 2, 1])
    hidden_layer = tf.add(tf.matmul(impl(y, weights, biases), weights['disc_hidden1']), biases['disc_hidden1'])
    hidden_layer = lrelu(hidden_layer)
    out_layer = tf.add(tf.matmul(hidden_layer, weights['disc_out']), biases['disc_out'])
    return out_layer

gen_sample = generator(gen_input)
gen_sample_re = tf.reshape(gen_sample,[-1, CLASS_NUM, INPUT_LENGTH])
gen_loss = -tf.reduce_mean(discriminator(gen_sample_re))
disc_loss = -tf.reduce_mean(discriminator(disc_input) - discriminator(gen_sample_re))

alpha = tf.random_uniform(shape=[batch_size,1], minval=0.,maxval=1.)
interpolated = tf.reshape(disc_input,[-1, CLASS_NUM*INPUT_LENGTH]) + \
                alpha*(gen_sample-tf.reshape(disc_input,[-1, CLASS_NUM*INPUT_LENGTH]))
#从实际和生成的x中间位置随机抽x
inte_logit = discriminator(tf.reshape(interpolated, [-1, CLASS_NUM, INPUT_LENGTH]), reuse=True)
#把tensor中的list单独提取出来，要不下面的reduce_sum结果不对
gradients = tf.gradients(inte_logit, [interpolated])[0] 
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_loss += LAMBDA*gradient_penalty

gen_vars = [weights['gen_hidden1'], weights['gen_out'], biases['gen_hidden1'], biases['gen_out']]
disc_vars = [weights['disc_hidden1'], weights['disc_out'], biases['disc_hidden1'], biases['disc_out'],
             weights['lstm_weight'], biases['lstm_bias']]
train_gen = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)
train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)

dataset = tf.data.TFRecordDataset('Dataset/dataset_1.tfrecord')
def _parse(example_proto):
    feature = {'roll' : tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, feature)
    data = tf.decode_raw(parsed['roll'], tf.uint8)
    data = tf.py_func(func=np.unpackbits, inp=[data], Tout=tf.uint8)
    data = tf.cast(data, tf.float32)
    data = tf.reshape(data, [CLASS_NUM, INPUT_LENGTH])
    data = data * 2 - 1
    return data

dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=60000, count=3))
dataset = dataset.apply(data.map_and_batch(_parse, batch_size=batch_size, num_parallel_batches=2))
iterator = dataset.prefetch(batch_size).make_one_shot_iterator()
real_input_next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        tfdata = sess.run(real_input_next_element)
        reshape_tfdata = tfdata.reshape([-1, CLASS_NUM, INPUT_LENGTH])    #       
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl = sess.run([train_gen, gen_loss], feed_dict={gen_input: z})
        for j in range(3): #
            _, dl = sess.run([train_disc, disc_loss], feed_dict={disc_input: reshape_tfdata, gen_input: z})
        if i % 50 == 0:
            print('Step %i' % (i+1))
            print('Generator Loss:, Discriminator Loss: ', gl, dl)

    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    g = sess.run(gen_sample, feed_dict={gen_input: z})
    count = 0
    for i in g:
        count += 1
        pm = pretty_midi.PrettyMIDI(resolution=96)
        inst = pretty_midi.Instrument(program=0, is_drum=False, name='my piano')
        pm.instruments.append(inst)
        velocity = 100
        g_shape = i.reshape([CLASS_NUM, INPUT_LENGTH])
        for j in range(len(g_shape)):
            pitch = j + 24
            start, end = [], []
            flag = 0 # flag will be 1 while already start
            for k in range(len(g_shape[j])):
                if g_shape[j][k] == 1 and flag == 0:
                    start += [k/100]
                    flag = 1
                elif j == len(g_shape[j])-1 and g_shape[j][k] == 1:
                    end += [k/100]
                    flag = 0
                elif g_shape[j][k] == -1 and flag == 1:
                    end += [(k-1)/100]
                    flag = 0
                else: continue
            if len(start) > len(end):
                end += [len(g_shape[j])-1]
            if len(start) > 0 :
                for single_start, single_end in zip(start, end):
                    inst.notes.append(pretty_midi.Note(velocity, pitch, single_start, single_end))
        pm.write('out_'+str(count)+'.mid')