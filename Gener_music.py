# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:50:19 2018

@author: 罗骏
"""
import tensorflow as tf
import numpy as np
import pretty_midi
from tensorflow.contrib import data
CLASS_NUM = 82
INPUT_LENGTH = 514
# total_row = 2400
stride = 3
total_batch = 250
batch_size = 20
gen_hidden_dim = ((CLASS_NUM-1)//stride**3+1) * ((INPUT_LENGTH-1)//stride**3+1) * (4**3) # 4*20*4**3
disc_hidden_dim = ((CLASS_NUM-1)//stride**3+1) * ((INPUT_LENGTH-1)//stride**3+1) * (4**3)
noise_dim = ((CLASS_NUM-1)//stride**3+1) * ((INPUT_LENGTH-1)//stride**3+1)
LAMBDA = 10 # Gradient penalty lambda hyperparameter
learning_rate = 0.0002
image_dim = CLASS_NUM * INPUT_LENGTH  # need 82*514
DIM = 3 # Model dimensionality

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

def lrelu(x, leak=0.3, name="lrelu"):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def construct_filter_bias(style, input_dim, output_dim, filter_size, stride = 2):
    if style == 'conv2d':
        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2
        filter_value = np.random.uniform(
        low = -np.sqrt(4./(fan_in+fan_out)) * np.sqrt(3),
        high = np.sqrt(4./(fan_in+fan_out)) * np.sqrt(3),
        size = (filter_size, filter_size, input_dim, output_dim)
        ).astype('float32')
        
    elif style == 'deconv2d':
        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)
        
        filter_value = np.random.uniform(
        low = -np.sqrt(4./(fan_in+fan_out)) * np.sqrt(3),
        high = np.sqrt(4./(fan_in+fan_out)) * np.sqrt(3),
        size = (filter_size, filter_size, output_dim, input_dim)
        ).astype('float32')
    bias_value = np.zeros(output_dim, dtype='float32')

    return filter_value, bias_value

deconv2d_filter_value, deconv2d_bias_value, conv2d_filter_value, conv2d_bias_value = [], [], [], []
for i in range(3):
    a, b = construct_filter_bias('deconv2d', 4**(DIM-i), 4**(DIM-i-1), 5, stride)
    deconv2d_filter_value.append(a)
    deconv2d_bias_value.append(b)
for i in range(3, 0, -1):
    a, b = construct_filter_bias('conv2d', 4**(DIM-i), 4**(DIM-i+1), 5, stride)
    conv2d_filter_value.append(a)
    conv2d_bias_value.append(b)

weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_deconv2d_filter1':tf.Variable(deconv2d_filter_value[0], name = 'Deconv2D.filter1'),
    'gen_deconv2d_filter2':tf.Variable(deconv2d_filter_value[1], name = 'Deconv2D.filter2'),
    'gen_deconv2d_filter3':tf.Variable(deconv2d_filter_value[2], name = 'Deconv2D.filter3'),
    'disc_conv2d_filter1':tf.Variable(conv2d_filter_value[0], name = 'Conv2D.filter1'),
    'disc_conv2d_filter2':tf.Variable(conv2d_filter_value[1], name = 'Conv2D.filter2'),
    'disc_conv2d_filter3':tf.Variable(conv2d_filter_value[2], name = 'Conv2D.filter3'),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_deconv2d_bias1':tf.Variable(deconv2d_bias_value[0], name = 'Deconv2D.bias1'),
    'gen_deconv2d_bias2':tf.Variable(deconv2d_bias_value[1], name = 'Deconv2D.bias2'),
    'gen_deconv2d_bias3':tf.Variable(deconv2d_bias_value[2], name = 'Deconv2D.bias3'),
    'disc_conv2d_bias1':tf.Variable(conv2d_bias_value[0], name = 'Conv2D.bias1'),
    'disc_conv2d_bias2':tf.Variable(conv2d_bias_value[1], name = 'Conv2D.bias2'),
    'disc_conv2d_bias3':tf.Variable(conv2d_bias_value[2], name = 'Conv2D.bias3'),
    'disc_out': tf.Variable(tf.zeros([1])),
}

def generator(x):
    hidden_layer1 = tf.add(tf.matmul(x, weights['gen_hidden1']), biases['gen_hidden1'])
    hidden_layer1 = lrelu(hidden_layer1)
    hidden_layer1 = tf.reshape(hidden_layer1, 
                               [-1, 4**DIM, (CLASS_NUM-1)//stride**3+1, (INPUT_LENGTH-1)//stride**3+1])
    
    input_shape = tf.shape(hidden_layer1)
    hidden_layer2 = tf.nn.conv2d_transpose(value=hidden_layer1, filter=weights['gen_deconv2d_filter1'],
            output_shape=tf.stack([input_shape[0], 4**(DIM-1), 10, 58]), 
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer2 = tf.nn.bias_add(hidden_layer2, biases['gen_deconv2d_bias1'], data_format='NCHW')
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    
    input_shape = tf.shape(hidden_layer2)
    hidden_layer3 = tf.nn.conv2d_transpose(value=hidden_layer2, filter=weights['gen_deconv2d_filter2'],
            output_shape=tf.stack([input_shape[0], 4**(DIM-2), 28, 172]), 
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer3 = tf.nn.bias_add(hidden_layer3, biases['gen_deconv2d_bias2'], data_format='NCHW')
    hidden_layer3 = tf.nn.relu(hidden_layer3)
    
    input_shape = tf.shape(hidden_layer3)
    hidden_layer4 = tf.nn.conv2d_transpose(value=hidden_layer3, filter=weights['gen_deconv2d_filter3'],
            output_shape=tf.stack([input_shape[0], 4**(DIM-3), 82, 514]), 
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer4 = tf.nn.bias_add(hidden_layer4, biases['gen_deconv2d_bias3'], data_format='NCHW')
    hidden_layer4 = tf.nn.relu(hidden_layer4)
    
    out_layer = tf.reshape(hidden_layer4, [-1, CLASS_NUM * INPUT_LENGTH])
    return out_layer


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
    hidden_layer1 = tf.reshape(inputs, [-1, 1, CLASS_NUM, INPUT_LENGTH])
    
    hidden_layer2 = tf.nn.conv2d(input=hidden_layer1, filter=weights['disc_conv2d_filter1'],
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer2 = tf.nn.bias_add(hidden_layer2, biases['disc_conv2d_bias1'], data_format='NCHW')
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    
    hidden_layer3 = tf.nn.conv2d(input=hidden_layer2, filter=weights['disc_conv2d_filter2'],
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer3 = tf.nn.bias_add(hidden_layer3, biases['disc_conv2d_bias2'], data_format='NCHW')
    hidden_layer3 = tf.nn.relu(hidden_layer3)
    
    hidden_layer4 = tf.nn.conv2d(input=hidden_layer3, filter=weights['disc_conv2d_filter3'],
            strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW'
        )
    hidden_layer4 = tf.nn.bias_add(hidden_layer4, biases['disc_conv2d_bias3'], data_format='NCHW')
    hidden_layer4 = tf.nn.relu(hidden_layer4)
    
    output_layer = tf.reshape(hidden_layer4, [-1, ((CLASS_NUM-1)//stride**3+1) * ((INPUT_LENGTH-1)//stride**3+1) * (4**3)])
    out_layer = tf.add(tf.matmul(output_layer, weights['disc_out']), biases['disc_out'])   
    return out_layer


gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

gen_sample = generator(gen_input)
gen_loss = -tf.reduce_mean(discriminator(gen_sample))
disc_loss = -tf.reduce_mean(discriminator(disc_input) - discriminator(gen_sample))

alpha = tf.random_uniform(shape=[batch_size,1], minval=0.,maxval=1.)
interpolated = disc_input + alpha*(gen_sample-disc_input)#从实际和生成的x中间位置随机抽x
inte_logit = discriminator(interpolated, reuse=True)
#把tensor中的list单独提取出来，要不下面的reduce_sum结果不对
gradients = tf.gradients(inte_logit, [interpolated])[0] 
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_loss += LAMBDA*gradient_penalty

gen_vars = [weights['gen_hidden1'], weights['gen_deconv2d_filter1'], weights['gen_deconv2d_filter2'], weights['gen_deconv2d_filter3'],
            biases['gen_hidden1'], biases['gen_deconv2d_bias1'], biases['gen_deconv2d_bias2'], biases['gen_deconv2d_bias3'],
            ]
disc_vars = [weights['disc_conv2d_filter1'], weights['disc_conv2d_filter2'], weights['disc_conv2d_filter3'], weights['disc_out'],
             biases['disc_conv2d_bias1'], biases['disc_conv2d_bias2'], biases['disc_conv2d_bias3'], biases['disc_out']
             ]

train_gen = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)
train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)


dataset = tf.data.TFRecordDataset('Dataset/dataset_1.tfrecord')
def _parse(example_proto):
    feature = {'roll' : tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, feature)
    data = tf.decode_raw(parsed['roll'], tf.uint8)
    data = tf.py_func(func=np.unpackbits, inp=[data], Tout=tf.uint8)
    data = tf.cast(data, tf.float32)
    data = tf.reshape(data, [CLASS_NUM, 600])
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
        reshape_tfdata = tfdata.reshape([-1, CLASS_NUM, 600])    #  
        cuted_tfdata = reshape_tfdata[:, :, :INPUT_LENGTH]
        reshape_cuted_tfdata = cuted_tfdata.reshape([-1, CLASS_NUM, INPUT_LENGTH])
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl = sess.run([train_gen, gen_loss], feed_dict={gen_input: z})
        for j in range(3): #
            _, dl = sess.run([train_disc, disc_loss], feed_dict={disc_input: reshape_cuted_tfdata, gen_input: z})
        if i % 20 == 0:
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