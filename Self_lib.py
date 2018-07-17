# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:00:44 2018

@author: 罗骏
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import base as base_layer
batch_size = 20
num_hidden = 256

class GRUCell(base_layer.Layer):
    def __init__(self, inputs, num_units, kernel, kernel2, reuse=None):
        self._num_units = num_units
        self._gate_linear = None
        self._candidate_linear = None
        self._kernel = kernel
        self._kernel2 = kernel2
        self._bias_initializer = None
        
    def __call__(self, inputs, state):
        with tf.variable_scope("gates"):  # Reset gate and update gate.
            self._gate_linear = math_ops.matmul(array_ops.concat([inputs, state], 1), self._kernel)
        self._gate_linear = math_ops.matmul(array_ops.concat([inputs, state], 1), self._kernel)
        value = math_ops.sigmoid(math_ops.add(self._gate_linear,
                                              tf.Variable(tf.ones([batch_size, 1]))))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        with tf.variable_scope("candidate"):
            self._candidate_linear = math_ops.matmul(array_ops.concat([inputs, r_state], 1), self._kernel2)

        c = math_ops.tanh(self._candidate_linear)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

class BasicLSTMCell(base_layer.Layer):
    def __init__(self, inputs, kernel, bias, forget_bias=1.0, reuse=None, 
                 ass_type=tf.int32):
        self._forget_bias = forget_bias
        self.type = ass_type
        self._kernel = kernel
        self._bias = bias
        
    def __call__(self, inputs, state):
        sigmoid = math_ops.sigmoid
        c, h = state

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=1)
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    
        add = math_ops.add
        multiply = math_ops.multiply   
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), math_ops.tanh(j)))
        new_h = multiply(math_ops.tanh(new_c), sigmoid(o))
        new_state = (new_c, new_h)
        return new_h, new_state

def impl(x, weights, biases):
    state = (tf.Variable(tf.zeros([batch_size, num_hidden]), trainable=False),
             tf.Variable(tf.zeros([batch_size, num_hidden]), trainable=False))
    outputs = []
    x = tf.unstack(x, axis = 1)
    inputs = x[0]
    lstmcell = BasicLSTMCell(inputs, weights['lstm_weight'], biases['lstm_bias'], forget_bias=1.0)
    for input_ in x:
        output, state = lstmcell(input_, state)
        outputs.append(output)
    return outputs[-1]

def Conv2D(name, hidden_layer, filters, _biases, stride):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:
        hidden_layer = tf.transpose(hidden_layer, [0,2,3,1], name='NCHW_to_NHWC')
        hidden_layer = tf.nn.conv2d(input=hidden_layer, filter=filters,
            strides=[1, stride, stride, 1], padding='SAME'
            )
        hidden_layer = tf.nn.bias_add(hidden_layer, _biases)
        hidden_layer = tf.transpose(hidden_layer, [0,3,1,2], name='NHWC_to_NCHW')
        hidden_layer = tf.nn.relu(hidden_layer)
        return hidden_layer

def Deconv2D(name, hidden_layer, filters, _biases, stride, height, width, out_dim):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    returns: tensor of shape (batch size, num channels, 2*height, 2*width)
    """
    with tf.name_scope(name) as scope:        
        hidden_layer = tf.transpose(hidden_layer, [0,2,3,1], name='NCHW_to_NHWC')
        input_shape = tf.shape(hidden_layer)
        hidden_layer = tf.nn.conv2d_transpose(value=hidden_layer, filter=filters,
            output_shape=tf.stack([input_shape[0], height, width, out_dim]), 
            strides=[1, stride, stride, 1], padding='SAME'
            )
        hidden_layer = tf.nn.bias_add(hidden_layer, _biases)
        hidden_layer = tf.transpose(hidden_layer, [0,3,1,2], name='NHWC_to_NCHW')
        hidden_layer = tf.nn.relu(hidden_layer)
        return hidden_layer