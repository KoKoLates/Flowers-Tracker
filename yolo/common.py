#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """ """
    def call(self, x, training:bool=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
    

def convolutional(input_layer, filters_shape, downsample:bool=False, 
                  activate:bool=True, bn:bool=True, activate_type='leaky'):
    """ 2D Convolutional Layers """
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding, strides = 'valid', 2
    else:
        padding, strides = 'same', 1

    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1], 
        kernel_size = filters_shape[0], 
        strides=strides, padding=padding, use_bias=not bn, 
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
    )(input_layer)
    
    if bn: conv = BatchNormalization()(conv)
    if activate and activate_type in ['leaky', 'mish']:
        conv = tf.nn.leaky_relu(conv, alpha=0.1) if activate_type == "leaky" else mish(conv)
    return conv

def mish(x):
    """ 
    Mish Activation Function.
    `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))` 
    """
    return x * tf.math.tanh(tf.math.softplus(x))

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    """ 
    Residual Block of ResNet with `2 weight layers` and `1 shortcut` from input layer
    """
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    return short_cut + conv

def route_group(input_layer, groups, group_id):
    """ Route Group """
    return tf.split(input_layer, num_or_size_splits=groups, axis=-1)[group_id]

def upsample(input_layer):
    """ Up Sample """
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
