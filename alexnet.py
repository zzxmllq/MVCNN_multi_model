import tensorflow as tf
import numpy as np


def conv(input, name, kh, kw, sh, sw, n_out, padding='VALID', reuse=False):
    n_in = input.get_shape()[-1].value
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(input, kernel, [1, sh, sw, 1], padding=padding)
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)
    return relu


def maxpool(input, name, kh, kw, sh, sw, padding='VALID'):
    pool = tf.nn.max_pool(value=input, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding=padding, name=name)
    return pool


def _view_pool(input, name):
    vp = tf.expand_dims(input[0], 0)
    for v in input[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    vp = tf.reduce_max(vp, [0], name=name)
    return vp


def fc(input, name, n_out, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        n_in = input.get_shape().as_list()[-1]
        w_fc = tf.get_variable(name=name + '_w', shape=[n_in, n_out], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        weight_decay=tf.multiply(tf.nn.l2_loss(w_fc),0.004,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
        b_fc = tf.get_variable(name=name + '_b', shape=[n_out], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(input, w_fc) + b_fc, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)
    return fc


def fc_norelu(input, name, n_out, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        n_in = input.get_shape().as_list()[-1]
        w_fc = tf.get_variable(name=name + '_w', shape=[n_in, n_out], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        weight_decay=tf.multiply(tf.nn.l2_loss(w_fc),0.004,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
        b_fc = tf.get_variable(name=name + '_b', shape=[n_out], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w_fc) + b_fc
        fc = tf.nn.dropout(fc, dropout)
    return fc


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y


def model(input_data, n_classes, keep_prob):
    """
    views: N x V x W x H x C tensor
    """

    n_views = input_data.get_shape().as_list()[1]

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    input_data = tf.transpose(input_data, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in range(n_views):
        reuse = (i != 0)
        current_image = input_data[i]
        conv1 = conv(input=current_image, name='conv1', kh=11, kw=11, sh=4, sw=4, n_out=96, reuse=reuse)
        pool1 = maxpool(input=conv1, name='pool1', kh=3, kw=3, sh=2, sw=2)
        conv2 = conv(input=pool1, name='conv2', kh=5, kw=5, sh=1, sw=1, n_out=256, reuse=reuse)
        pool2 = maxpool(input=conv2, name='pool2', kh=3, kw=3, sh=2, sw=2)
        conv3 = conv(input=pool2, name='conv3', kh=3, kw=3, sh=1, sw=1, n_out=384, reuse=reuse)
        conv4 = conv(input=conv3, name='conv4', kh=3, kw=3, sh=1, sw=1, n_out=384, reuse=reuse)
        conv5 = conv(input=conv4, name='conv5', kh=3, kw=3, sh=1, sw=1, n_out=256, reuse=reuse)
        pool5 = maxpool(input=conv5, name='pool5', kh=3, kw=3, sh=2, sw=2)
        dim = np.prod(pool5.get_shape().as_list()[1:])
        reshape = tf.reshape(pool5, [-1, dim])
        view_pool.append(reshape)
    pool5_vp = _view_pool(input=view_pool, name='pool5_vp')
    fc6 = fc(input=pool5_vp, name='fc6', n_out=4096, dropout=keep_prob)
    fc7 = fc(input=fc6, name='fc7', n_out=4096, dropout=keep_prob)
    fc8 = fc_norelu(input=fc7, name='fc8', n_out=n_classes)
    return fc8
