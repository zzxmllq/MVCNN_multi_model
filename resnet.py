import tensorflow as tf
import numpy as np


def conv(input, name, kh, kw, n_out, sh=1, sw=1, padding='SAME', reuse=False):
    n_in = input.get_shape()[-1].value
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(input, kernel, [1, sh, sw, 1], padding=padding, name=scope.name)
        # biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        # bias = tf.nn.bias_add(conv, biases)
    return conv


def blocks3_3(input, name, planes, reuse, stride=1, downsample=False):
    x = input
    if downsample:
        x = conv(input=input, name=name + 'downsample', kh=1, kw=1, sh=stride, sw=stride, n_out=planes, reuse=reuse)
        x = tf.layers.batch_normalization(x)
    residual = x
    conv1 = conv(input=input, name=name + 'conv_1', kh=3, kw=3, sh=stride, sw=stride, n_out=planes, reuse=reuse)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = conv(input=conv1, name=name + 'conv_2', kh=3, kw=3, n_out=planes, reuse=reuse)
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 += residual
    out = tf.nn.relu(conv2)
    return out


def blocks1_3_1(input, name, planes, reuse, stride=1, downsample=False):
    x = input
    if downsample:
        x = conv(input=input, name=name + 'downsample', kh=1, kw=1, sh=stride, sw=stride, n_out=planes * 4, reuse=reuse)
        x = tf.layers.batch_normalization(x)
    residual = x
    conv1 = conv(input=input, name=name + 'conv_1', kh=1, kw=1, n_out=planes, reuse=reuse)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = conv(input=conv1, name=name + 'conv_2', kh=3, kw=3, n_out=planes, sh=stride, sw=stride, reuse=reuse)
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)
    conv3 = conv(input=conv2, name=name + 'conv_3', kh=1, kw=1, n_out=planes * 4, reuse=reuse)
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 += residual
    out = tf.nn.relu(conv3)
    return out


def make_layer(input, name, block3_3, planes, blocks, reuse, stride=1):
    downsample = False
    if block3_3:
        if stride != 1:
            downsample = True
        input = blocks3_3(input=input, name=name + '0', planes=planes, stride=stride, downsample=downsample,
                          reuse=reuse)
        for i in range(1, blocks):
            input = blocks3_3(input=input, name=name + str(i), planes=planes, stride=1, downsample=False, reuse=reuse)
        return input
    else:
        if stride != 1:
            downsample = True
        input = blocks1_3_1(input=input, name=name + '0', planes=planes, stride=stride, downsample=downsample,
                            reuse=reuse)
        for i in range(1, blocks):
            input = blocks1_3_1(input=input, name=name + str(i), planes=planes, stride=1, downsample=False, reuse=reuse)
        return input


def ResNet(input, block3_3, layers, reuse):
    conv1 = conv(input=input, name='conv1', kh=7, kw=7, sh=2, sw=2, n_out=64, reuse=reuse)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2_pool = tf.nn.max_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = make_layer(conv2_pool, 'conv2', block3_3, 64, layers[0], reuse=reuse)
    conv3 = make_layer(conv2, 'conv3', block3_3, 128, layers[1], stride=2, reuse=reuse)
    conv4 = make_layer(conv3, 'conv4', block3_3, 256, layers[2], stride=2, reuse=reuse)
    conv5 = make_layer(conv4, 'conv5', block3_3, 512, layers[3], stride=2, reuse=reuse)
    avg_pool = tf.nn.avg_pool(value=conv5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    dim = np.prod(avg_pool.get_shape().as_list()[1:])
    reshape = tf.reshape(avg_pool, [-1, dim])
    return reshape


def resnet18(input, reuse):
    model = ResNet(input=input, block3_3=True, layers=[2, 2, 2, 2], reuse=reuse)
    return model


def resnet34(input, reuse):
    model = ResNet(input=input, block3_3=True, layers=[3, 4, 6, 3], reuse=reuse)
    return model


def resnet50(input, reuse):
    model = ResNet(input=input, block3_3=False, layers=[3, 4, 6, 3], reuse=reuse)
    return model


def resnet101(input, reuse):
    model = ResNet(input=input, block3_3=False, layers=[3, 4, 23, 3], reuse=reuse)
    return model

def resnet152(input, reuse):
    model = ResNet(input=input, block3_3=False, layers=[3, 8, 36, 3], reuse=reuse)
    return model


def _view_pool(input, name):
    vp = tf.expand_dims(input[0], 0)
    for v in input[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    vp = tf.reduce_max(vp, [0], name=name)
    return vp


def fc(input, name, n_out, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        n_in = input.get_shape().as_list()[-1]
        w_fc = tf.get_variable(name=name + '_w', shape=[n_in, n_out], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        # weight_decay=tf.multiply(tf.nn.l2_loss(w_fc),0.004,name='weight_loss')
        # tf.add_to_collection('losses',weight_decay)
        b_fc = tf.get_variable(name=name + '_b', shape=[n_out], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        # fc = tf.nn.relu(tf.matmul(input, w_fc) + b_fc, name=scope.name)
        # fc = tf.nn.dropout(fc, dropout)
        fc = tf.matmul(input, w_fc) + b_fc
    return fc


def model(input_data, n_classes):
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
        output = resnet18(current_image, reuse)
        view_pool.append(output)
    avgpool_vp = _view_pool(input=view_pool, name='avgpool_vp')
    fc8 = fc(input=avgpool_vp, name='fc8', n_out=n_classes)
    return fc8


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y
