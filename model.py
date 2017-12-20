import numpy as np
import tensorflow as tf

length = 44100
predict = int(length / (4 * 9))
batch = 1


def makemodel():
    with tf.name_scope("audio_network") as scope:
        input_layer = tf.placeholder(dtype=np.float32,
                                     shape=[batch, length, 1],
                                     name="input")

        conv1 = tf.layers.conv1d(inputs=input_layer,
                                 filters=32,
                                 kernel_size=5,
                                 padding="same",
                                 name="conv1",
                                 trainable=True)
        pool1 = tf.layers.max_pooling1d(inputs=conv1,
                                        pool_size=4,
                                        strides=4,
                                        name='pool1')

        conv2 = tf.layers.conv1d(inputs=pool1,
                                 filters=32,
                                 padding="same",
                                 kernel_size=10,
                                 name="conv2",
                                 trainable=True)
        pool2 = tf.layers.max_pooling1d(inputs=conv2,
                                        pool_size=9,
                                        strides=9,
                                        name="pool2")

        conv3 = tf.layers.conv1d(inputs=pool2,
                                 filters=32,
                                 padding="same",
                                 kernel_size=100,
                                 name="conv3",
                                 trainable=True)
        pool3 = tf.layers.max_pooling1d(inputs=conv3,
                                        pool_size=49,
                                        strides=49,
                                        name="pool3")

        convolutionedFlat = tf.reshape(pool3,
                                       [np.int64(batch), np.int64(length / 9 / 4 / 49 * 32)],
                                       name="flatConv")

        result = tf.layers.dense(convolutionedFlat,
                                 units=np.int64(predict),
                                 activation=tf.nn.relu,
                                 name="deconv1",
                                 trainable=True)

        prediction = tf.placeholder(dtype=np.float32,
                                shape=[batch, predict],
                                name="prediction")

        abs_diff = tf.reduce_sum(tf.square(prediction - result))
        loss = tf.reduce_sum(abs_diff)

    return input_layer, result, prediction, loss
