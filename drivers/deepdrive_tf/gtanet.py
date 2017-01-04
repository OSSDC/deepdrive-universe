import tensorflow as tf
from drivers.deepdrive_tf.layers import conv2d, max_pool_2x2, linear, lrn


class GTANetModel(object):
    def __init__(self, x, num_targets=6, is_training=True):
        self.x = x
        conv1 = tf.nn.relu(conv2d(x, "conv1", 96, 11, 4, 1))
        lrn1 = lrn(conv1)
        maxpool1 = max_pool_2x2(lrn1)
        conv2 = tf.nn.relu(conv2d(maxpool1, "conv2", 256, 5, 1, 2))
        lrn2 = lrn(conv2)
        maxpool2 = max_pool_2x2(lrn2)
        conv3 = tf.nn.relu(conv2d(maxpool2, "conv3", 384, 3, 1, 1))
        conv4 = tf.nn.relu(conv2d(conv3, "conv4", 384, 3, 1, 2))
        conv5 = tf.nn.relu(conv2d(conv4, "conv5", 256, 3, 1, 2))
        maxpool5 = max_pool_2x2(conv5)
        fc6 = tf.nn.relu(linear(maxpool5, "fc6", 4096))
        # if is_training:
        #     fc6 = tf.nn.dropout(fc6, 0.5)
        fc7 = tf.nn.relu(linear(fc6, "fc7", 4096))
        # if is_training:
        #     fc7 = tf.nn.dropout(fc7, 0.95)
        fc8 = linear(fc7, "fc8", num_targets)

        self.fc8 = fc8
        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                           trainable=False)
