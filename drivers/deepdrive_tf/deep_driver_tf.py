from __future__ import print_function
import time

from driver_base import DriverBase
from universe.spaces.joystick_event import JoystickAxisXEvent, JoystickAxisZEvent
import logging
import numpy as np
from scipy.misc import imresize

logger = logging.getLogger()

import tensorflow as tf
import os
from drivers.deepdrive_tf.gtanet import GTANetModel

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class DeepDriverTF(DriverBase):
    def __init__(self):
        super(DeepDriverTF, self).__init__()
        self.sess = None
        self.net = None
        self.image_var = None
        self.net_out_var = None
        self.image_shape = (227, 227, 3)
        self.image = None
        self.num_targets = 6
        self.mean_pixel = np.array([104., 117., 123.], np.float32)
        self.min_outputs = np.array([10., 10., 10., 10., 10., 10.])
        self.max_outputs = np.array([-10., -10., -10., -10., -10., -10.])

    def load_net(self):
        saver = tf.train.import_meta_graph(os.path.join(DIR_PATH, 'model.ckpt-20048.meta'))
        self.sess = tf.Session()

        # self.image_var = tf.placeholder(tf.float32, (None,) + self.image_shape)
        # self.net_out_var = tf.placeholder(tf.float32, (None, self.num_targets))
        # self.sess.run(tf.initialize_all_variables())

        saver.restore(self.sess, os.path.join(DIR_PATH, 'model.ckpt-20048'))
        pass

        # self.net = GTANetModel(self.image_var, is_training=False)

    def get_next_action(self, net_out, info):
        spin, direction, speed, speed_change, steer, throttle = net_out[0]
        steer = -float(steer)
        steer -= 0.20
        print(steer)
        # throttle = -float(throttle)
        # speed += 1.0
        steer_dead_zone = 0.2
        self.max_outputs = np.max(np.array([self.max_outputs, net_out[0]]).T, axis=1)
        self.min_outputs = np.min(np.array([self.min_outputs, net_out[0]]).T, axis=1)

        print('max outputs', self.max_outputs)
        print('min outputs', self.min_outputs)

        # Add dead zones
        if steer > 0:
            steer += steer_dead_zone
        elif steer < 0:
            steer -= steer_dead_zone

        logger.debug('steer %f', steer)
        x_axis_event = JoystickAxisXEvent(steer)
        if 'n' in info and 'speed' in info['n'][0]:
            current_speed = info['n'][0]['speed']
            desired_speed = speed * 20.  # Denormalize per deep_drive.h in deepdrive-caffe
            if desired_speed < current_speed:
                logger.debug('braking')
                throttle = self.throttle - (current_speed - desired_speed) * 0.085  # Magic number
                throttle = max(throttle, 0.0)
            else:
                throttle += 13. / 50.  # Joystick dead zone

            z_axis_event = JoystickAxisZEvent(float(throttle))
            logging.debug('throttle %s', throttle)
        else:
            z_axis_event = JoystickAxisZEvent(0)
            logging.debug('No info received from environment for this frame - sending noop')
        next_action_n = [[x_axis_event, z_axis_event]]

        self.throttle = throttle
        self.steer = steer

        # return self.get_net_out()
        return self.get_noop()

    def set_input(self, img):
        img = imresize(img, self.image_shape)
        img = img.astype(np.float32)
        img -= self.mean_pixel
        self.image = img

    def get_net_out(self):
        begin = time.time()
        net_out = self.sess.run('model/add_5:0', feed_dict={'Placeholder:0': self.image.reshape(1, 227, 227, 3)})
        print(net_out)
        end = time.time()
        logger.debug('inference time %s', end - begin)
        return net_out
