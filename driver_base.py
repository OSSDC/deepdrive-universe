from __future__ import print_function
import time
from universe.spaces.joystick_event import JoystickAxisXEvent, JoystickAxisZEvent
import logging
logger = logging.getLogger()


class DriverBase(object):
    def __init__(self):
        self.net = None
        self.image_transformer = None
        self.env = None
        self.action_space = None
        self.target = None
        self.observation_n = None
        self.reward_n = None
        self.done_n = None
        self.throttle = 0.
        self.steer = 0.

    def load_net(self):
        raise NotImplementedError('Show us how to load your net')

    def set_input(self, image):
        raise NotImplementedError('Show us how to set your net\'s inputs')

    def get_next_action(self, net_out, info):
        raise NotImplementedError('Show us how to get the next action given the net\'s outputs')

    def get_net_out(self):
        raise NotImplementedError('Show us how to get output from your net')

    def setup(self):
        self.load_net()

    def step(self, observation_n, reward_n, done_n, info):
        if observation_n[0] is None:
            return self.get_noop()
        image = observation_n[0]['vision']
        if image is None:
            logger.warning('No image received. '
                           'You may need to restart the environment - Also make sure to leave the RDP session open.')
            return self.get_noop()
        else:
            begin = time.time()
            self.set_input(image)
            end = time.time()
            logger.debug('time to set net input %s', end - begin)

            begin = time.time()
            net_out = self.get_net_out()
            end = time.time()
            logger.debug('time to get net out %s', end - begin)

            begin = time.time()
            next_action_n = self.get_next_action(net_out, info)
            end = time.time()
            logger.debug('time to get next action %s', end - begin)

            errored = [i for i, info_i in enumerate(info['n']) if 'error' in info_i]
            if errored:
                logger.error('had errored indexes: %s: %s', errored, info)

            logger.debug('reward % s', reward_n)
            # if any(done_n) or any(r != 0.0 and r is not None for r in reward_n):
            #     logger.info('reward_n=%s done_n=%s info=%s', reward_n, done_n, info)

            return next_action_n


    def get_noop(self):
        x_axis_event = JoystickAxisXEvent(0)
        z_axis_event = JoystickAxisZEvent(0)
        noop = [[x_axis_event, z_axis_event]]
        return noop
