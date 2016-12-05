#!/usr/bin/env python
import argparse
import logging
import gym
import sys
import universe
from universe import pyprofile, wrappers

from GameSettingsEvent import GTASetting
from drivers.deepdrive.deep_driver import DeepDriverBase


# if not os.getenv("PYPROFILE_FREQUENCY"):
#     pyprofile.profile.print_frequency = 5

logger = logging.getLogger()
extra_logger = logging.getLogger('universe')

stdout_log_handler = logging.StreamHandler(sys.stdout)
stdout_log_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_log_handler.setFormatter(formatter)
extra_logger.addHandler(stdout_log_handler)


def main():
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger.setLevel(logging.INFO)
    universe.configure_logging()

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', default='gym-core.Pong-v3', help='Which environment to run on.')
    parser.add_argument('-m', '--monitor', action='store_false', help='Whether to activate the monitor.')
    parser.add_argument('-r', '--remote', default='http://allocator.sci.openai-tech.com', help='The number of environments to create (e.g. -r 20), or the address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,localhost:5901+15901), or a query to the allocator (e.g. -r http://allocator.sci.openai-tech.com?n=2)')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-R', '--no-render', action='store_true', help='Do not render the environment locally.')
    parser.add_argument('-f', '--fps', default=60., type=float, help='Desired frames per second')
    parser.add_argument('-N', '--max-steps', type=int, default=10**7, help='Maximum number of steps to take')
    parser.add_argument('-d', '--driver', default='DeepDriver', help='Choose your driver')
    parser.add_argument('-c', '--custom_camera',  action='store_false', help='Customize the GTA camera')

    args = parser.parse_args()

    logging.getLogger('gym').setLevel(logging.NOTSET)
    logging.getLogger('universe').setLevel(logging.NOTSET)
    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if args.env_id is not None:
        env = gym.make(args.env_id)
    else:
        env = wrappers.WrappedVNCEnv()
    if not isinstance(env, wrappers.GymCoreAction):
        # The GymCoreSyncEnv's try to mimic their core counterparts,
        # and thus came pre-wrapped wth an action space
        # translator. Everything else probably wants a SafeActionSpace
        # wrapper to shield them from random-agent clicking around
        # everywhere.
        env = wrappers.SafeActionSpace(env)
    else:
        # Only gym-core are seedable
        env.seed([0])
    env = wrappers.Logger(env)

    env.configure(
        fps=args.fps,
        # print_frequency=None,
        # ignore_clock_skew=True,
        remotes=args.remote,
        vnc_driver='go', vnc_kwargs={
            'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50, 'subsample_level': 0, 'quality_level': 5,
        },
    )

    if args.driver == 'DeepDriver':
        driver = DeepDriverBase()
    elif args.driver == 'pt':
        from drivers.pt.pt_driver import PTDriverBase
        driver = PTDriverBase()
    else:
        raise Exception('That driver is not available')

    # driver = DeepDriver()
    driver.setup()

    if args.monitor:
        env.monitor.start('/tmp/vnc_random_agent', force=True, video_callable=lambda i: True)

    render = not args.no_render
    observation_n = env.reset()
    reward_n = [0] * env.n
    done_n = [False] * env.n
    info = None

    for i in range(args.max_steps):
        # print(observation_n)
        # user_input.handle_events()

        if render:
            # Note the first time you call render, it'll be relatively
            # slow and you'll have some aggregated rewards. We could
            # open the render() window before `reset()`, but that's
            # confusing since it pops up a black window for the
            # duration of the reset.
            env.render()

        action_n = driver.process_step(observation_n, reward_n, done_n, info)

        if args.custom_camera:
            # Sending this every step is probably overkill
            for action in action_n:
                action.append(GTASetting('use_custom_camera', True))

        # Take an action
        with pyprofile.push('env.step'):
            observation_n, reward_n, done_n, info = env.step(action_n)

    # We're done! clean up
    env.close()

if __name__ == '__main__':
    sys.exit(main())

