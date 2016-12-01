#!/usr/bin/env python
import argparse
import logging
import time
import gym
import universe
from universe import pyprofile, wrappers

from drivers.deepdrive.deep_driver import DeepDriver


# if not os.getenv("PYPROFILE_FREQUENCY"):
#     pyprofile.profile.print_frequency = 5

logger = logging.getLogger()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger.setLevel(logging.INFO)
    universe.configure_logging()

    # Actions this agent will take, 'random' is the default
    action_choices = ['random', 'noop', 'forward']

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', default='gym-core.Pong-v3', help='Which environment to run on.')
    parser.add_argument('-m', '--monitor', action='store_true', help='Whether to activate the monitor.')
    parser.add_argument('-r', '--remote', default='http://allocator.sci.openai-tech.com', help='The number of environments to create (e.g. -r 20), or the address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,localhost:5901+15901), or a query to the allocator (e.g. -r http://allocator.sci.openai-tech.com?n=2)')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-R', '--no-render', action='store_true', help='Do not render the environment locally.')
    parser.add_argument('-f', '--fps', default=60., type=float, help='Desired frames per second')
    parser.add_argument('-N', '--max-steps', type=int, default=10**7, help='Maximum number of steps to take')
    parser.add_argument('-d', '--driver', default='DeepDriver', help='Choose your driver')

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

    driver = None
    if args.driver == 'DeepDriver':
        driver = DeepDriver()
        driver.setup()
    else:
        raise Exception('That driver is not available')

    driver = DeepDriver()
    driver.setup()

    if args.monitor:
        env.monitor.start('/tmp/vnc_random_agent', force=True, video_callable=lambda i: True)

    render = not args.no_render
    observation_n = env.reset()
    target = time.time()
    reward_n = [0] * env.n
    done_n = [False] * env.n
    action_n = driver.get_noop()
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

        # Take an action
        with pyprofile.push('env.step'):
            observation_n, reward_n, done_n, info = env.step(action_n)

    # We're done! clean up
    env.close()
