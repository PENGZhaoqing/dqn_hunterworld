from actionQnet_ import *
import argparse
from itertools import chain
import time
from utils import *
import numpy as np
import mxnet as mx
from config import Config
from hunterworld import HunterWorld
from ple import PLE
import matplotlib.pyplot as plt

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)


def _2d_list(n):
    return [[] for _ in range(n)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=1)

    args = parser.parse_args()
    config = Config(args)
    np.random.seed(args.seed)

    if not os.path.exists('A3c'):
        os.makedirs('A3c')

    testing = False
    if testing:
        args.num_envs = 1
    else:
        logging_config("log", "single-output-custom")

    rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": -0.002,
        "loss": -2.0,
        "win": 2.0
    }

    envs = []
    for i in range(args.num_envs):
        game = HunterWorld(width=256, height=256, num_preys=10, draw=False,
                           num_hunters=1, num_toxins=5)
        env = PLE(game, fps=30, force_fps=True, display_screen=False, reward_values=rewards,
                  resized_rows=80, resized_cols=80, num_steps=3)
        envs.append(env)

    params_file = None
    if testing:
        envs[0].force_fps = False
        envs[0].game.draw = True
        envs[0].display_screen = True
        replay_start_size = 32
        params_file = 'A3c/network-dqn_mx0012.params'

    a3c = Agent(74, 4, config=config)

    if params_file is not None:
        a3c.model.load_params(params_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(a3c.model)

    action_set = envs[0].get_action_set()
    action_map = []
    for action in action_set[0].values():
        action_map.append(action)
    action_num = len(action_map)

    running_reward = None
    t_max = args.t_max
    num_envs = len(envs)
    q_ctx = config.ctx
    gamma, lambda_ = config.gamma, config.lambda_
    epoch_num = 20
    steps_per_epoch = 100000

    for epoch in range(epoch_num):
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()

        while steps_left > 0:
            episode += 1
            time_episode_start = time.time()
            collisions = [0.0] * len(envs)
            episode_reward = np.zeros(num_envs, dtype=np.float)
            episode_step = 0
            env_ob, env_action = _2d_list(num_envs), _2d_list(num_envs)
            env_reward, env_value = _2d_list(num_envs), _2d_list(num_envs)
            for env in envs:
                env.reset_game()
            step_ob = [env.get_states() for env in envs]

            done = [False] * num_envs
            all_done = False
            t = 1
            training_steps = 0
            episode_advs = 0

            while not all_done:
                data_batch = mx.io.DataBatch(data=[mx.nd.array(step_ob, ctx=q_ctx)], label=None)
                a3c.model.reshape([('data', (num_envs, 74))])
                a3c.model.forward(data_batch, is_train=False)
                _, step_value, _, step_policy = a3c.model.get_outputs()

                step_policy = step_policy.asnumpy()
                step_value = step_value.asnumpy()

                us = np.random.uniform(size=step_policy.shape[0])[:, np.newaxis]
                step_action = (np.cumsum(step_policy, axis=1) > us).argmax(axis=1)

                for i, env in enumerate(envs):
                    if not done[i]:
                        next_ob, reward, done[i] = env.act_action([action_map[step_action[i]]])
                        env_ob[i].append(step_ob[i])
                        env_action[i].append(step_action[i])
                        env_value[i].append(step_value[i][0])
                        env_reward[i].append(reward[0])
                        step_ob[i] = next_ob
                        episode_reward[i] += reward[0]
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1

                if t == t_max:
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(step_ob, ctx=q_ctx)], label=None)
                    a3c.model.forward(data_batch, is_train=False)
                    _, extra_value, _, _ = a3c.model.get_outputs()

                    extra_value = extra_value.asnumpy()
                    for i in range(num_envs):
                        if done[i]:
                            env_value[i].append(0.0)
                        else:
                            env_value[i].append(extra_value[i][0])

                    advs = []
                    for i in range(len(env_value)):
                        R = env_value[i][-1]
                        tmp = []
                        for j in range(len(env_reward[i]))[::-1]:
                            R = env_reward[i][j] + gamma * R
                            tmp.append(R - env_value[i][j])
                        advs.extend(tmp[::-1])
                    neg_advs_v = -np.asarray(advs)

                    neg_advs_np = np.zeros((len(advs), action_num), dtype=np.float32)
                    action = np.array(list(chain.from_iterable(env_action)))
                    neg_advs_np[np.arange(neg_advs_np.shape[0]), action] = neg_advs_v
                    neg_advs = mx.nd.array(neg_advs_np, ctx=q_ctx)

                    value_grads = mx.nd.array(config.vf_wt * neg_advs_v[:, np.newaxis],
                                              ctx=q_ctx)
                    env_ob = list(chain.from_iterable(env_ob))
                    a3c.model.reshape([('data', (len(env_ob), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob, ctx=q_ctx)], label=None)
                    a3c.model.forward(data_batch, is_train=True)
                    a3c.model.backward(out_grads=[neg_advs, value_grads])
                    a3c.model.update()

                    env_ob, env_action = _2d_list(num_envs), _2d_list(num_envs)
                    env_reward, env_value = _2d_list(num_envs), _2d_list(num_envs)

                    training_steps += 1
                    episode_advs += np.mean(neg_advs_v)

                    t = 0
                all_done = np.all(done)
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_reward
            time_episode_end = time.time()
            if episode % args.print_every == 0:
                for reward in episode_reward:
                    running_reward = reward if running_reward is None else (
                        0.99 * running_reward + 0.01 * reward)

                info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, Running Reward:%f, Collisions:%f, fps:%f, Avg Advs:%f" \
                           % (
                               epoch, episode, steps_left, episode_step, steps_per_epoch, np.mean(episode_reward),
                               np.mean(running_reward), np.mean(collisions),
                               episode_step / (time_episode_end - time_episode_start),
                               episode_advs / training_steps)
                logging.info(info_str)

        end = time.time()
        fps = steps_per_epoch / (end - time_episode_start)
        a3c.model.save_params('A3c/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))
