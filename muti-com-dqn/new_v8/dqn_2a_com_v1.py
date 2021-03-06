from actionQnet_ import *
import argparse
from itertools import chain
import time
from utils import *
import numpy as np
import mxnet as mx
from config import Config
from hunterworld_v2 import HunterWorld
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
    parser.add_argument('--t-max', type=int, default=50)
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=1)

    args = parser.parse_args()
    config = Config(args)
    np.random.seed(args.seed)

    if not os.path.exists('a1_A3c'):
        os.makedirs('a1_A3c')
    if not os.path.exists('a2_A3c'):
        os.makedirs('a2_A3c')
    if not os.path.exists('a1_comNet'):
        os.makedirs('a1_comNet')
    if not os.path.exists('a2_comNet'):
        os.makedirs('a2_comNet')

    testing = False
    if testing:
        args.num_envs = 1
        # else:
        # logging_config("log", "single-output-custom")

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
                           num_hunters=2, num_toxins=5)
        env = PLE(game, fps=30, force_fps=True, display_screen=False, reward_values=rewards,
                  resized_rows=80, resized_cols=80, num_steps=3)
        envs.append(env)

    a1_A3c_file = None
    a2_A3c_file = None
    a1_comNet_file = None
    a2_comNet_file = None
    if testing:
        envs[0].force_fps = False
        envs[0].game.draw = True
        envs[0].display_screen = True
        a1_A3c_file = 'a1_A3c/network-dqn_mx0012.params'
        a2_A3c_file = 'a2_A3c/network-dqn_mx0012.params'
        a1_comNet_file = 'a1_comNet/network-dqn_mx0012.params'
        a2_comNet_file = 'a2_comNet/network-dqn_mx0012.params'

    action_set = envs[0].get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)

    action_num = 4
    a1_A3c = A3c(state_dim=74, signal_num=4, act_space=action_num, config=config)
    a2_A3c = A3c(state_dim=74, signal_num=4, act_space=action_num, config=config)

    a1_comNet = ComNet(state_dim=74, signal_num=4, config=config)
    a2_comNet = ComNet(state_dim=74, signal_num=4, config=config)

    if a1_A3c_file is not None:
        a1_A3c.model.load_params(a1_A3c_file)
        a2_A3c.model.load_params(a2_A3c_file)
        a1_comNet.model.load_params(a1_comNet_file)
        a2_comNet.model.load_params(a2_comNet_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(a1_A3c.model)
    print_params(a1_comNet.model)

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
            env_ob1, env_ob2 = _2d_list(num_envs), _2d_list(num_envs)
            env_action1, env_action2 = _2d_list(num_envs), _2d_list(num_envs)
            env_value1, env_value2 = _2d_list(num_envs), _2d_list(num_envs)
            env_reward1, env_reward2 = _2d_list(num_envs), _2d_list(num_envs)

            for env in envs:
                env.reset_game()
            step_ob = np.asarray([env.get_states() for env in envs])

            done = [False] * num_envs
            all_done = False
            t = 1
            training_steps = 0
            episode_advs = 0

            while not all_done:

                a1_ob = step_ob[:, :74]
                a2_ob = step_ob[:, -74:]

                signal1 = a1_comNet.forward(a2_ob, is_train=False)[0]
                signal2 = a2_comNet.forward(a1_ob, is_train=False)[0]

                _, step_value1, _, step_policy1 = a1_A3c.forward(a1_ob, signal1, is_train=False)
                _, step_value2, _, step_policy2 = a2_A3c.forward(a2_ob, signal2, is_train=False)

                step_policy1 = step_policy1.asnumpy()
                step_policy2 = step_policy2.asnumpy()

                step_value1 = step_value1.asnumpy()
                step_value2 = step_value2.asnumpy()

                us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
                step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                for i, env in enumerate(envs):
                    if not done[i]:
                        next_ob, reward, done[i] = env.act_action(
                            [action_map1[step_action1[i]], action_map2[step_action2[i]]])

                        env_ob1[i].append(step_ob[i][:74])
                        env_ob2[i].append(step_ob[i][-74:])

                        env_action1[i].append(step_action1[i])
                        env_action2[i].append(step_action2[i])

                        env_value1[i].append(step_value1[i][0])
                        env_value2[i].append(step_value2[i][0])

                        env_reward1[i].append(reward[0])
                        env_reward2[i].append(reward[1])

                        episode_reward[i] += reward[0]
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1
                        step_ob[i] = next_ob

                if t == t_max:

                    a1_ob = step_ob[:, :74]
                    a2_ob = step_ob[:, -74:]

                    signal1 = a1_comNet.forward(a2_ob, is_train=False)[0]
                    signal2 = a2_comNet.forward(a1_ob, is_train=False)[0]

                    extra_value1 = a1_A3c.forward(a1_ob, signal1, is_train=False)[1]
                    extra_value2 = a2_A3c.forward(a2_ob, signal2, is_train=False)[1]

                    extra_value1 = extra_value1.asnumpy()
                    extra_value2 = extra_value2.asnumpy()

                    for i in range(num_envs):
                        if not done[i]:
                            env_value1[i].append(extra_value1[i][0])
                            env_value1[i].append(extra_value2[i][0])
                        else:
                            env_value1[i].append(0.0)
                            env_value2[i].append(0.0)

                    advs_p1 = []
                    advs_p2 = []
                    for i in xrange(len(env_value1)):
                        R1 = env_value1[i][-1]
                        R2 = env_value2[i][-1]
                        tmp_R1 = []
                        tmp_R2 = []
                        for j in range(len(env_reward1[i]))[::-1]:
                            R1 = env_reward1[i][j] + gamma * R1
                            R2 = env_reward2[i][j] + gamma * R2
                            tmp_R1.append(R1 - env_value1[i][j])
                            tmp_R2.append(R2 - env_value2[i][j])
                        advs_p1.extend(tmp_R1[::-1])
                        advs_p2.extend(tmp_R2[::-1])

                    a1_advs = -np.asarray(advs_p1)
                    a2_advs = -np.asarray(advs_p2)
                    env_action1 = np.array(env_action1)
                    env_action2 = np.array(env_action2)

                    a1_advs_grads = compute_adv(neg_advs_v=a1_advs, action_num=action_num,
                                                action=env_action1,
                                                q_ctx=q_ctx)
                    a2_advs_grads = compute_adv(neg_advs_v=a2_advs, action_num=action_num,
                                                action=env_action2,
                                                q_ctx=q_ctx)

                    a1_value_grads = mx.nd.array(0.5 * a1_advs[:, np.newaxis],
                                                 ctx=q_ctx)
                    a2_value_grads = mx.nd.array(0.5 * a2_advs[:, np.newaxis],
                                                 ctx=q_ctx)

                    a1_env_ob = np.asarray(list(chain.from_iterable(env_ob1)))
                    a2_env_ob = np.asarray(list(chain.from_iterable(env_ob2)))

                    signal1 = a1_comNet.forward(a2_env_ob, is_train=True)[0]
                    signal2 = a2_comNet.forward(a1_env_ob, is_train=True)[0]

                    a1_A3c.forward(a1_env_ob, signal1, is_train=True)
                    a2_A3c.forward(a2_env_ob, signal2, is_train=True)

                    a1_A3c.model.backward(out_grads=[a1_advs_grads, a1_value_grads])
                    a2_A3c.model.backward(out_grads=[a2_advs_grads, a2_value_grads])
                    a1_A3c.model.update()
                    a2_A3c.model.update()

                    a1_comNet.model.backward(out_grads=[a2_A3c.model._exec_group.execs[0].grad_dict['signal'][:]])
                    a2_comNet.model.backward(out_grads=[a1_A3c.model._exec_group.execs[0].grad_dict['signal'][:]])
                    a1_comNet.model.update()
                    a2_comNet.model.update()

                    env_ob1, env_ob2 = _2d_list(num_envs), _2d_list(num_envs)
                    env_action1, env_action2 = _2d_list(num_envs), _2d_list(num_envs)
                    env_value1, env_value2 = _2d_list(num_envs), _2d_list(num_envs)
                    env_reward1, env_reward2 = _2d_list(num_envs), _2d_list(num_envs)

                    training_steps += 1
                    episode_advs += np.mean(a1_advs)

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
        a1_A3c.model.save_params('a1_A3c/network-dqn_mx%04d.params' % epoch)
        a2_A3c.model.save_params('a2_A3c/network-dqn_mx%04d.params' % epoch)
        a1_comNet.model.save_params('a1_comNet/network-dqn_mx%04d.params' % epoch)
        a2_comNet.model.save_params('a2_comNet/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))
