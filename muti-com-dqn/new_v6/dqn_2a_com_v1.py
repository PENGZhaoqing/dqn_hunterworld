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

    if not os.path.exists('Qnet'):
        os.makedirs('Qnet')
    if not os.path.exists('policy1'):
        os.makedirs('policy1')
    if not os.path.exists('policy2'):
        os.makedirs('policy2')

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

    Qnet_file = None
    a1_pnet = None
    a2_pnet = None
    if testing:
        envs[0].force_fps = False
        envs[0].game.draw = True
        envs[0].display_screen = True
        replay_start_size = 32
        Qnet_file = 'Qnet/network-dqn_mx0012.params'
        a1_pnet = 'policy1/network-dqn_mx0012.params'
        a2_pnet = 'policy2/network-dqn_mx0012.params'

    action_set = envs[0].get_action_set()
    action_map = []
    action_set = envs[0].get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)

    action_map3 = []
    for action in action_set[2].values():
        action_map3.append(action)

    action_num = 4
    Qnet = Qnet(156, config=config)
    a1_policy = PolicyNet(74, action_num, config=config)
    a2_policy = PolicyNet(74, action_num, config=config)

    if Qnet_file is not None:
        Qnet.model.load_params(Qnet_file)
        a1_policy.model.load_params(Qnet_file)
        a2_policy.model.load_params(Qnet_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(Qnet.model)
    print_params(a1_policy.model)

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
            step_ob = np.asarray([env.get_states() for env in envs])

            done = [False] * num_envs
            all_done = False
            t = 1
            training_steps = 0
            episode_advs = 0

            while not all_done:

                data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob[:, :74], ctx=q_ctx)], label=None)
                data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob[:, -74:], ctx=q_ctx)], label=None)

                a1_policy.model.reshape([('data', (num_envs, 74))])
                a2_policy.model.reshape([('data', (num_envs, 74))])
                Qnet.model.reshape([('data', (num_envs, 156))])

                a1_policy.model.forward(data_batch1, is_train=False)
                a2_policy.model.forward(data_batch2, is_train=False)

                step_policy1 = a1_policy.model.get_outputs()[2].asnumpy()
                step_policy2 = a2_policy.model.get_outputs()[2].asnumpy()

                data_batch = np.append(step_ob, step_policy1, axis=1)
                data_batch = np.append(data_batch, step_policy2, axis=1)

                Qnet.model.forward(mx.io.DataBatch(data=[mx.nd.array(data_batch, ctx=q_ctx)], label=None),
                                   is_train=False)

                step_value = Qnet.model.get_outputs()[0].asnumpy()

                us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
                step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                for i, env in enumerate(envs):
                    if not done[i]:
                        next_ob, reward, done[i] = env.act_action(
                            [action_map1[step_action1[i]], action_map2[step_action2[i]]])
                        env_ob[i].append(step_ob[i])
                        env_action[i].append([step_action1[i], step_action2[i]])
                        env_value[i].append(step_value[i][0])
                        env_reward[i].append(reward)
                        episode_reward[i] += reward[0]
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1
                        if done[i]:
                            env_value[i].append(0.0)
                        else:
                            step_ob[i] = next_ob

                if t == t_max:

                    data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob[:, :74], ctx=q_ctx)], label=None)
                    data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob[:, -74:], ctx=q_ctx)], label=None)

                    a1_policy.model.forward(data_batch1, is_train=False)
                    a2_policy.model.forward(data_batch2, is_train=False)

                    step_policy1 = a1_policy.model.get_outputs()[2].asnumpy()
                    step_policy2 = a2_policy.model.get_outputs()[2].asnumpy()

                    data_batch = np.append(step_ob, step_policy1, axis=1)
                    data_batch = np.append(data_batch, step_policy2, axis=1)

                    Qnet.model.forward(mx.io.DataBatch(data=[mx.nd.array(data_batch, ctx=q_ctx)], label=None),
                                       is_train=False)

                    extra_value = Qnet.model.get_outputs()[0].asnumpy()

                    for i in range(num_envs):
                        if not done[i]:
                            env_value[i].append(extra_value[i][0])

                    advs_Q = []
                    advs_p1 = []
                    advs_p2 = []
                    for i in xrange(len(env_value)):
                        R = R1 = R2 = env_value[i][-1]
                        tmp_R = []
                        tmp_R1 = []
                        tmp_R2 = []
                        for j in range(len(env_reward[i]))[::-1]:
                            R = sum(env_reward[i][j]) + gamma * R
                            R1 = env_reward[i][j][0] + gamma * R1
                            R2 = env_reward[i][j][1] + gamma * R2
                            tmp_R.append(R - env_value[i][j])
                            tmp_R1.append(R1 - env_value[i][j])
                            tmp_R2.append(R2 - env_value[i][j])
                        advs_Q.extend(tmp_R[::-1])
                        advs_p1.extend(tmp_R1[::-1])
                        advs_p2.extend(tmp_R2[::-1])

                    neg_advs_v = -np.asarray(advs_Q)
                    neg_advs_v1 = -np.asarray(advs_p1)
                    neg_advs_v2 = -np.asarray(advs_p2)

                    env_action = np.asarray(env_action)

                    neg_advs1 = compute_adv(neg_advs_v=neg_advs_v1, action_num=action_num, action=env_action[:, :, 0],
                                            q_ctx=q_ctx)
                    neg_advs2 = compute_adv(neg_advs_v=neg_advs_v2, action_num=action_num, action=env_action[:, :, 1],
                                            q_ctx=q_ctx)
                    v_grads = mx.nd.array(0.5 * neg_advs_v[:, np.newaxis],
                                          ctx=q_ctx)

                    env_ob = np.asarray(list(chain.from_iterable(env_ob)))
                    data_batch1 = mx.io.DataBatch(data=[mx.nd.array(env_ob[:, :74], ctx=q_ctx)], label=None)
                    data_batch2 = mx.io.DataBatch(data=[mx.nd.array(env_ob[:, -74:], ctx=q_ctx)], label=None)

                    a1_policy.model.reshape([('data', (len(env_ob), 74))])
                    a2_policy.model.reshape([('data', (len(env_ob), 74))])
                    a1_policy.model.forward(data_batch1, is_train=True)
                    a2_policy.model.forward(data_batch2, is_train=True)
                    a1_policy.model.backward(out_grads=[neg_advs1])
                    a2_policy.model.backward(out_grads=[neg_advs2])
                    a1_policy.model.update()
                    a2_policy.model.update()

                    step_policy1 = a1_policy.model.get_outputs()[2].asnumpy()
                    step_policy2 = a2_policy.model.get_outputs()[2].asnumpy()

                    data_batch = np.append(env_ob, step_policy1, axis=1)
                    data_batch = np.append(data_batch, step_policy2, axis=1)

                    Qnet.model.reshape([('data', (len(env_ob), 156))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(data_batch, ctx=q_ctx)], label=None)
                    Qnet.model.forward(data_batch, is_train=True)
                    Qnet.model.backward(out_grads=[v_grads])
                    Qnet.model.update()

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
        a1_policy.model.save_params('policy1/network-dqn_mx%04d.params' % epoch)
        a2_policy.model.save_params('policy2/network-dqn_mx%04d.params' % epoch)
        Qnet.model.save_params('Qnet/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))
