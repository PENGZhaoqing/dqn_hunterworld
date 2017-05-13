from actionQnet_ import *
import argparse
from itertools import chain
from utils import *
import numpy as np
import mxnet as mx
from config import Config
import matplotlib.pyplot as plt
from Env import Env

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

    if not os.path.exists('a3c'):
        os.makedirs('a3c')

    testing = False
    if testing:
        args.num_envs = 1
        # else:
        # logging_config("log", "single-output-custom")

    envs = []
    for i in range(args.num_envs):
        envs.append(Env())

    params_file = None
    if testing:
        envs[0].force_fps = False
        envs[0].game.draw = True
        envs[0].display_screen = True
        params_file = 'a3c/network-dqn_mx0012.params'

    a3c = A3c(74, 4, config=config)

    if params_file is not None:
        a3c.model.load_params(params_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(a3c.model)

    action_set = envs[0].ple.get_action_set()
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
            for env in envs:
                env.ple.reset_game()
                env.clear()
            step_ob = [env.ple.get_states() for env in envs]
            step_ob_np = np.array(step_ob)
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
                    if not env.done:
                        next_ob, reward, terminal_flag = env.ple.act_action(
                            [action_map[step_action[i]]])
                        env.add(step_ob[i], np.array(step_ob_np[i]), step_action[i], step_value[i][0], reward)
                        env.done = terminal_flag
                        step_ob[i] = next_ob
                        step_ob_np[i] = next_ob
                        episode_reward[i] += reward[0]
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1

                if t == t_max:
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(step_ob, ctx=q_ctx)], label=None)
                    a3c.model.forward(data_batch, is_train=False)
                    _, extra_value, _, _ = a3c.model.get_outputs()
                    extra_value = extra_value.asnumpy()
                    for idx, env in enumerate(envs):
                        if env.done:
                            env.value.append(0.0)
                        else:
                            env.value.append(extra_value[idx][0])

                    advs = []
                    for env in envs:
                        R = env.value[-1]
                        tmp = []
                        for j in range(len(env.reward))[::-1]:
                            R = env.reward[j][0] + gamma * R
                            tmp.append(R - env.value[j])
                        advs.extend(tmp[::-1])

                    neg_advs_v = -np.asarray(advs)
                    env_action = []
                    for env in envs:
                        for action in env.action:
                            env_action.append(action)
                    # action = np.asarray(list(chain.from_iterable(env_action)))
                    neg_advs_np = np.zeros((len(advs), action_num), dtype=np.float32)
                    neg_advs_np[np.arange(neg_advs_np.shape[0]), env_action] = neg_advs_v
                    neg_advs = mx.nd.array(neg_advs_np, ctx=q_ctx)
                    value_grads = mx.nd.array(config.vf_wt * neg_advs_v[:, np.newaxis],
                                              ctx=q_ctx)

                    env_ob = []
                    env_ob_np = []

                    for env in envs:
                        for idx, ob in enumerate(env.ob):
                            env_ob.append(env.ob[idx])
                            env_ob_np.append(env.ob_np[idx])
                            # print env.ob[idx]
                            # print
                            # print env.ob_np[idx]
                            # print

                    # extend_ob = list(chain.from_iterable(env_ob))
                    a3c.model.reshape([('data', (len(env_ob), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob_np, ctx=q_ctx)], label=None)
                    a3c.model.forward(data_batch, is_train=True)
                    a3c.model.backward(out_grads=[neg_advs, value_grads])
                    a3c.model.update()

                    for env in envs:
                        env.clear()

                    training_steps += 1
                    episode_advs += np.mean(neg_advs_v)
                    t = 0
                all_done = np.all([env.done for env in envs])
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
        a3c.model.save_params('a2_A3c/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))
