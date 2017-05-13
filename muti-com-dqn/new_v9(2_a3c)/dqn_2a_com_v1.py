from actionQnet_ import *
import argparse
from itertools import chain
from utils import *
import numpy as np
import mxnet as mx
from config import Config
from Env_two import Env
import time

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)
#
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# root.addHandler(ch)
# mx.random.seed(100)


def _2d_list(n):
    return [[] for _ in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=1)
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=1)

    args = parser.parse_args()
    config = Config(args)
    np.random.seed(args.seed)

    if not os.path.exists('a1_A3c'):
        os.makedirs('a1_A3c')
    if not os.path.exists('a2_A3c'):
        os.makedirs('a2_A3c')

    testing = True
    if testing:
        args.num_envs = 2
    else:
        logging_config("log", "single-output-custom")

    envs = []
    for i in range(args.num_envs):
        envs.append(Env())

    a1_A3c_file = None
    a2_A3c_file = None
    if testing:
        envs[0].ple.force_fps = True
        envs[0].game.draw = True
        envs[0].ple.display_screen = True

        envs[1].ple.force_fps = True
        envs[1].game.draw = True
        envs[1].ple.display_screen = True

        a1_A3c_file = 'a1_A3c/network-dqn_mx0014.params'
        a2_A3c_file = 'a2_A3c/network-dqn_mx0014.params'

    a1_A3c = A3c(74, 4, config=config)
    a2_A3c = A3c(74, 4, config=config)

    if a1_A3c_file is not None:
        a1_A3c.model.load_params(a1_A3c_file)
        a2_A3c.model.load_params(a2_A3c_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(a1_A3c.model)

    action_set = envs[0].ple.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)
    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)

    action_num = len(action_map1)

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

            step_ob1 = []
            step_ob2 = []
            for env in envs:
                env.reset_var()
                env.ple.reset_game()
                ob = env.ple.get_states()
                step_ob1.append(ob[0])
                step_ob2.append(ob[1])

            all_done = False
            t = 1
            training_steps = 0.0
            episode_advs1 = 0
            episode_advs2 = 0
            episode_values1 = 0
            episode_values2 = 0
            episode_policys1 = 0
            episode_policys2 = 0

            while not all_done:

                data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob1, ctx=q_ctx)], label=None)
                a1_A3c.model.reshape([('data', (num_envs, 74))])
                a1_A3c.model.forward(data_batch1, is_train=False)
                _, step_value1, _, step_policy1 = a1_A3c.model.get_outputs()

                data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob2, ctx=q_ctx)], label=None)
                a2_A3c.model.reshape([('data', (num_envs, 74))])
                a2_A3c.model.forward(data_batch2, is_train=False)
                _, step_value2, _, step_policy2 = a2_A3c.model.get_outputs()

                step_policy1 = step_policy1.asnumpy()
                step_value1 = step_value1.asnumpy()
                step_policy2 = step_policy2.asnumpy()
                step_value2 = step_value2.asnumpy()

                us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
                step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                for i, env in enumerate(envs):
                    if not env.done:
                        next_ob, reward, done = env.ple.act(
                            [action_map1[step_action1[i]], action_map2[step_action2[i]]])
                        env.add(step_ob1[i], step_ob2[i], step_action1[i], step_action2[i], step_value1[i][0],
                                step_value2[i][0], reward[0], reward[1])

                        env.done = done
                        step_ob1[i] = next_ob[0]
                        step_ob2[i] = next_ob[1]
                        episode_reward[i] += sum(reward)
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1

                if t == t_max:

                    data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob1, ctx=q_ctx)], label=None)
                    a1_A3c.model.forward(data_batch1, is_train=False)
                    _, extra_value1, _, _ = a1_A3c.model.get_outputs()

                    data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob2, ctx=q_ctx)], label=None)
                    a2_A3c.model.forward(data_batch2, is_train=False)
                    _, extra_value2, _, _ = a2_A3c.model.get_outputs()

                    extra_value1 = extra_value1.asnumpy()
                    extra_value2 = extra_value2.asnumpy()

                    for idx, env in enumerate(envs):
                        if env.done:
                            env.value1.append(extra_value1[idx][0])
                            env.value2.append(extra_value2[idx][0])
                        else:
                            env.value1.append(extra_value1[idx][0])
                            env.value2.append(extra_value2[idx][0])

                    a1_advs = []
                    a2_advs = []
                    for env in envs:
                        R1 = env.value1[-1]
                        R2 = env.value2[-1]
                        tmp_R1 = []
                        tmp_R2 = []
                        for j in range(len(env.reward1))[::-1]:
                            R1 = env.reward1[j] + gamma * R1
                            R2 = env.reward2[j] + gamma * R2
                            tmp_R1.append(R1 - env.value1[j])
                            tmp_R2.append(R2 - env.value2[j])
                        a1_advs.extend(tmp_R1[::-1])
                        a2_advs.extend(tmp_R2[::-1])

                    a1_neg_advs_v = -np.asarray(a1_advs)
                    a2_neg_advs_v = -np.asarray(a2_advs)

                    env_action1 = []
                    for env in envs:
                        env_action1.extend(env.action1)
                    neg_advs_np1 = np.zeros((len(a1_advs), action_num), dtype=np.float32)
                    neg_advs_np1[np.arange(neg_advs_np1.shape[0]), env_action1] = a1_neg_advs_v
                    neg_advs1 = mx.nd.array(neg_advs_np1, ctx=q_ctx)

                    env_action2 = []
                    for env in envs:
                        env_action2.extend(env.action2)
                    neg_advs_np2 = np.zeros((len(a2_advs), action_num), dtype=np.float32)
                    neg_advs_np2[np.arange(neg_advs_np2.shape[0]), env_action2] = a2_neg_advs_v
                    neg_advs2 = mx.nd.array(neg_advs_np2, ctx=q_ctx)

                    value_grads1 = mx.nd.array(config.vf_wt * a1_neg_advs_v[:, np.newaxis],
                                               ctx=q_ctx)
                    value_grads2 = mx.nd.array(config.vf_wt * a2_neg_advs_v[:, np.newaxis],
                                               ctx=q_ctx)

                    env_ob1 = []
                    for env in envs:
                        env_ob1.extend(env.ob1)
                    a1_A3c.model.reshape([('data', (len(env_ob1), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob1, ctx=q_ctx)], label=None)
                    a1_A3c.model.forward(data_batch, is_train=True)
                    _, episode_value1, _, episode_policy1 = a1_A3c.model.get_outputs()
                    a1_A3c.model.backward(out_grads=[neg_advs1, value_grads1])
                    a1_A3c.model.update()

                    env_ob2 = []
                    for env in envs:
                        env_ob2.extend(env.ob2)
                    a2_A3c.model.reshape([('data', (len(env_ob2), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob2, ctx=q_ctx)], label=None)
                    a2_A3c.model.forward(data_batch, is_train=True)
                    _, episode_value2, _, episode_policy2 = a2_A3c.model.get_outputs()
                    a2_A3c.model.backward(out_grads=[neg_advs2, value_grads2])
                    a2_A3c.model.update()

                    for env in envs:
                        env.clear()

                    training_steps += 1
                    episode_advs1 += np.mean(a1_neg_advs_v)
                    episode_advs2 += np.mean(a2_neg_advs_v)
                    episode_values1 += np.mean(episode_value1.asnumpy())
                    episode_values2 += np.mean(episode_value2.asnumpy())
                    episode_policys1 += np.max(episode_policy1[0].asnumpy())
                    episode_policys2 += np.max(episode_policy2[0].asnumpy())
                    t = 0

                all_done = np.all([env.done for env in envs])
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_reward
            time_episode_end = time.time()
            if episode % args.print_every == 0 and training_steps > 0:
                info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, Collisions:%f, fps:%f, Advs:%f/%f/%d, Value:%f/%f, Policys:%f/%f" \
                           % (
                               epoch, episode, steps_left, episode_step, steps_per_epoch,
                               np.mean(episode_reward), np.mean(collisions),
                               episode_step / (time_episode_end - time_episode_start),
                               episode_advs1 / training_steps, episode_advs2 / training_steps, training_steps,
                               episode_values1 / training_steps, episode_values2 / training_steps,
                               episode_policys1 / training_steps, episode_policys2 / training_steps)
                info_str += ", " + str(episode_reward)
                logging.info(info_str)

        end = time.time()
        fps = steps_per_epoch / (end - start)
        a1_A3c.model.save_params('a1_A3c/network-dqn_mx%04d.params' % epoch)
        a2_A3c.model.save_params('a2_A3c/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))


if __name__ == '__main__':
    main()
