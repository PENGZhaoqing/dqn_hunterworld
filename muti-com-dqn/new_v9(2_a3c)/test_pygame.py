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

# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter()
# ch.setFormatter(formatter)
# root.addHandler(ch)

mx.random.seed(100)
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

            a1_A3c.model.reshape([('data', (1, 74))])
            a2_A3c.model.reshape([('data', (1, 74))])
            while not all_done:

                for i, env in enumerate(envs):
                    if not env.done:

                        # if i == 1:
                        #     continue

                        data_batch1 = mx.io.DataBatch(
                            data=[mx.nd.array(np.array(step_ob1[i]).reshape((1, 74)), ctx=q_ctx)], label=None)
                        a1_A3c.model.forward(data_batch1, is_train=False)
                        _, step_value1, _, step_policy1 = a1_A3c.model.get_outputs()

                        data_batch2 = mx.io.DataBatch(
                            data=[mx.nd.array(np.array(step_ob2[i]).reshape((1, 74)), ctx=q_ctx)], label=None)
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

                        next_ob, reward, done = env.ple.act(
                            [action_map1[step_action1[0]], action_map2[step_action2[0]]])

                        env.done = done
                        step_ob1[i] = next_ob[0]
                        step_ob2[i] = next_ob[1]

                        # print (step_value1[0][0], step_value2[0][0])

                all_done = np.all([env.done for env in envs])
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_reward

        end = time.time()
        fps = steps_per_epoch / (end - start)
        a1_A3c.model.save_params('a1_A3c/network-dqn_mx%04d.params' % epoch)
        a2_A3c.model.save_params('a2_A3c/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))


if __name__ == '__main__':
    main()
