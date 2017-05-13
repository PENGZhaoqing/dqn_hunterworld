import argparse
import logging
from Env_two import Env
from A3c_network import *
from config import Config
from utils import *


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
    parser.add_argument('--logging.info-every', type=int, default=1)

    args = parser.parse_args()
    config = Config(args)
    np.random.seed(args.seed)

    if not os.path.exists('a1_A3c'):
        os.makedirs('a1_A3c')

    testing = False
    if testing:
        args.num_envs = 32
    else:
        logging_config(logging, 'log', 'centra_tmax_' + str(args.t_max))

    envs = []
    for i in range(args.num_envs):
        envs.append(Env(i))

    a1_A3c_file = None
    if testing:
        envs[0].force_fps = True
        envs[0].game.draw = True
        envs[0].display_screen = True
        a1_A3c_file = 'a1_A3c/network-dqn_mx0019.params'

    action_num = 4
    a1_A3c = A3c(state_dim=148, act_space=action_num, config=config)

    if a1_A3c_file is not None:
        a1_A3c.model.load_params(a1_A3c_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(logging, a1_A3c.model)

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
            for i in range(num_envs):
                envs[i].reset_var()
                envs[i].ple.reset_game()
                ob = envs[i].ple.get_states()
                ob = np.append(ob[0], ob[1])
                step_ob1.append(ob)

            all_done = False
            t = 1
            training_steps = 0
            sample_num = 0.0
            episode_advs1 = 0
            episode_values1 = 0
            episode_policys1 = 0
            episode_policys2 = 0

            while not all_done:
                _, _, step_value1, _, _, step_policy1, step_policy2 = a1_A3c.forward(step_ob1, is_train=False)

                step_policy1 = step_policy1.asnumpy()
                step_policy2 = step_policy2.asnumpy()
                step_value1 = step_value1.asnumpy()

                us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
                step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                for i in range(num_envs):
                    if not envs[i].done:
                        next_ob, reward, done = envs[i].ple.act(
                            [action_map1[step_action1[i]], action_map2[step_action2[i]]])
                        reward = sum(reward)
                        envs[i].add(step_ob1[i], step_action1[i], step_action2[i], step_value1[i][0], reward)
                        envs[i].done = done
                        step_ob1[i] = np.append(next_ob[0], next_ob[1])
                        episode_reward[i] += reward
                        if reward < 0:
                            collisions[i] += 1
                        episode_step += 1

                if t == t_max:

                    extra_value1 = a1_A3c.forward(step_ob1, is_train=False)[2]
                    extra_value1 = extra_value1.asnumpy()

                    for i in range(num_envs):
                        if envs[i].done:
                            envs[i].value1.append(extra_value1[i][0])
                        else:
                            envs[i].value1.append(extra_value1[i][0])

                    a1_advs = []
                    for i in range(num_envs):
                        R1 = envs[i].value1[-1]
                        tmp_R1 = []
                        for j in range(len(envs[i].reward1))[::-1]:
                            R1 = envs[i].reward1[j] + gamma * R1
                            tmp_R1.append(R1 - envs[i].value1[j])
                        a1_advs.extend(tmp_R1[::-1])

                    a1_neg_advs_v = -np.asarray(a1_advs)

                    env_action1 = []
                    env_action2 = []
                    for i in range(num_envs):
                        env_action1.extend(envs[i].action1)
                        env_action2.extend(envs[i].action2)

                    neg_advs_np1 = np.zeros((len(a1_advs), action_num), dtype=np.float32)
                    neg_advs_np1[np.arange(neg_advs_np1.shape[0]), env_action1] = a1_neg_advs_v
                    neg_advs1 = mx.nd.array(neg_advs_np1, ctx=q_ctx)

                    neg_advs_np2 = np.zeros((len(a1_advs), action_num), dtype=np.float32)
                    neg_advs_np2[np.arange(neg_advs_np2.shape[0]), env_action2] = a1_neg_advs_v
                    neg_advs2 = mx.nd.array(neg_advs_np2, ctx=q_ctx)

                    value_grads1 = mx.nd.array(config.vf_wt * a1_neg_advs_v[:, np.newaxis],
                                               ctx=q_ctx)
                    env_ob1 = []
                    for i in range(num_envs):
                        env_ob1.extend(envs[i].ob1)

                    _, _, episode_value1, _, _, episode_policy1, episode_policy2 = a1_A3c.forward(env_ob1,
                                                                                                  is_train=True)
                    a1_A3c.model.backward(out_grads=[neg_advs1, neg_advs2, value_grads1])
                    a1_A3c.model.update()

                    for i in range(num_envs):
                        envs[i].clear()

                    training_steps += 1
                    if training_steps % 10 == 0:
                        sample_num += 1
                        episode_advs1 += np.mean(a1_neg_advs_v)
                        episode_values1 += np.mean(episode_value1.asnumpy())
                        episode_policys1 += np.max(episode_policy1[0].asnumpy())
                        episode_policys2 += np.max(episode_policy2[0].asnumpy())
                    t = 0

                all_done = np.all([envs[i].done for i in range(num_envs)])
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_reward
            time_episode_end = time.time()
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, Collisions:%f, fps:%f, Advs:%f/%d, Value:%f, Policys:%f/%f" \
                       % (
                           epoch, episode, steps_left, episode_step, steps_per_epoch,
                           np.mean(episode_reward), np.mean(collisions),
                           episode_step / (time_episode_end - time_episode_start),
                           episode_advs1 / sample_num, training_steps, episode_values1 / sample_num,
                           episode_policys1 / sample_num, episode_policys2 / sample_num)
            logging.info(info_str)
            logging.info(str(episode_reward))
            print info_str
            print str(episode_reward)

        end = time.time()
        fps = steps_per_epoch / (end - start)
        a1_A3c.model.save_params('a1_A3c/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))


if __name__ == '__main__':
    main()
