import numpy as np
from env.v8.hunterworld import HunterWorld
from env.ple_v5 import PLE
from utils import *
import matplotlib.pyplot as plt
from actionQnet import *
from replay_memory import ReplayMemory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


# kernprof -l hunterdqn-v3-single/asyn_dqn.py
# python -m line_profiler asyn_dqn.py.lprof
# @profile
def main():
    rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": -0.002,
        "loss": -2.0,
        "win": 2.0
    }

    testing = True
    training_continue = False

    q_ctx = mx.gpu(0)

    game = HunterWorld(width=256, height=256, num_preys=10,
                       num_hunters=1, num_toxins=5)

    env = PLE(game, fps=30, force_fps=True, display_screen=False,
              reward_values=rewards,
              resized_rows=80, resized_cols=80, num_steps=2)

    freeze_interval = 10000
    epoch_num = 20
    steps_per_epoch = 100000
    update_interval = 5
    replay_memory_size = 1000000
    discount = 0.99
    replay_start_size = 50000
    history_length = 1
    eps_start = 1.0
    eps_min = 0.1
    epoch_start = 0
    a1_act_param_file = None

    if training_continue:
        epoch_start = 0

    if testing:
        env.force_fps = False
        env.display_screen = True
        replay_start_size = 32
        a1_act_param_file = 'a1_actQnet/network-dqn_mx0008.params'
        epoch_start = 19
        eps_start = 0.1
        eps_min = 0.1

    if not testing:
        logging_config("log", "single-custom")

    replay_memory = ReplayMemory(state_dim=(74,),
                                 history_length=history_length,
                                 memory_size=replay_memory_size,
                                 replay_start_size=replay_start_size, state_dtype='float32')

    eps_decay = (eps_start - eps_min) / 500000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32
    soft_target_tau = 0.001
    action_set = env.get_action_set()
    action_map1 = []
    for action1 in action_set[0].values():
        action_map1.append(action1)

    action_num = 4
    a1_actQnet = actQnet(actions_num=action_num, ctx=q_ctx)
    a1_actTarget = target_actQnet(Qnet=a1_actQnet, ctx=q_ctx)

    if a1_act_param_file is not None:
        load_params(a1_actQnet, a1_act_param_file)
    copy_params_to(a1_actQnet, a1_actTarget)

    print_params1(a1_actQnet)

    training_steps = 0
    total_steps = 0
    for epoch in range(epoch_start, epoch_num):
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        env.reset_game()
        while steps_left > 0:
            episode += 1
            episode_loss = 0.0
            episode_q_value = 0.0
            episode_update_step = 0
            episode_action_step = 0
            episode_reward = 0
            episode_step = 0
            collisions = 0.0
            time_episode_start = time.time()
            env.reset_game()
            # action_status[:] = 0
            while not env.game_over():
                if replay_memory.size >= history_length and replay_memory.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(action_num)
                    else:
                        current_state = replay_memory.latest_slice()
                        state1 = nd.array(current_state, ctx=q_ctx)
                        q_value1 = a1_actTarget.get_qval(state1)[0].asnumpy()
                        action1 = numpy.argmax(q_value1)
                        episode_q_value += q_value1[action1]
                        episode_action_step += 1
                else:
                    action1 = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act([action_map1[action1], 0])
                replay_memory.append(next_ob[0], action1, reward, terminal_flag)

                total_steps += 1
                sum_reward = sum(reward)
                episode_reward += sum_reward
                if sum_reward < 0:
                    collisions += 1
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory.size > replay_start_size:
                    training_steps += 1

                    state_batch, actions, rewards, nextstate_batch, terminate_flags = replay_memory.sample(
                        batch_size=minibatch_size)
                    state_batch1 = nd.array(state_batch.reshape(32, 74), ctx=q_ctx)
                    nextstate_batch1 = nd.array(nextstate_batch.reshape(32, 74), ctx=q_ctx)

                    actions_batch1 = nd.array(actions, ctx=q_ctx)
                    reward_batch1 = nd.array(rewards, ctx=q_ctx)
                    terminate_flags = nd.array(terminate_flags, ctx=q_ctx)

                    a1_Qvalue = a1_actTarget.get_qvals(nextstate_batch1)

                    y_batch1 = reward_batch1 + nd.choose_element_0index(a1_Qvalue, nd.argmax_channel(a1_Qvalue)) * (
                        1.0 - terminate_flags) * discount

                    Qnet1 = a1_actQnet.update_params(state_batch1, actions_batch1, y_batch1)

                    if training_steps % 10 == 0:
                        loss1 = 0.5 * nd.square(nd.choose_element_0index(Qnet1, actions_batch1) - y_batch1)
                        episode_loss += nd.sum(loss1).asnumpy()
                        episode_update_step += 1

                    soft_copy_params_to(a1_actQnet, a1_actTarget, soft_target_tau)

            steps_left -= episode_step
            time_episode_end = time.time()
            epoch_reward += episode_reward
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                       % (epoch, episode, steps_left, episode_step, steps_per_epoch, episode_reward,
                          episode_step / (time_episode_end - time_episode_start), eps_curr)

            info_str += ", Collision:%f/%d " % (collisions / episode_step,
                                                collisions)

            if episode_update_step > 0:
                info_str += ", Avg Loss:%f/%d" % (episode_loss / episode_update_step,
                                                  episode_update_step * 10)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d " % (episode_q_value / episode_action_step,
                                                      episode_action_step)

            if episode % 1 == 0:
                logging.info(info_str)

        end = time.time()
        fps = steps_per_epoch / (end - start)
        save_params(a1_actQnet, 'a1_actQnet/network-dqn_mx%04d.params' % epoch)

        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))


if __name__ == '__main__':
    main()
