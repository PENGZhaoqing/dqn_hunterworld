import numpy as np
from env.v9.hunterworld_v2 import HunterWorld
from env.ple_v5 import PLE
from utils import *
import matplotlib.pyplot as plt
from ANN_2 import createQNetwork, copyTargetQNetwork, createQNetwork1
from replay_memory import ReplayMemory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


# kernprof -l centralized_dqn/asyn_dqn.py
# python -m line_profiler asyn_dqn.py.lprof

# @profile
def main():
    rewards = {
        "positive": 1.0,
        "negative": -1,
        "tick": -0.002,
        "loss": -2.0,
        "win": 2.0
    }
    testing = False
    training_continue = False

    ctx = parse_ctx('gpu')
    q_ctx = mx.Context(*ctx[0])

    game = HunterWorld(width=256, height=256, num_preys=10,
                       num_hunters=2, num_toxins=5)

    env = PLE(game, fps=30, force_fps=True, display_screen=False,
              reward_values=rewards,
              resized_rows=80, resized_cols=80, num_steps=2)

    env.init()

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
    param_file1a = None
    param_file2a = None

    if training_continue:
        epoch_start = 0

    if testing:
        env.force_fps = False
        env.display_screen = True
        replay_start_size = 32
        eps_start = 0.11
        epoch_start = 19
        param_file1a = "saved_networks_agent1/network-dqn_mx0009.params"
        param_file2a = "saved_networks_agent2/network-dqn_mx0009.params"

    if not testing:
        logging_config(folder_name="log", name="com_network_a2")

    replay_memory1 = ReplayMemory(state_dim=(74,),
                                  history_length=history_length,
                                  memory_size=replay_memory_size,
                                  replay_start_size=replay_start_size, state_dtype='float16')

    replay_memory2 = ReplayMemory(state_dim=(78,),
                                  history_length=history_length,
                                  memory_size=replay_memory_size,
                                  replay_start_size=replay_start_size, state_dtype='float16')

    eps_decay = (eps_start - eps_min) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32

    action_num = 4
    action_matrix = np.zeros((action_num, action_num), dtype=np.int8)
    for i in range(action_num):
        action_matrix[i, i] = 1

    action_set = env.get_action_set()

    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)

    a1_target1 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1)
    a1_target32 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32)
    a1_Qnet = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32)

    a2_target1 = createQNetwork1(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1)
    a2_target32 = createQNetwork1(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32)
    a2_Qnet = createQNetwork1(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32)

    if param_file1a is not None:
        a1_Qnet.load_params(param_file1a)
    copyTargetQNetwork(a1_Qnet, a1_target1)
    copyTargetQNetwork(a1_Qnet, a1_target32)

    if param_file2a is not None:
        a2_Qnet.load_params(param_file2a)
    copyTargetQNetwork(a2_Qnet, a2_target1)
    copyTargetQNetwork(a2_Qnet, a2_target32)

    print_params(a1_Qnet)
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
            while not env.game_over():
                if replay_memory1.size >= history_length and replay_memory1.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(action_num)
                        action2 = np.random.randint(action_num)
                    else:
                        current_state1 = replay_memory1.latest_slice()
                        current_state2 = replay_memory2.latest_slice()
                        state1 = nd.array(current_state1.reshape((1,) + current_state1.shape),
                                          ctx=q_ctx)
                        a1_target1.forward(mx.io.DataBatch([state1], []))
                        q_value1 = a1_target1.get_outputs()[0].asnumpy()[0]
                        action1 = numpy.argmax(q_value1)
                        episode_q_value += q_value1[action1]

                        # signal
                        signal = action_matrix[action1]
                        current_state2 = np.append(current_state2[:, :-4], [signal], axis=1)
                        state2 = nd.array(current_state2.reshape((1,) + current_state2.shape),
                                          ctx=q_ctx)

                        a2_target1.forward(mx.io.DataBatch([state2], []))
                        q_value2 = a2_target1.get_outputs()[0].asnumpy()[0]
                        action2 = numpy.argmax(q_value2)
                        episode_q_value += q_value2[action2]

                        episode_action_step += 1
                else:
                    action1 = np.random.randint(action_num)
                    action2 = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act([action_map1[action1], action_map2[action2]])
                sum_reward = sum(reward)
                if sum_reward < 0.0:
                    collisions += 1

                replay_memory1.append(next_ob[0].flatten(), action1, reward[0], terminal_flag)
                replay_memory2.append(np.append(next_ob[1].flatten(), action_matrix[action1]), action2, reward[1],
                                      terminal_flag)

                # if np.sum(reward) != 0:
                # plt.imshow(game.get_screen_rgb(), cmap='gray')
                # plt.show()
                # print reward

                total_steps += 1
                episode_reward += sum_reward

                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory1.size > replay_start_size:
                    training_steps += 1

                    state_batch1, actions1, rewards1, nextstate_batch1, terminate_flags1 = replay_memory1.sample(
                        batch_size=minibatch_size)
                    state_batch1 = nd.array(state_batch1, ctx=q_ctx)
                    actions_batch1 = nd.array(actions1, ctx=q_ctx)
                    reward_batch1 = nd.array(rewards1, ctx=q_ctx)
                    terminate_flags1 = nd.array(terminate_flags1, ctx=q_ctx)

                    state_batch2, actions2, rewards2, nextstate_batch2, terminate_flags2 = replay_memory2.sample(
                        batch_size=minibatch_size)
                    state_batch2 = nd.array(state_batch2, ctx=q_ctx)
                    actions_batch2 = nd.array(actions2, ctx=q_ctx)
                    reward_batch2 = nd.array(rewards2, ctx=q_ctx)
                    terminate_flags2 = nd.array(terminate_flags2, ctx=q_ctx)

                    a1_target32.forward(mx.io.DataBatch([nd.array(nextstate_batch1, ctx=q_ctx)], []))
                    Qvalue1 = a1_target32.get_outputs()[0]

                    a2_target32.forward(mx.io.DataBatch([nd.array(nextstate_batch2, ctx=q_ctx)], []))
                    Qvalue2 = a2_target32.get_outputs()[0]

                    y_batch1 = reward_batch1 + nd.choose_element_0index(Qvalue1, nd.argmax_channel(Qvalue1)) * (
                        1.0 - terminate_flags1) * discount

                    y_batch2 = reward_batch2 + nd.choose_element_0index(Qvalue2, nd.argmax_channel(Qvalue2)) * (
                        1.0 - terminate_flags2) * discount

                    a1_Qnet.forward(mx.io.DataBatch([state_batch1, actions_batch1, y_batch1],
                                                    []), is_train=True)
                    a1_Qnet.backward()
                    a1_Qnet.update()

                    a2_Qnet.forward(mx.io.DataBatch([state_batch2, actions_batch2, y_batch2],
                                                    []), is_train=True)
                    a2_Qnet.backward()
                    a2_Qnet.update()

                    if training_steps % 10 == 0:
                        loss1 = 0.5 * nd.square(
                            nd.choose_element_0index(a1_Qnet.get_outputs()[0], actions_batch1) - y_batch1)
                        episode_loss += nd.sum(loss1).asnumpy()
                        loss2 = 0.5 * nd.square(
                            nd.choose_element_0index(a2_Qnet.get_outputs()[0], actions_batch2) - y_batch2)
                        episode_loss += nd.sum(loss2).asnumpy()
                        episode_update_step += 1

                    if training_steps % freeze_interval == 0:
                        copyTargetQNetwork(a1_Qnet, a1_target1)
                        copyTargetQNetwork(a1_Qnet, a1_target32)
                        copyTargetQNetwork(a2_Qnet, a2_target1)
                        copyTargetQNetwork(a2_Qnet, a2_target32)

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
                                                  episode_update_step)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d " % (episode_q_value / episode_action_step,
                                                      episode_action_step)
            if episode % 1 == 0:
                logging.info(info_str)

        end = time.time()
        fps = steps_per_epoch / (end - start)
        a1_Qnet.save_params('saved_networks_agent1/network-dqn_mx%04d.params' % epoch)
        a2_Qnet.save_params('saved_networks_agent2/network-dqn_mx%04d.params' % epoch)

        info_str = "Epoch:%d, FPS:%f, Avg Reward: %f/%d" % (epoch, fps, epoch_reward / float(episode), episode)
        logging.info(info_str)


if __name__ == '__main__':
    main()
