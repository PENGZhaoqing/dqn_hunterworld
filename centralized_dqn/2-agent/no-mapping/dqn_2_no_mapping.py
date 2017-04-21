import numpy as np
from env.v9.hunterworld_three import HunterWorld
from env.ple_v5 import PLE
from utils import *
import matplotlib.pyplot as plt
from ANN_2_no_mapping import createQNetwork, copyTargetQNetwork
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
    replay_start_size = 5000
    history_length = 1
    eps_start = 1.0
    eps_min = 0.1
    epoch_start = 0
    param_file = None

    if training_continue:
        epoch_start = 0

    if testing:
        env.force_fps = False
        env.display_screen = True
        replay_start_size = 32
        param_file = "saved_networks/network-dqn_mx0009.params"
        epoch_start = 19
        eps_start = 0.01
        eps_min = 0.01

    replay_memory = ReplayMemory(state_dim=(148,), action_dim=(2,), reward_dim=(2,),
                                 history_length=history_length,
                                 memory_size=replay_memory_size,
                                 replay_start_size=replay_start_size, state_dtype='float32')

    eps_decay = (eps_start - eps_min) / 800000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32

    action_set = env.get_action_set()
    action1_map = action_set[0].values()
    action2_map = action_set[1].values()
    # action3_map = action_set[2].values()
    action_matrix = np.zeros(8, dtype=np.int8)
    action_num = 8

    target1 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1)
    target32 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32)
    Qnet = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32)

    if param_file is not None:
        Qnet.load_params(param_file)
    copyTargetQNetwork(Qnet, target1)
    copyTargetQNetwork(Qnet, target32)

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
            frozen = 0
            while not env.game_over():
                if replay_memory.size >= history_length and replay_memory.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(4)
                        action2 = np.random.randint(4)
                        # action3 = np.random.randint(4)
                    else:
                        current_state = replay_memory.latest_slice()
                        state = nd.array(current_state.reshape((1,) + current_state.shape),
                                         ctx=q_ctx)
                        target1.forward(mx.io.DataBatch([state], []))
                        q_value = target1.get_outputs()[0].asnumpy()[0]
                        action1 = numpy.argmax(q_value[0:4])
                        action2 = numpy.argmax(q_value[4:8])
                        # action3 = numpy.argmax(q_value[8:12])
                        episode_q_value += q_value[action1]
                        episode_q_value += q_value[action2 + 4]
                        # episode_q_value += q_value[action3 + 8]
                        episode_action_step += 1
                else:
                    action1 = np.random.randint(4)
                    action2 = np.random.randint(4)
                    # action3 = np.random.randint(4)

                next_ob, reward, terminal_flag = env.act(
                    [action1_map[action1], action2_map[action2]])

                sum_reward = sum(reward)
                if sum_reward == 0.0:
                    frozen += 1
                elif sum_reward < 0.0:
                    collisions += 1
                    frozen = 0
                else:
                    frozen = 0

                if frozen > 100:
                    break

                replay_memory.append(np.array(next_ob).flatten(), [action1, action2], reward, terminal_flag)

                # if reward > 0:
                # plt.imshow(next_ob, cmap='gray')
                # plt.show()

                total_steps += 1
                episode_reward += sum_reward
                if sum_reward < 0:
                    collisions += 1
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory.size > replay_start_size:
                    training_steps += 1

                    state_batch, actions, rewards, nextstate_batch, terminate_flags = replay_memory.sample(
                        batch_size=minibatch_size)
                    state_batch = nd.array(state_batch, ctx=q_ctx)
                    actions_batch = nd.array(actions, ctx=q_ctx)
                    reward_batch = nd.array(rewards, ctx=q_ctx)
                    terminate_flags = nd.array(terminate_flags, ctx=q_ctx)

                    target32.forward(mx.io.DataBatch([nd.array(nextstate_batch, ctx=q_ctx)], []))
                    Qvalue = target32.get_outputs()[0]

                    Qvalue1 = nd.slice_axis(Qvalue, axis=1, begin=0, end=4)
                    Qvalue2 = nd.slice_axis(Qvalue, axis=1, begin=4, end=8)
                    # Qvalue3 = nd.slice_axis(Qvalue, axis=1, begin=8, end=12)

                    reward1 = nd.choose_element_0index(reward_batch, nd.array([0] * 32, ctx=q_ctx))
                    reward2 = nd.choose_element_0index(reward_batch, nd.array([1] * 32, ctx=q_ctx))
                    # reward3 = nd.choose_element_0index(reward_batch, nd.array([2] * 32, ctx=q_ctx))

                    y_batch1 = reward1 + nd.choose_element_0index(Qvalue1, nd.argmax_channel(Qvalue1)) * (
                        1.0 - terminate_flags) * discount

                    y_batch2 = reward2 + nd.choose_element_0index(Qvalue2, nd.argmax_channel(Qvalue2)) * (
                        1.0 - terminate_flags) * discount

                    # y_batch3 = reward3 + nd.choose_element_0index(Qvalue3, nd.argmax_channel(Qvalue3)) * (
                    #     1.0 - terminate_flags) * discount

                    Qnet.forward(mx.io.DataBatch([state_batch, actions_batch, y_batch1, y_batch2], []),
                                 is_train=True)
                    Qnet.backward()
                    Qnet.update()

                    if training_steps % 10 == 0:
                        action1_batch = nd.choose_element_0index(reward_batch, nd.array([0] * 32, ctx=q_ctx))
                        action2_batch = nd.choose_element_0index(reward_batch, nd.array([1] * 32, ctx=q_ctx))
                        # action3_batch = nd.choose_element_0index(reward_batch, nd.array([2] * 32, ctx=q_ctx))
                        loss1 = nd.square(nd.choose_element_0index(Qnet.get_outputs()[0], action1_batch) - y_batch1)
                        loss2 = nd.square(nd.choose_element_0index(Qnet.get_outputs()[0], action2_batch) - y_batch2)
                        # loss3 = nd.square(nd.choose_element_0index(Qnet.get_outputs()[0], action3_batch) - y_batch3)
                        episode_loss += nd.sum(loss1).asnumpy()
                        episode_loss += nd.sum(loss2).asnumpy()
                        # episode_loss += nd.sum(loss3).asnumpy()
                        episode_update_step += 1

                    if training_steps % freeze_interval == 0:
                        copyTargetQNetwork(Qnet, target1)
                        copyTargetQNetwork(Qnet, target32)

            steps_left -= episode_step
            time_episode_end = time.time()
            epoch_reward += episode_reward
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                       % (epoch, episode, steps_left, episode_step, steps_per_epoch, episode_reward,
                          episode_step / (time_episode_end - time_episode_start), eps_curr)

            if collisions > 0:
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
        Qnet.save_params('saved_networks/network-dqn_mx%04d.params' % epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))


if __name__ == '__main__':
    main()
