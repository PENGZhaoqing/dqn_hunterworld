import numpy as np
from hunterworld_v2 import HunterWorld
from ple_v5 import PLE
from logutils import *
import matplotlib.pyplot as plt
from actionQnet_ import *
from replay_memory import ReplayMemory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)


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

    q_ctx = mx.gpu(0)

    game = HunterWorld(width=256, height=256, num_preys=10,
                       num_hunters=2, num_toxins=5)

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
    a1_com_param_file = None
    a2_act_param_file = None
    a2_com_param_file = None

    if training_continue:
        epoch_start = 0

    if testing:
        env.force_fps = False
        env.display_screen = True
        replay_start_size = 32
        a1_act_param_file = 'a1_actQnet/network-dqn_mx0004.params'
        a1_com_param_file = 'a1_comNet/network-dqn_mx0004.params'
        a2_act_param_file = 'a2_actQnet/network-dqn_mx0004.params'
        a2_com_param_file = 'a2_comNet/network-dqn_mx0004.params'
        epoch_start = 19
        eps_start = 0.1
        eps_min = 0.1

    if not testing:
        logging_config("log", "single-output-custom")

    replay_memory1 = ReplayMemory(state_dim=(2, 74),
                                  history_length=history_length,
                                  memory_size=replay_memory_size,
                                  replay_start_size=replay_start_size, state_dtype='float32')

    replay_memory2 = ReplayMemory(state_dim=(2, 74),
                                  history_length=history_length,
                                  memory_size=replay_memory_size,
                                  replay_start_size=replay_start_size, state_dtype='float32')

    eps_decay = (eps_start - eps_min) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32
    soft_target_tau = 0.001
    action_set = env.get_action_set()
    action_map1 = []
    for action1 in action_set[0].values():
        action_map1.append(action1)

    action_map2 = []
    for action2 in action_set[1].values():
        action_map2.append(action2)

    # action_map1 = np.array(action_map1)
    # action_map2 = np.array(action_map2)

    action_num = 4
    signal_num = 10
    a1_comNet = comNet(signal_num=signal_num, ctx=q_ctx)
    a1_comTarget = target_comNet(comNet=a1_comNet, ctx=q_ctx)
    a1_actQnet = actQnet(actions_num=action_num, signal_num=signal_num, ctx=q_ctx)
    a1_actTarget = target_actQnet(Qnet=a1_actQnet, signal_num=signal_num, ctx=q_ctx)

    # a1_Qnet_grad_dict = dict(zip(
    #     a1_actQnet.dqn.list_arguments(),
    #     a1_actQnet.grad_arrays))

    a2_comNet = comNet(signal_num=signal_num, ctx=q_ctx)
    a2_comTarget = target_comNet(comNet=a2_comNet, ctx=q_ctx)
    a2_actQnet = actQnet(actions_num=action_num, signal_num=signal_num, ctx=q_ctx)
    a2_actTarget = target_actQnet(Qnet=a2_actQnet, signal_num=signal_num, ctx=q_ctx)

    # a2_Qnet_grad_dict = dict(zip(
    #     a2_actQnet.dqn.list_arguments(),
    #     a2_actQnet.grad_arrays))

    if a1_act_param_file is not None:
        load_params(a1_actQnet, a1_act_param_file)
        load_params(a1_comNet, a1_com_param_file)
        load_params(a2_actQnet, a2_act_param_file)
        load_params(a2_comNet, a2_com_param_file)

    print_params1(a1_actQnet)
    print_params1(a1_comNet)

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
                if replay_memory1.size >= history_length and replay_memory1.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(action_num)
                        # signal1 = (np.random.random((1, signal_num)) - 0.5) * 2
                        # signal2 = (np.random.random((1, signal_num)) - 0.5) * 2
                        action2 = np.random.randint(action_num)
                    else:
                        current_state1 = replay_memory1.latest_slice()
                        current_state2 = replay_memory2.latest_slice()
                        state1 = nd.array(current_state1[:, 0], ctx=q_ctx)
                        state2 = nd.array(current_state2[:, 1], ctx=q_ctx)
                        signal1 = a2_comTarget.get_singal(nd.array(current_state1[:, 1], ctx=q_ctx))
                        signal2 = a1_comTarget.get_singal(nd.array(current_state2[:, 0], ctx=q_ctx))
                        q_value1 = a1_actTarget.get_qval(state1, signal1)[0].asnumpy()
                        q_value2 = a1_actTarget.get_qval(state2, signal2)[0].asnumpy()
                        action1 = numpy.argmax(q_value1)
                        action2 = numpy.argmax(q_value2)
                        episode_q_value += q_value1[action1]
                        episode_q_value += q_value2[action2]
                        episode_action_step += 1
                else:
                    # signal1 = (np.random.random((1, signal_num)) - 0.5) * 2
                    # signal2 = (np.random.random((1, signal_num)) - 0.5) * 2
                    action1 = np.random.randint(action_num)
                    action2 = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act([action_map1[action1], action_map2[action2]])
                replay_memory1.append(next_ob, action1, reward[0], terminal_flag)
                replay_memory2.append(next_ob, action2, reward[1], terminal_flag)

                total_steps += 1
                sum_reward = sum(reward)
                episode_reward += sum_reward
                if sum_reward < 0:
                    collisions += 1
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory1.size > replay_start_size:
                    training_steps += 1

                    state_batch1, actions1, rewards1, nextstate_batch1, terminate_flags1 = replay_memory1.sample(
                        batch_size=minibatch_size)

                    state_batch2, actions2, rewards2, nextstate_batch2, terminate_flags2 = replay_memory2.sample(
                        batch_size=minibatch_size)

                    a1_state_batch = nd.array(state_batch1[:, :, 0].reshape(32, 74), ctx=q_ctx)
                    a1_signal_state = nd.array(state_batch1[:, :, 1].reshape(32, 74), ctx=q_ctx)
                    a2_state_batch = nd.array(state_batch2[:, :, 1].reshape(32, 74), ctx=q_ctx)
                    a2_signal_state = nd.array(state_batch2[:, :, 0].reshape(32, 74), ctx=q_ctx)

                    a1_nextstate_batch = nd.array(nextstate_batch1[:, :, 0].reshape(32, 74), ctx=q_ctx)
                    a1_signal_nextstate = nd.array(nextstate_batch2[:, :, 1].reshape(32, 74), ctx=q_ctx)
                    a2_nextstate_batch = nd.array(nextstate_batch2[:, :, 1].reshape(32, 74), ctx=q_ctx)
                    a2_signal_nextstate = nd.array(nextstate_batch2[:, :, 0].reshape(32, 74), ctx=q_ctx)

                    actions_batch1 = nd.array(actions1, ctx=q_ctx)
                    actions_batch2 = nd.array(actions2, ctx=q_ctx)

                    reward_batch1 = nd.array(rewards1, ctx=q_ctx)
                    reward_batch2 = nd.array(rewards2, ctx=q_ctx)

                    terminate_flags1 = nd.array(terminate_flags1, ctx=q_ctx)
                    terminate_flags2 = nd.array(terminate_flags2, ctx=q_ctx)

                    a1_signal = a2_comTarget.get_singals(a1_signal_nextstate)
                    a2_signal = a1_comTarget.get_singals(a2_signal_nextstate)

                    a1_Qvalue = a1_actTarget.get_qvals(a1_nextstate_batch, a1_signal)
                    a2_Qvalue = a2_actTarget.get_qvals(a2_nextstate_batch, a2_signal)

                    y_batch1 = reward_batch1 + nd.choose_element_0index(a1_Qvalue, nd.argmax_channel(a1_Qvalue)) * (
                        1.0 - terminate_flags1) * discount

                    y_batch2 = reward_batch2 + nd.choose_element_0index(a2_Qvalue, nd.argmax_channel(a2_Qvalue)) * (
                        1.0 - terminate_flags2) * discount

                    a1_signal = a2_comNet.get_signals(a1_signal_state)
                    a2_signal = a1_comNet.get_signals(a2_signal_state)

                    Qnet1 = a1_actQnet.update_params(a1_state_batch, a1_signal, actions_batch1, y_batch1)
                    Qnet2 = a2_actQnet.update_params(a2_state_batch, a2_signal, actions_batch2, y_batch2)

                    a1_comNet.update_params(a2_actQnet.exe.grad_dict['com'])
                    a2_comNet.update_params(a1_actQnet.exe.grad_dict['com'])

                    # sys.exit(0)
                    # print "before fc1_weight"
                    # print a2_comNet.exe.grad_dict['fc1_weight'][:1].asnumpy()
                    # print "com grad"
                    # print(a2_actQnet.exe.grad_dict['com'][:1].asnumpy())
                    #
                    # print "updated fc1_weight"
                    # print a2_comNet.exe.grad_dict['fc1_weight'][:1].asnumpy()

                    if training_steps % 10 == 0:
                        loss1 = 0.5 * nd.square(nd.choose_element_0index(Qnet1, actions_batch1) - y_batch1)
                        loss2 = 0.5 * nd.square(nd.choose_element_0index(Qnet2, actions_batch2) - y_batch2)
                        episode_loss += nd.sum(loss1).asnumpy()
                        episode_loss += nd.sum(loss2).asnumpy()
                        episode_update_step += 1

                    if training_steps % freeze_interval == 0:
                        copy_params_to(a1_actQnet, a1_actTarget)
                        copy_params_to(a1_comNet, a1_comTarget)
                        copy_params_to(a2_actQnet, a2_actTarget)
                        copy_params_to(a2_comNet, a2_comTarget)

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
        save_params(a1_actQnet, 'a1_actQnet/network-dqn_mx%04d.params' % epoch, q_ctx)
        save_params(a1_comNet, 'a1_comNet/network-dqn_mx%04d.params' % epoch, q_ctx)
        save_params(a2_actQnet, 'a2_actQnet/network-dqn_mx%04d.params' % epoch, q_ctx)
        save_params(a2_comNet, 'a2_comNet/network-dqn_mx%04d.params' % epoch, q_ctx)

        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))


if __name__ == '__main__':
    main()
