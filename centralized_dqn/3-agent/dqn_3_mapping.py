import numpy as np
from env.v10.hunterworld_v2 import HunterWorld
from env.ple_v5 import PLE
from utils import *
import matplotlib.pyplot as plt
from ANN_3_mapping import createQNetwork, copyTargetQNetwork
from replay_memory import ReplayMemory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# os.chdir("saved_network")
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
                       num_hunters=3, num_toxins=5)

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

    if not testing:
        logging_config(folder_name="log", name="centralized_network_a3")

    replay_memory = ReplayMemory(state_dim=(222,),
                                 history_length=history_length,
                                 memory_size=replay_memory_size,
                                 replay_start_size=replay_start_size, state_dtype='float32')

    eps_decay = (eps_start - eps_min) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32

    action_set = env.get_action_set()
    action_map = []
    for action1 in action_set[0].values():
        for action2 in action_set[1].values():
            for action3 in action_set[2].values():
                action_map.append([action1, action2, action3])
    action_map = np.array(action_map)
    action_num = action_map.shape[0]

    target1 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1)
    target32 = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32)
    Qnet = createQNetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32)

    if param_file is not None:
        Qnet.load_params(param_file)
    copyTargetQNetwork(Qnet, target1)
    copyTargetQNetwork(Qnet, target32)

    print_params(Qnet)
    reward_behavior = np.zeros(3)
    collision_behavior = np.zeros(3)
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
            reward_behavior[:] = 0
            collision_behavior[:] = 0
            time_episode_start = time.time()
            env.reset_game()
            while not env.game_over():
                if replay_memory.size >= history_length and replay_memory.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action = np.random.randint(action_num)
                    else:
                        current_state = replay_memory.latest_slice()
                        state = nd.array(current_state.reshape((1,) + current_state.shape),
                                         ctx=q_ctx)
                        target1.forward(mx.io.DataBatch([state], []))
                        q_value = target1.get_outputs()[0].asnumpy()[0]
                        action = numpy.argmax(q_value)
                        episode_q_value += q_value[action]
                        episode_action_step += 1
                else:
                    action = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act(action_map[action])

                sum_reward = sum(reward)
                if sum_reward < 0.0:
                    collision_behavior += reward
                elif sum_reward > 0:
                    if reward[0] > 0 and reward[1] > 0:
                        reward_behavior[0] += 1
                    elif reward[1] > 0 and reward[2] > 0:
                        reward_behavior[1] += 1
                    elif reward[0] > 0 and reward[2] > 0:
                        reward_behavior[2] += 1

                replay_memory.append(np.array(next_ob).flatten(), action, sum_reward, terminal_flag)

                # if reward > 0:
                # plt.imshow(next_ob, cmap='gray')
                # plt.show()

                total_steps += 1
                episode_reward += sum_reward
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

                    y_batch = reward_batch + nd.choose_element_0index(Qvalue, nd.argmax_channel(Qvalue)) * (
                        1.0 - terminate_flags) * discount

                    Qnet.forward(mx.io.DataBatch([state_batch, actions_batch,y_batch],
                                                 []), is_train=True)
                    Qnet.backward()
                    Qnet.update()

                    if training_steps % 10 == 0:
                        loss1 = 0.5 * nd.square(
                            nd.choose_element_0index(Qnet.get_outputs()[0], actions_batch) - y_batch)
                        episode_loss += nd.sum(loss1).asnumpy()
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

            info_str += ", Collision:" + str(collision_behavior) + ", Reward Behavior:" + str(reward_behavior)

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
