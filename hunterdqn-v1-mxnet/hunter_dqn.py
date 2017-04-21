from env.v1.hunterworld import HunterWorld
from base import Base
from env.ple import PLE
from operators import *
from utils import *
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = numpy.random


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1


# @profile
def main():
    replay_start_size = 50000
    replay_memory_size = 1000000
    history_length = 4

    rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": -0.02,
        "loss": -2.0,
        "win": 2.0
    }

    rows = 84
    cols = 84
    q_ctx = mx.gpu(0)

    game = HunterWorld(width=128, height=128,
                       num_preys=5,
                       num_hunters=2)

    env = PLE(game, fps=30, force_fps=True, display_screen=False,
              reward_values=rewards,
              resized_rows=rows, resized_cols=cols)
    env.init()

    replay_memory = ReplayMemory(state_dim=(84, 84),
                                 history_length=history_length,
                                 memory_size=replay_memory_size,
                                 replay_start_size=replay_start_size)

    freeze_interval = 10000
    epoch_num = 30
    steps_per_epoch = 100000
    update_interval = 4
    discount = 0.99

    action_set = env.get_action_set()
    action_map = []
    for action1 in action_set[0].values():
        for action2 in action_set[1].values():
            action_map.append([action1, action2])
    action_map = numpy.array(action_map)

    action_num = action_map.shape[0]

    eps_start = 1.0
    eps_min = 0.1
    eps_decay = (eps_start - eps_min) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32

    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}
    dqn_sym = dqn_sym_nature(action_num)
    qnet = Base(data_shapes=data_shapes, sym_gen=dqn_sym, name='QNet',
                initializer=DQNInitializer(factor_type="in"),
                ctx=q_ctx)
    target_qnet = qnet.copy(name="TargetQNet", ctx=q_ctx)

    optimizer = mx.optimizer.create(name="adagrad", learning_rate=0.01, eps=0.01,
                                    clip_gradient=None,
                                    rescale_grad=1.0, wd=0.0)

    updater = mx.optimizer.get_updater(optimizer)

    qnet.print_stat()
    target_qnet.print_stat()

    training_steps = 0
    total_steps = 0
    for epoch in range(epoch_num):
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
            time_episode_start = time.time()
            env.reset_game()
            while not env.game_over():
                if replay_memory.size >= history_length and replay_memory.size > replay_start_size:
                    do_exploration = (npy_rng.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action = npy_rng.randint(action_num)
                    else:
                        current_state = replay_memory.latest_slice()
                        state = nd.array(current_state.reshape((1,) + current_state.shape),
                                         ctx=q_ctx) / float(255.0)

                        qval_npy = qnet.forward(is_train=False, data=state)[0].asnumpy()
                        action = numpy.argmax(qval_npy[0])
                        episode_q_value += qval_npy[0, action]
                        episode_action_step += 1
                else:
                    action = npy_rng.randint(action_num)

                next_ob, reward, terminal_flag = env.act(action_map[action])
                replay_memory.append(next_ob, action, reward, terminal_flag)
                total_steps += 1
                episode_reward += reward
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory.size > replay_start_size:
                    training_steps += 1
                    episode_update_step += 1
                    states, actions, rewards, next_states, terminate_flags = replay_memory.sample(
                        batch_size=minibatch_size)
                    states = nd.array(states, ctx=q_ctx) / float(255.0)
                    next_states = nd.array(next_states, ctx=q_ctx) / float(255.0)
                    actions = nd.array(actions, ctx=q_ctx)
                    rewards = nd.array(rewards, ctx=q_ctx)
                    terminate_flags = nd.array(terminate_flags, ctx=q_ctx)

                    target_qval = target_qnet.forward(is_train=False, data=next_states)[0]
                    target_rewards = rewards + nd.choose_element_0index(target_qval,
                                                                        nd.argmax_channel(target_qval)) \
                                               * (1.0 - terminate_flags) * discount

                    outputs = qnet.forward(is_train=True,
                                           data=states,
                                           dqn_action=actions,
                                           dqn_reward=target_rewards)
                    qnet.backward()
                    qnet.update(updater=updater)

                    diff = nd.abs(nd.choose_element_0index(outputs[0], actions) - target_rewards)
                    quadratic_part = nd.clip(diff, -1, 1)
                    loss = 0.5 * nd.sum(nd.square(quadratic_part)).asnumpy()[0] + \
                           nd.sum(diff - quadratic_part).asnumpy()[0]
                    episode_loss += loss

                    if training_steps % freeze_interval == 0:
                        qnet.copy_params_to(target_qnet)

                        # if episode_action_step > 0:
                        #     print episode_q_value / episode_action_step

            steps_left -= env.episode_step
            time_episode_end = time.time()
            # Update the statistics
            epoch_reward += env.episode_reward
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                       % (epoch, episode, steps_left, env.episode_step, steps_per_epoch, env.episode_reward,
                          env.episode_step / (time_episode_end - time_episode_start), eps_curr)
            if episode_update_step > 0:
                info_str += ", Avg Loss:%f/%d" % (episode_loss / episode_update_step,
                                                  episode_update_step)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d" % (episode_q_value / episode_action_step,
                                                     episode_action_step)
            if episode % 100 == 0:
                logging.info(info_str)
        end = time.time()
        fps = steps_per_epoch / (end - start)
        qnet.save_params(dir_path="dqn-breakout-lr0.01", epoch=epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))


if __name__ == '__main__':
    main()
