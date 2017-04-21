import mxnet as mx
import numpy as np

# random seed for reproduction
SEED = 12345



def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))


def rollout(env, agent, max_path_length=np.inf):
    reward = []
    o = env.reset()
    # agent.reset()
    path_length = 0
    while path_length < max_path_length:
        o = o.reshape((1, -1))
        a = agent.get_action(o)
        next_o, r, d, _ = env.step(a)
        reward.append(r)
        path_length += 1
        if d:
            break
        o = next_o

    return reward


def sample_rewards(env, policy, eval_samples, max_path_length=np.inf):
    rewards = []
    for _ in range(eval_samples):
        rewards.append(rollout(env, policy, max_path_length))

    return rewards
