import mxnet as mx
import numpy


class ReplayMemory(object):
    def __init__(self, history_length, memory_size=1000000, replay_start_size=100,
                 state_dim=(), action_dim=(), state_dtype='uint8', action_dtype='uint8',
                 ctx=mx.gpu(), reward_dim=(), signal_dim=()):
        self.ctx = ctx
        assert type(action_dim) is tuple and type(state_dim) is tuple, \
            "Must set the dimensions of state and action for replay memory"
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.signal_dim = signal_dim
        if action_dim == (1,):
            self.action_dim = ()
        else:
            self.action_dim = action_dim
        self.states = numpy.zeros((memory_size,) + state_dim, dtype=state_dtype)
        self.actions = numpy.zeros((memory_size,) + action_dim, dtype=action_dtype)
        self.rewards = numpy.zeros((memory_size,) + reward_dim, dtype='float32')
        self.signals = numpy.zeros((memory_size,) + signal_dim, dtype='float32')
        self.terminate_flags = numpy.zeros(memory_size, dtype='bool')
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.history_length = history_length
        self.top = 0
        self.size = 0

    def latest_slice(self):
        if self.size >= self.history_length:
            return self.states.take(numpy.arange(self.top - self.history_length, self.top),
                                    axis=0, mode="wrap")
        else:
            assert False, "We can only slice from the replay memory if the " \
                          "replay size is larger than the length of frames we want to take" \
                          "as the input."

    def clear(self):
        """
        Clear all contents in the relay memory
        """
        self.states[:] = 0
        self.actions[:] = 0
        self.rewards[:] = 0
        self.terminate_flags[:] = 0
        self.top = 0
        self.size = 0

    def reset(self):
        """
        Reset all the flags stored in the replay memory.
        It will not clear the inner-content and is a light/quick version of clear()
        """
        self.top = 0
        self.size = 0

    def append(self, obs, action, reward, terminate_flag):
        self.states[self.top] = obs
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminate_flags[self.top] = terminate_flag
        self.top = (self.top + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1

    def sample(self, batch_size):
        assert self.size >= batch_size and self.replay_start_size >= self.history_length
        assert (0 <= self.size <= self.memory_size)
        assert (0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be bigger than "
                             "start_size! Currently, size=%d, start_size=%d"
                             % (self.size, self.replay_start_size))
        # TODO Possibly states + inds for less memory access
        states = numpy.zeros((batch_size, self.history_length) + self.state_dim,
                             dtype=self.states.dtype)
        actions = numpy.zeros((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.zeros((batch_size,) + self.reward_dim, dtype='float16')
        terminate_flags = numpy.zeros(batch_size, dtype='bool')
        next_states = numpy.zeros((batch_size, self.history_length) + self.state_dim,
                                  dtype=self.states.dtype)
        counter = 0
        while counter < batch_size:
            index = numpy.random.randint(low=self.top - self.size + 1, high=self.top - self.history_length)
            transition_indices = numpy.arange(index, index + self.history_length)
            initial_indices = transition_indices - 1
            end_index = index + self.history_length - 1
            while numpy.any(self.terminate_flags.take(initial_indices, mode='wrap')):
                # Check if terminates in the middle of the sample!
                index -= 1
                transition_indices = numpy.arange(index, index + self.history_length)
                initial_indices = transition_indices - 1
                end_index = index + self.history_length - 1
            states[counter] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, axis=0, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            # if rewards[counter] > 0:
            #     print(rewards[counter])
            # print numpy.clip(reward, -1, 1)
            # plt.imshow(ob, cmap='gray')
            # plt.show()
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            next_states[counter] = self.states.take(transition_indices, axis=0, mode='wrap')
            counter += 1
        return states, actions, rewards, next_states, terminate_flags

    def append_signal(self, obs, signal, action, reward, terminate_flag):
        self.states[self.top] = obs
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.signals[self.top] = signal
        self.terminate_flags[self.top] = terminate_flag
        self.top = (self.top + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1

    def sample_one(self, batch_size):
        assert self.replay_start_size >= self.history_length
        assert (0 <= self.size <= self.memory_size)
        assert (0 <= self.top <= self.memory_size)
        states = numpy.zeros((batch_size,) + self.state_dim,
                             dtype=self.states.dtype)
        signals = numpy.zeros((batch_size,) + self.signal_dim,
                              dtype=self.signals.dtype)
        actions = numpy.zeros((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.zeros((batch_size,) + self.reward_dim, dtype='float16')
        terminate_flags = numpy.zeros(batch_size, dtype='bool')
        next_states = numpy.zeros((batch_size,) + self.state_dim,
                                  dtype=self.states.dtype)
        next_signals = numpy.zeros((batch_size,) + self.signal_dim,
                                   dtype=self.signals.dtype)
        counter = 0
        while counter < batch_size:
            index = numpy.random.randint(low=self.top - self.size + 1, high=self.top - self.history_length)
            states[counter] = self.states[index]
            signals[counter] = self.signals[index]
            actions[counter] = self.actions[index]
            rewards[counter] = self.rewards[index]
            terminate_flags[counter] = self.terminate_flags[index]
            next_states[counter] = self.states[index + 1]
            next_signals[counter] = self.signals[index + 1]
            counter += 1
        return states, signals, actions, rewards, next_states, next_signals, terminate_flags
