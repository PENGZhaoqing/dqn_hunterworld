class HunterGame():

    def start(self):
        self.ale.reset_game()
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.max_episode_step = DEFAULT_MAX_EPISODE_STEP
        self.start_lives = self.ale.lives()

    def force_restart(self):
        self.start()
        self.replay_memory.clear()


    def begin_episode(self, max_episode_step=DEFAULT_MAX_EPISODE_STEP):
        """
            Begin an episode of a game instance. We can play the game for a maximum of
            `max_episode_step` and after that, we are forced to restart
        """
        if self.episode_step > self.max_episode_step or self.ale.game_over():
            self.start()
        else:
            for i in range(self.screen_buffer_length):
                self.ale.act(0)
                self.ale.getScreenGrayscale(self.screen_buffer[i % self.screen_buffer_length, :, :])
        self.max_episode_step = max_episode_step
        self.start_lives = self.ale.lives()
        self.episode_reward = 0
        self.episode_step = 0

    @property
    def episode_terminate(self):
        termination_flag = self.ale.game_over() or self.episode_step >= self.max_episode_step
        return termination_flag

    def play(self, a):
        assert not self.episode_terminate,\
            "Warning, the episode seems to have terminated. " \
            "We need to call either game.begin_episode(max_episode_step) to continue a new " \
            "episode or game.start() to force restart."
        self.episode_step += 1
        reward = 0.0
        action = self.action_set[int(a)]
        for i in range(self.frame_skip):
            reward += self.ale.act(action)
            self.ale.getScreenGrayscale(self.screen_buffer[i % self.screen_buffer_length, :, :])
        self.total_reward += reward
        self.episode_reward += reward
        ob = self.get_observation()
        # plt.imshow(ob, cmap="gray")
        # plt.show()

        terminate_flag = self.episode_terminate

        self.replay_memory.append(ob, a, numpy.clip(reward, -1, 1), terminate_flag)
        return reward, terminate_flag
