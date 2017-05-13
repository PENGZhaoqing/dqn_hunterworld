from hunterworld import HunterWorld
from ple import PLE

rewards = {
    "positive": 1.0,
    "negative": -1.0,
    "tick": -0.002,
    "loss": -2.0,
    "win": 2.0
}


class Env:
    def __init__(self):
        self.game = HunterWorld(width=256, height=256, num_preys=10, draw=False,
                                num_hunters=1, num_toxins=5)
        self.ple = PLE(self.game, fps=30, force_fps=True, display_screen=False, reward_values=rewards,
                       resized_rows=80, resized_cols=80, num_steps=3)
        self.ob = []
        self.value = []
        self.reward = []
        self.action = []
        self.ob_np = []
        self.done = False

    def add(self, ob, ob_np, action, value, reward):
        self.ob.append(ob)
        self.ob_np.append(ob_np)
        self.value.append(value)
        self.reward.append(reward)
        self.action.append(action)

    def clear(self):
        self.ob = []
        self.value = []
        self.reward = []
        self.ob_np = []
        self.action = []
        self.done = False
