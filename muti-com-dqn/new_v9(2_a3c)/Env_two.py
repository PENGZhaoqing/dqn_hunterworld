from env.v11.hunterworld_v2 import HunterWorld
from env.ple_v7 import PLE

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
                                num_hunters=2, num_toxins=0)
        self.ple = PLE(self.game, fps=30, force_fps=True, display_screen=False, reward_values=rewards,
                       resized_rows=80, resized_cols=80, num_steps=3)
        self.ob1 = []
        self.ob2 = []
        self.value1 = []
        self.value2 = []
        self.reward1 = []
        self.reward2 = []
        self.action1 = []
        self.action2 = []
        self.done = False

    def add(self, ob1, ob2, action1, action2, value1, value2, reward1, reward2):
        self.ob1.append(ob1)
        self.ob2.append(ob2)
        self.value1.append(value1)
        self.value2.append(value2)
        self.reward1.append(reward1)
        self.reward2.append(reward2)
        self.action1.append(action1)
        self.action2.append(action2)

    def clear(self):
        self.ob1 = []
        self.ob2 = []
        self.value1 = []
        self.value2 = []
        self.reward1 = []
        self.reward2 = []
        self.action1 = []
        self.action2 = []

    def reset_var(self):
        self.clear()
        self.done = False
