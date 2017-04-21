import numpy
from env.ple_v6 import PLE
from env.v6.hunterworld import HunterWorld

rewards = {
    "positive": 1.0,
    "negative": -1.0,
    "tick": -0.002,
    "loss": -3.0,
    "win": 3.0
}

game = HunterWorld(width=256, height=256,
                   num_preys=5,
                   num_hunters=2)
env = PLE(game, fps=60, force_fps=True, display_screen=False,
          reward_values=rewards)

env.init()
actions = env.get_action_set()
# seed = numpy.random.RandomState(123456)

action_set = env.get_action_set()
action_map = []
for action1 in action_set[0].values():
    for action2 in action_set[1].values():
        action_map.append([action1, action2])
action_map = numpy.array(action_map)

for i in range(10000):

    if env.game_over():
        env.reset_game()

    # action_list = []
    # for action in actions:
    #     action_list.append(numpy.random.choice(actions[action].values()))    #

    action= numpy.random.randint(16)
    ob, reward, terminal_flag = env.act(action_map[action])

    # plt.imshow(frame, cmap='gray')
    # plt.show()

    # if i > 100:
    # env.force_fps = True
    # env.display_screen = True

    if reward[0] > 0:
        print "Score: {:0.3f} | Reward: {:0.3f} ".format(env.score(), reward[0])
