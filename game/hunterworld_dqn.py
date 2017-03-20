from random import uniform

import pygame
from agent import Hunter, Prey
from pygamewrapper import PyGameWrapper
from pygame.constants import K_w, K_a, K_s, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT

from game.vec2d import *
from replay_memory import ReplayMemory
from utils import *

logger = logging.getLogger(__name__)

class HunterWorld(PyGameWrapper):
    def __init__(self,
                 width=48,
                 height=48,
                 num_preys=1,
                 num_hunters=2,
                 replay_memory_size=1000000,
                 replay_start_size=10000,
                 history_length=4):

        # self.replay_memory = ReplayMemory(state_dim=(resized_rows, resized_cols),
        #                                   history_length=history_length,
        #                                   memory_size=replay_memory_size,
        #                                   replay_start_size=replay_start_size)

        multi_actions = {0: {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }, 1: {
            "up": K_UP,
            "left": K_LEFT,
            "right": K_RIGHT,
            "down": K_DOWN
        }}

        PyGameWrapper.__init__(self, width, height, actions=multi_actions)
        self.BG_COLOR = (255, 255, 255)
        self.PREY_NUM = num_preys
        self.HUNTER_NUM = num_hunters

        self.HUNTER_COLOR = (60, 60, 140)
        self.HUNTER_SPEED = 0.25 * width
        self.HUNTER_RADIUS = percent_round_int(width, 0.07)

        self.PREY_COLOR = (40, 140, 40)
        self.PREY_SPEED = 0.25 * width
        self.PREY_RADIUS = percent_round_int(width, 0.047)

        self.hunters = pygame.sprite.Group()
        self.hunters_list = []
        self.preys = pygame.sprite.Group()
        self.agents = []

    def state_enabled(self):
        return self.replay_memory.size >= self.replay_memory.history_length

    def getGameState(self):
        """
        Returns
        -------

        dict
            * player x position.
            * player y position.
            * player x velocity.
            * player y velocity.
            * player distance to each creep

        """
        state = {}
        for index, agent in enumerate(self.agents):
            state[index] = {"x": agent.pos.x,
                            "y": agent.pos.y}

        return state

    def _rand_start(self, agents):
        pos = []
        for agent in agents:
            pos_x = uniform(agent.radius, self.width - agent.radius)
            pos_y = uniform(agent.radius, self.height - agent.radius)
            pos.append(vec2d((pos_x, pos_y)))

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = math.sqrt((pos[i].x - pos[j].x) ** 2 + (pos[i].y - pos[j].y) ** 2)
                while dist <= (agents[i].radius + agents[j].radius):
                    pos[i].x = uniform(agents[i].radius, self.width - agents[i].radius)
                    pos[i].y = uniform(agents[i].radius, self.height - agents[i].radius)
                    dist = math.sqrt((pos[i].x - pos[j].x) ** 2 + (pos[i].y - pos[j].y) ** 2)
        return pos

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return len(self.preys) == 0

    def init(self):

        """
            Starts/Resets the game to its inital state
        """

        if len(self.hunters) == 0:
            for i in range(self.HUNTER_NUM):
                hunter = Hunter(
                    self.HUNTER_RADIUS,
                    self.HUNTER_COLOR,
                    self.HUNTER_SPEED,
                    self.width,
                    self.height
                )
                self.hunters.add(hunter)
                self.agents.append(hunter)
                self.hunters_list.append(hunter)

        if len(self.preys) == 0:
            for i in range(self.PREY_NUM):
                prey = Prey(
                    self.PREY_RADIUS,
                    self.PREY_COLOR,
                    self.PREY_SPEED,
                    self.width,
                    self.height
                )
                self.preys.add(prey)
                self.agents.append(prey)

        pos = self._rand_start(self.agents)

        for i in range(len(self.agents)):
            self.agents[i].set_pos(pos[i])

        self.score = 0
        self.lives = -1

    def _handle_player_events(self, hunters):

        assert len(self.multi_actions.keys()) == len(self.hunters)

        for hunter in hunters:
            hunter.dx = 0
            hunter.dy = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.multi_actions.iteritems():

                    if key == actions["left"]:
                        hunters[idx].dx -= hunters[idx].speed

                    if key == actions["right"]:
                        hunters[idx].dx += hunters[idx].speed

                    if key == actions["up"]:
                        hunters[idx].dy -= hunters[idx].speed

                    if key == actions["down"]:
                        hunters[idx].dy = + hunters[idx].speed

                        # for pressed_key, action in self.multi_actions.iteritems():
                        # if key == pressed_key:
                        # if action.keys() == ["left"] :
                        #     hunters[action["left"]].dx -= hunters[action["left"]].speed
                        #
                        # if action.keys() == ["right"]:
                        #     hunters[action["right"]].dx += hunters[action["right"]].speed
                        #
                        # if action.keys() == ["up"]:
                        #     hunters[action["up"]].dy -= hunters[action["up"]].speed
                        #
                        # if action.keys() == ["down"]:
                        #     hunters[action["down"]].dy += hunters[action["down"]].speed

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)
        self.score += self.rewards["tick"]

        self._handle_player_events(self.hunters_list)

        for prey in self.preys:
            count = 0
            for hunter in self.hunters:
                if count_distant(prey, hunter) < (hunter.out_radius - prey.radius):
                    count += 1
            if count >= 2:
                self.score += self.rewards["positive"]
                self.preys.remove(prey)
                self.agents.remove(prey)

        self.hunters.update(dt, self.agents)
        self.preys.update(dt, self.agents)

        if len(self.preys) == 0:
            self.score += self.rewards["win"]

        self.hunters.draw(self.screen)
        self.preys.draw(self.screen)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = HunterWorld(width=256, height=256, num_preys=5, num_hunters=2)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.init()
        game.step(dt)
        pygame.display.update()
        if game.getScore() > 0:
            print "Score: {:0.3f} ".format(game.getScore())
