from pygamewrapper import PyGameWrapper
from agent import Hunter, Prey, Toxin
from utils import *
from pygame.constants import *
import pygame
from numpy import random
from math import sqrt, sin, cos
import sys

COLOR_MAP = {"white": (255, 255, 255),
             "hunter": {0: (0, 0, 255),
                        1: (3, 168, 158),
                        2: (64, 244, 205)},
             "prey": (20, 255, 20),
             "toxin": (255, 20, 20),
             'black': (144, 144, 144)}


class HunterWorld(PyGameWrapper):
    def __init__(self, draw=False,
                 width=48,
                 height=48,
                 num_preys=10,
                 num_hunters=3, num_toxins=5):

        self.actions = {0: {"up": K_w, "left": K_a, "right": K_d, "down": K_s},
                        1: {"up": K_UP, "left": K_LEFT, "right": K_RIGHT, "down": K_DOWN},
                        2: {"up": K_t, "left": K_f, "right": K_h, "down": K_g},
                        3: {"up": K_i, "left": K_j, "right": K_l, "down": K_k}}

        PyGameWrapper.__init__(self, width, height, actions=self.actions)
        self.draw = draw
        self.BG_COLOR = COLOR_MAP['white']
        self.EYES = 24

        self.HUNTER_NUM = num_hunters
        self.HUNTER_COLOR = COLOR_MAP['hunter']
        self.HUNTER_SPEED = width * 0.25
        self.HUNTER_RADIUS = percent_round_int(width, 0.03)
        self.hunters = pygame.sprite.Group()
        self.hunters_list = []

        self.PREY_COLOR = COLOR_MAP['prey']
        self.PREY_SPEED = width * 0.25
        self.PREY_NUM = num_preys
        self.PREY_RADIUS = percent_round_int(width, 0.03)
        self.preys = pygame.sprite.Group()

        self.TOXIN_NUM = num_toxins
        self.TOXIN_COLOR = COLOR_MAP['toxin']
        self.TOXIN_SPEED = 0.25 * width
        self.TOXIN_RADIUS = percent_round_int(width, 0.03)
        self.toxins = pygame.sprite.Group()

        self.FIX_DIR = self._fix_direction()
        self.FIX_POS = self._fix_postion()

        self.AGENTS = []
        self.reward = np.zeros(self.HUNTER_NUM)

        self.agent_map = ["hunter"] * num_hunters
        self.agent_map.extend(["prey"] * num_preys)
        self.agent_map.extend(["toxin"] * num_toxins)

        self.init_flag = False
        self.observation = []
        self.hungrey = 0

    def _rand_postion(self, agents):
        pos = []
        for agent in agents:
            pos_x = random.uniform(agent.radius, self.width - agent.radius)
            pos_y = random.uniform(agent.radius, self.height - agent.radius)
            pos.append([pos_x, pos_y])

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = math.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                while dist <= (agents[i].radius + agents[j].radius):
                    pos[i][0] = random.uniform(agents[i].radius, self.width - agents[i].radius)
                    pos[i][1] = random.uniform(agents[i].radius, self.height - agents[i].radius)
                    dist = math.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
        return pos

    def _fix_postion(self):
        pos_list = [[0.45, 0.45], [0.65, 0.65], [0.25, 0.85],
                    [0.25, 0.15], [0.75, 0.85], [0.65, 0.15], [0.15, 0.85], [0.25, 0.35],
                    [0.95, 0.65], [0.75, 0.45], [0.75, 0.55], [0.65, 0.95], [0.05, 0.05],
                    [0.56, 0.84], [0.67, 0.42], [0.98, 0.12], [0.45, 0.26], [0.48, 0.15]]
        return [[pos_list[i][0] * self.width, pos_list[i][1] * self.height] for i in range(len(pos_list))]

    def _fix_direction(self):
        dir_list = [[-0.25, 0.15], [0.75, -0.85], [-0.65, 0.15], [-0.15, -0.85], [0.25, 0.35],
                    [0.95, -0.65], [0.75, 0.45], [-0.75, 0.55], [0.65, 0.95], [0.05, -0.05],
                    [0.56, 0.84], [0.67, 0.42], [0.98, 0.12], [0.45, 0.26], [0.48, 0.15]]
        return [normalization(dir_list[i]) for i in range(len(dir_list))]

    def get_score(self):
        return self.score

    def game_over(self):
        return False

    def init(self):
        assert self.init_flag == False, "Init Game Twice!!!"
        for id, kind in enumerate(self.agent_map):
            if kind is "hunter":
                hunter = Hunter(id, self.HUNTER_RADIUS, self.HUNTER_COLOR[id], self.HUNTER_SPEED, self.width,
                                self.height)
                self.hunters.add(hunter)
                self.AGENTS.append(hunter)
                self.hunters_list.append(hunter)
            elif kind is "prey":
                prey = Prey(id, self.PREY_RADIUS, self.PREY_COLOR, self.PREY_SPEED, self.width, self.height)
                self.preys.add(prey)
                self.AGENTS.append(prey)
            elif kind is "toxin":
                toxin = Toxin(id, self.TOXIN_RADIUS, self.TOXIN_COLOR, self.TOXIN_SPEED, self.width, self.height)
                self.toxins.add(toxin)
                self.AGENTS.append(toxin)

        for i in range(len(self.AGENTS)):
            self.AGENTS[i].init_positon(self.FIX_POS[i])

        count = 0
        for i in range(len(self.AGENTS)):
            if type(self.AGENTS[i]) == Hunter:
                self.AGENTS[i].init_direction((0, 0))
            else:
                self.AGENTS[i].init_direction(self.FIX_DIR[count])
                count += 1
        self.init_flag = True

    def reset(self):
        assert self.init_flag, "Init Game First"
        for agent in self.AGENTS:
            agent.reset_pos()
            agent.reset_orientation()
        self.hungrey = 0

    def _handle_player_events(self, hunters):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.actions.iteritems():

                    if key == actions["left"]:
                        hunters[idx].dx = -hunters[idx].speed
                        hunters[idx].dy = 0
                        hunters[idx].accelerate = True

                    if key == actions["right"]:
                        hunters[idx].dx = hunters[idx].speed
                        hunters[idx].dy = 0
                        hunters[idx].accelerate = True

                    if key == actions["up"]:
                        hunters[idx].dy = -hunters[idx].speed
                        hunters[idx].dx = 0
                        hunters[idx].accelerate = True

                    if key == actions["down"]:
                        hunters[idx].dy = hunters[idx].speed
                        hunters[idx].dx = 0
                        hunters[idx].accelerate = True

    # @profile
    def step(self, dt):

        self.reward[:] = 0.0
        self.hungrey += self.rewards["tick"]
        if self.game_over():
            return self.reward

        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)
        self._handle_player_events(self.hunters_list)

        for prey in self.preys:
            hunter_pair = []
            for hunter in self.hunters:
                if count_distant(prey, hunter) <= (hunter.range - prey.radius):
                    hunter_pair.append(hunter.id)
            if len(hunter_pair) >= 2:
                for hunter_id in hunter_pair:
                    self.reward[hunter_id] += self.rewards["positive"]
                self.hungrey += self.rewards["positive"]
                prey.rand_orientation()
                prey.rand_pos()

        for hunter in self.hunters:
            for toxin in self.toxins:
                if count_distant(toxin, hunter) < (hunter.radius + toxin.radius):
                    self.lives -= 1
                    toxin.rand_orientation()
                    toxin.rand_pos()
                    self.reward[hunter.id] += self.rewards["negative"]

        self.hunters.update(dt)
        self.preys.update(dt)
        self.toxins.update(dt)

        if self.draw:
            self.hunters.draw(self.screen)
            self.preys.draw(self.screen)
            self.toxins.draw(self.screen)
            self.get_game_state()
        return self.reward

    def get_game_state(self):
        self.observation[:] = []
        for i in range(len(self.hunters_list)):
            hunter = self.hunters_list[i]
            other_agents = []
            for j in range(len(self.AGENTS)):
                agent = self.AGENTS[j]
                if agent is hunter:
                    continue
                if count_distant(agent, hunter) <= agent.radius + hunter.out_radius:
                    other_agents.append(agent)
            ob = self.observe1(hunter, other_agents)
            if self.draw:
                self.draw_line(hunter, ob)
            state = np.append(ob, [hunter.velocity[0] / self.width, hunter.velocity[1] / self.height])
            self.observation.append(state)
        return self.observation

    def draw_line(self, hunter, observation):
        center = list(hunter.rect.center)
        radius = hunter.radius
        angle = 2 * np.pi / self.EYES
        observation = (1 - observation) * (hunter.out_radius - hunter.radius)
        for i in range(0, self.EYES):
            sin_angle = math.sin(angle * i)
            cos_angle = math.cos(angle * i)
            color = COLOR_MAP["black"]
            index = np.argmin(observation[i])
            line = observation[i][index]

            if line == hunter.out_radius - hunter.radius:
                color = COLOR_MAP["black"]
            elif index == 0:
                color = COLOR_MAP["prey"]
            elif index == 1:
                color = COLOR_MAP["hunter"][hunter.id]
            elif index == 2:
                color = COLOR_MAP["toxin"]

            if line > 0:
                start_pos = [center[0] + sin_angle * radius, center[1] - cos_angle * radius]
                end_pos = [0, 0]
                end_pos[0] = start_pos[0] + int(sin_angle * line)
                end_pos[1] = start_pos[1] - int(cos_angle * line)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def observe1(self, hunter, others):
        center = list(hunter.rect.center)
        out_radius = hunter.out_radius - hunter.radius
        observation = np.zeros((self.EYES, 3))
        angle = 2 * np.pi / self.EYES
        other_agents = others[:]
        for i in range(0, self.EYES):
            sin_angle = sin(angle * i)
            cos_angle = cos(angle * i)
            for agent in other_agents:
                dis = self.line_distance1(center, [sin_angle, -cos_angle], hunter.out_radius, list(agent.rect.center),
                                          agent.radius)
                if dis is not False:
                    dis = max(dis - hunter.radius, 0)
                    assert 0 <= dis <= out_radius, str(dis)
                    if type(agent) is Prey:
                        observation[i] = [1.0 - dis / out_radius, 0, 0]
                    elif type(agent) is Hunter:
                        observation[i] = [0, 1.0 - dis / out_radius, 0]
                    elif type(agent) is Toxin:
                        observation[i] = [0, 0, 1.0 - dis / out_radius]
                    break
        return observation

    # http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
    def line_distance1(self, seg_a, seg_v_unit, seg_v_len, circ_pos, circ_rad):
        pt_v = [circ_pos[0] - seg_a[0], circ_pos[1] - seg_a[1]]
        proj = pt_v[0] * seg_v_unit[0] + pt_v[1] * seg_v_unit[1]
        if proj <= 0 or proj >= seg_v_len:
            return False
        proj_v = [seg_v_unit[0] * proj, seg_v_unit[1] * proj]
        closest = [int(proj_v[0] + seg_a[0]), int(proj_v[1] + seg_a[1])]
        dist_v = [circ_pos[0] - closest[0], circ_pos[1] - closest[1]]
        offset = sqrt(dist_v[0] ** 2 + dist_v[1] ** 2)
        if offset >= circ_rad:
            return False
        le = sqrt(circ_rad ** 2 - int(offset) ** 2)
        re = [closest[0] - seg_a[0], closest[1] - seg_a[1]]
        # if sqrt(re[0] ** 2 + re[1] ** 2) - le < 0:
        #     a = 1
        #     print a

        return sqrt(re[0] ** 2 + re[1] ** 2) - le


if __name__ == "__main__":
    import numpy as np
    import time

    pygame.init()
    game = HunterWorld(width=258, height=258, num_preys=10, num_hunters=3, draw=True)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    game.reset()

    while True:
        start = time.time()
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.init()
        reward = game.step(dt)
        pygame.display.update()
        end = time.time()
        # print 1 / (end - start)
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
