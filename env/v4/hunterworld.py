from env.pygamewrapper import PyGameWrapper
from env.v4.agent import Hunter, Prey, Toxin
from env.utils import *
from pygame.constants import K_w, K_a, K_s, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT
from env.vec2d import Vec2d

import mxnet as mx
import mxnet.ndarray as nd

COLOR_MAP = {"white": (255, 255, 255),
             "hunter": (20, 20, 255),
             "prey": (20, 255, 20),
             "toxin": (255, 20, 20),
             'black': (0, 0, 0)}


class HunterWorld(PyGameWrapper):
    def __init__(self,
                 width=48,
                 height=48,
                 num_preys=8,
                 num_hunters=2, num_toxins=5):

        self.actions = {0: {
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

        PyGameWrapper.__init__(self, width, height, actions=self.actions)

        self.BG_COLOR = COLOR_MAP['white']
        self.PREY_NUM = num_preys
        self.HUNTER_NUM = num_hunters
        self.TOXIN_NUM = num_toxins
        self.EYES = 24
        self.HUNTER_COLOR = COLOR_MAP['hunter']
        self.HUNTER_SPEED = width
        self.HUNTER_RADIUS = percent_round_int(width, 0.045)

        self.PREY_COLOR = COLOR_MAP['prey']
        self.PREY_SPEED = 0.25 * width
        self.PREY_RADIUS = percent_round_int(width, 0.035)

        self.TOXIN_COLOR = COLOR_MAP['toxin']
        self.TOXIN_SPEED = 0.25 * width
        self.TOXIN_RADIUS = percent_round_int(width, 0.020)

        self.hunters = pygame.sprite.Group()
        self.hunters_list = []
        self.preys = pygame.sprite.Group()
        self.preys_list = []
        self.toxins = pygame.sprite.Group()
        self.toxins_list = []

        self.agents = []
        # self.hungrey = 0
        self.lives = num_toxins
        self.previous_score = 0
        self.agents_pos = None
        self.preys_direct = None
        self.toxins_direct = None
        self.observation = []
        self.q_ctx = mx.gpu(0)

    def get_game_state(self):
        return np.array(self.observation).flatten()

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

    def fix_start(self):
        pos = []
        pos.append(vec2d((self.width * 0.45, self.height * 0.45)))
        pos.append(vec2d((self.width * 0.65, self.height * 0.65)))
        pos.append(vec2d((self.width * 0.25, self.height * 0.15)))
        pos.append(vec2d((self.width * 0.75, self.height * 0.85)))
        pos.append(vec2d((self.width * 0.65, self.height * 0.15)))
        pos.append(vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(vec2d((self.width * 0.75, self.height * 0.35)))
        return pos

    def _rand_direction(self, preys):
        dir = []
        for prey in preys:
            dir.append(vec2d((random() - 0.5, random() - 0.5)))
        return dir

    def get_score(self):
        return self.score

    def game_over(self):
        return len(self.preys) == 0 or self.lives <= 0

    def init(self):

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

        num = self.PREY_NUM - len(self.preys)
        if num > 0:
            for i in range(num):
                prey = Prey(
                    self.PREY_RADIUS,
                    self.PREY_COLOR,
                    self.PREY_SPEED,
                    self.width,
                    self.height
                )
                self.preys.add(prey)
                self.agents.append(prey)
                self.preys_list.append(prey)

        toxin_num = self.TOXIN_NUM - len(self.toxins)
        if toxin_num > 0:
            for i in range(toxin_num):
                toxin = Toxin(
                    self.TOXIN_RADIUS,
                    self.TOXIN_COLOR,
                    self.TOXIN_SPEED,
                    self.width,
                    self.height
                )
                self.toxins.add(toxin)
                self.agents.append(toxin)
                self.toxins_list.append(toxin)

        if self.agents_pos is None:
            # self.agents_pos = self._rand_start(self.agents)
            self.agents_pos = self._rand_start(self.agents)
            self.preys_direct = self._rand_direction(self.preys_list)
            self.toxins_direct = self._rand_direction(self.toxins_list)

        for i in range(len(self.agents)):
            self.agents[i].pos.x = self.agents_pos[i].x
            self.agents[i].pos.y = self.agents_pos[i].y

        for i in range(len(self.preys_list)):
            self.preys_list[i].direction.x = self.preys_direct[i].x
            self.preys_list[i].direction.y = self.preys_direct[i].y

        for i in range(len(self.toxins_list)):
            self.toxins_list[i].direction.x = self.toxins_direct[i].x
            self.toxins_list[i].direction.y = self.toxins_direct[i].y

        for i in range(len(self.hunters_list)):
            self.hunters_list[i].direction.x = 0
            self.hunters_list[i].direction.y = 0

        self.score = 0
        self.previous_score = 0
        self.lives = self.TOXIN_NUM
        # self.hungrey = 0

    def _handle_player_events(self, hunters):
        assert len(self.actions.keys()) == len(self.hunters)

        for hunter in hunters:
            hunter.dx = 0
            hunter.dy = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.actions.iteritems():

                    if key == actions["left"]:
                        hunters[idx].direction.x = -hunters[idx].speed
                        # hunters[idx].pos.x += 1 * 2
                        hunters[idx].direction.y = 0

                    if key == actions["right"]:
                        hunters[idx].direction.x = +hunters[idx].speed
                        hunters[idx].direction.y = 0

                    if key == actions["up"]:
                        hunters[idx].direction.y = -hunters[idx].speed
                        hunters[idx].direction.x = 0

                    if key == actions["down"]:
                        hunters[idx].direction.y = +hunters[idx].speed
                        hunters[idx].direction.x = 0

    # @profile
    def step(self, dt, frame_skip):

        if self.game_over():
            return 0.0

        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)
        # self.score += self.rewards["tick"]
        # self.hungrey += self.rewards["tick"]
        self._handle_player_events(self.hunters_list)

        for prey in self.preys:
            count = 0
            for hunter in self.hunters:
                if count_distant(prey, hunter) <= (hunter.out_radius - prey.radius):
                    count += 1
            if count >= 2:
                self.score += self.rewards["positive"]
                self.hungrey = self.rewards["positive"]
                self.preys.remove(prey)
                self.agents.remove(prey)
                self.preys_list.remove(prey)
                # pos_x = uniform(prey.radius, self.width - prey.radius)
                # pos_y = uniform(prey.radius, self.height - prey.radius)
                # prey.pos = vec2d((pos_x, pos_y))
                # dx = (random() - 0.5) * prey.speed * 4
                # dy = (random() - 0.5) * prey.speed * 4
                # prey.direction = vec2d((dx, dy))

        self.hunters.update(dt, self.hunters)
        self.preys.update(dt)
        self.toxins.update(dt)

        for index, hunter in enumerate(self.hunters):
            for toxin in self.toxins:
                if count_distant(toxin, hunter) <= (hunter.radius + toxin.radius):
                    self.lives -= 1
                    self.toxins_list.remove(toxin)
                    self.toxins.remove(toxin)
                    self.agents.remove(toxin)
                    self.score += self.rewards["negative"]

        if frame_skip is False:
            self.observation = []
            for index, hunter in enumerate(self.hunters):
                other_agents = []
                for agent in self.agents:
                    if agent is hunter:
                        continue
                    if count_distant(agent, hunter) <= agent.radius + hunter.out_radius:
                        other_agents.append(agent)
                self.observation.append(self.observe(hunter, other_agents))

        self.hunters.draw(self.screen)
        self.preys.draw(self.screen)
        self.toxins.draw(self.screen)

        # print self.lives
        reward = self.score - self.previous_score
        self.previous_score = self.score
        return reward

    # @profile
    def observe(self, hunter, others):
        center = Vec2d(hunter.rect.center)
        out_radius = hunter.out_radius
        radius = hunter.radius
        angle = 2 * np.pi / self.EYES
        observation = np.zeros((self.EYES, 3), dtype=np.float16)
        for i in range(1, self.EYES + 1):

            sin_angle = math.sin(angle * i)
            cos_angle = math.cos(angle * i)

            start = Vec2d(center.x + sin_angle * radius, center.y - cos_angle * radius).to_int()
            end = Vec2d((0, 0))
            color = COLOR_MAP["black"]
            end.x = center.x + int(sin_angle * out_radius)
            end.y = center.y - int(cos_angle * out_radius)

            for agent in others:
                dis = self.line_distance(center, end, Vec2d(agent.rect.center),
                                         agent.radius)
                if dis is not False and hunter.out_radius >= dis:
                    if dis >= hunter.radius:
                        end.x = center.x + int(sin_angle * dis)
                        end.y = center.y - int(cos_angle * dis)
                    elif dis <= hunter.radius:
                        end = False
                    if type(agent) is Prey:
                        color = COLOR_MAP['prey']
                        observation[i - 1] = [1.0 - dis / out_radius, 0, 0]
                    elif type(agent) is Hunter:
                        color = COLOR_MAP['hunter']
                        observation[i - 1] = [0, 1.0 - dis / out_radius, 0]
                    elif type(agent) is Toxin:
                        color = COLOR_MAP['toxin']
                        observation[i - 1] = [0, 0, 1.0 - dis / out_radius]
                    break

            if end is not False:
                pygame.draw.line(self.screen, color, start, end, 1)

        return observation

    # http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
    # @profile
    def closest_point_on_seg(self, seg_a, seg_b, circ_pos):
        seg_v = seg_b - seg_a
        pt_v = circ_pos - seg_a
        seg_v_unit = seg_v.normalized()
        proj = pt_v.dot(seg_v_unit)
        if proj <= 0:
            return False
        proj_v = seg_v_unit * proj
        closest = proj_v.to_int() + seg_a
        return closest

    # @profile
    def line_distance(self, seg_a, seg_b, circ_pos, circ_rad):
        closest = self.closest_point_on_seg(seg_a, seg_b, circ_pos)
        if closest is False:
            return False
        dist_v = circ_pos - closest
        offset = dist_v.length
        if offset >= circ_rad:
            return False
        le = math.sqrt(circ_rad ** 2 - offset ** 2)
        seg_v = seg_b - seg_a
        end = closest - (seg_v.normalized() * le).to_int()
        return (end - seg_a).length

    def nd_observe(self, hunter, others):
        center = nd.array(hunter.rect.center, ctx=self.q_ctx)
        out_radius = hunter.out_radius
        radius = hunter.radius
        angle = 2 * np.pi / self.EYES
        observation = nd.zeros((self.EYES, 3), ctx=self.q_ctx)
        for i in range(1, self.EYES + 1):

            sin_angle = math.sin(angle * i)
            cos_angle = math.cos(angle * i)

            start = [center[0] + sin_angle * radius, center[1] - cos_angle * radius]
            end = nd.zeros(2, ctx=self.q_ctx)
            color = COLOR_MAP["black"]
            end[0] = center[0] + int(sin_angle * out_radius)
            end[1] = center[1] - int(cos_angle * out_radius)

            for agent in others:
                dis = self.nd_line_distance(center, end, nd.array(agent.rect.center, ctx=self.q_ctx),
                                            agent.radius)
                if dis is not False:
                    dis = dis.asnumpy()[0]
                    if hunter.out_radius >= dis:
                        if dis >= hunter.radius:
                            end[0] = center[0] + sin_angle * dis
                            end[0] = center[1] - cos_angle * dis
                        elif dis <= hunter.radius:
                            end = False
                        if type(agent) is Prey:
                            color = COLOR_MAP['prey']
                            observation[i - 1] = nd.array([1.0 - dis / out_radius, 0, 0], ctx=self.q_ctx)
                        elif type(agent) is Hunter:
                            color = COLOR_MAP['hunter']
                            observation[i - 1] = nd.array([0, 1.0 - dis / out_radius, 0], ctx=self.q_ctx)
                        elif type(agent) is Toxin:
                            color = COLOR_MAP['toxin']
                            observation[i - 1] = nd.array([0, 0, 1.0 - dis / out_radius], ctx=self.q_ctx)
                        break

                if end is not False:
                    pygame.draw.line(self.screen, color, [start[0].asnumpy()[0], start[1].asnumpy()[0]], end.asnumpy(),
                                     1)

        return observation

    def nd_closest_point_on_seg(self, seg_a, seg_b, circ_pos):
        seg_v = seg_b - seg_a
        pt_v = circ_pos - seg_a
        seg_v_unit = seg_v / self.nd_lengh(seg_v)
        proj = nd.dot(pt_v, seg_v_unit)
        if proj.asnumpy()[0] <= 0:
            return False
        proj_v = seg_v_unit * proj
        closest = proj_v + seg_a
        return closest

    def nd_line_distance(self, seg_a, seg_b, circ_pos, circ_rad):
        closest = self.nd_closest_point_on_seg(seg_a, seg_b, circ_pos)
        if closest is False:
            return False
        dist_v = circ_pos - closest
        offset = self.nd_lengh(dist_v)
        if offset.asnumpy()[0] >= circ_rad:
            return False
        le = nd.sqrt(circ_rad ** 2 - offset ** 2)
        seg_v = seg_b - seg_a
        end = closest - seg_v / self.nd_lengh(seg_v) * le
        return self.nd_lengh(end - seg_a)

    def nd_lengh(self, ndarray):
        return nd.sqrt(ndarray[0] ** 2 + ndarray[1] ** 2)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = HunterWorld(width=1000, height=1000, num_preys=8, num_hunters=2)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.init()
        game.step(dt, False)
        # print game.lives
        pygame.display.update()
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
