from env.pygamewrapper import PyGameWrapper
from env.v6.agent import Hunter, Prey, Toxin
from env.utils import *
from pygame.constants import *
from env.vec2d import Vec2d

COLOR_MAP = {"white": (255, 255, 255),
             "hunter": (40, 40, 140),
             "prey": (20, 140, 20),
             "toxin": (140, 40, 40),
             'black': (0, 0, 0)}


class HunterWorld(PyGameWrapper):
    def __init__(self,
                 width=48,
                 height=48,
                 num_preys=10,
                 num_hunters=4, num_toxins=5):

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
        }, 2: {
            "up": K_t,
            "left": K_f,
            "right": K_h,
            "down": K_g
        }, 3: {
            "up": K_i,
            "left": K_j,
            "right": K_l,
            "down": K_k
        }}

        PyGameWrapper.__init__(self, width, height, actions=self.actions)

        self.BG_COLOR = COLOR_MAP['white']
        self.HUNTER_NUM = num_hunters
        self.HUNTER_COLOR = COLOR_MAP['hunter']
        self.HUNTER_SPEED = width
        self.HUNTER_RADIUS = percent_round_int(width, 0.045)
        self.hunters = pygame.sprite.Group()
        self.HUNTERS = []

        self.PREY_COLOR = COLOR_MAP['prey']
        self.PREY_SPEED = 0.25 * width
        self.PREY_NUM = num_preys
        self.PREY_RADIUS = percent_round_int(width, 0.035)
        self.preys = pygame.sprite.Group()
        self.PREYS_D = self._rand_direction(num_preys)

        self.TOXIN_NUM = num_toxins
        self.TOXIN_COLOR = COLOR_MAP['toxin']
        self.TOXIN_SPEED = 0.25 * width
        self.TOXIN_RADIUS = percent_round_int(width, 0.020)
        self.toxins = pygame.sprite.Group()
        self.TOXINS_D = self._rand_direction(num_toxins)

        self.AGENTS = []
        self.reward = np.zeros(self.HUNTER_NUM)

        self.agent_map = ["hunter"] * num_hunters
        self.agent_map.extend(["prey"] * num_preys)
        self.agent_map.extend(["toxin"] * num_toxins)

        self.agent_deleted = []
        self.observation = []
        self.agents_pos = None
        self.init_flag = True

    def init(self):
        for ID, kind in enumerate(self.agent_map):
            if kind is "hunter":
                hunter = Hunter(
                    ID,
                    self.HUNTER_RADIUS,
                    self.HUNTER_COLOR,
                    self.HUNTER_SPEED,
                    self.width,
                    self.height
                )
                self.hunters.add(hunter)
                self.AGENTS.append(hunter)
                self.HUNTERS.append(hunter)
            elif kind is "prey":
                prey = Prey(
                    ID,
                    self.PREY_RADIUS,
                    self.PREY_COLOR,
                    self.PREY_SPEED,
                    self.width,
                    self.height
                )
                self.preys.add(prey)
                self.AGENTS.append(prey)
            elif kind is "toxin":
                toxin = Toxin(
                    ID,
                    self.TOXIN_RADIUS,
                    self.TOXIN_COLOR,
                    self.TOXIN_SPEED,
                    self.width,
                    self.height
                )
                self.toxins.add(toxin)
                self.AGENTS.append(toxin)

        agents_pos = self._rand_start(self.AGENTS)

        for idx, agent in enumerate(self.AGENTS):
            agent.init_positon(agents_pos[idx])

        for idx, agent in enumerate(self.preys):
            agent.init_direction(self.PREYS_D[idx])

        for idx, agent in enumerate(self.toxins):
            agent.init_direction(self.TOXINS_D[idx])

        for idx, agent in enumerate(self.hunters):
            agent.init_direction(Vec2d((0, 0)))

    def get_game_state(self):
        self.observation[:] = []
        for hunter in self.hunters:
            tmp = []
            for idx, agent in enumerate(self.AGENTS):
                tmp_pos = agent.pos
                tmp.append([tmp_pos.x / self.width, tmp_pos.y / self.height])
            for agent in self.agent_deleted:
                tmp[agent.id] = [0.0, 0.0]
            del tmp[hunter.id]
            self.observation.append(tmp)
        return np.array(self.observation)

    def _rand_start(self, agents):
        pos = []
        for agent in agents:
            pos_x = uniform(agent.radius, self.width - agent.radius)
            pos_y = uniform(agent.radius, self.height - agent.radius)
            pos.append(Vec2d((pos_x, pos_y)))

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = math.sqrt((pos[i].x - pos[j].x) ** 2 + (pos[i].y - pos[j].y) ** 2)
                while dist <= (agents[i].radius + agents[j].radius):
                    pos[i].x = uniform(agents[i].radius, self.width - agents[i].radius)
                    pos[i].y = uniform(agents[i].radius, self.height - agents[i].radius)
                    dist = math.sqrt((pos[i].x - pos[j].x) ** 2 + (pos[i].y - pos[j].y) ** 2)
        return pos

    def _rand_direction(self, num):
        dir = []
        for _ in range(num):
            dir.append(Vec2d((random() - 0.5, random() - 0.5)).normalized())
        return dir

    def fix_start(self):
        pos = []
        pos.append(Vec2d((self.width * 0.45, self.height * 0.45)))
        pos.append(Vec2d((self.width * 0.65, self.height * 0.65)))
        pos.append(Vec2d((self.width * 0.25, self.height * 0.15)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.65, self.height * 0.15)))
        pos.append(Vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.35)))
        pos.append(Vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.35)))
        pos.append(Vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.35)))
        pos.append(Vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.35)))
        pos.append(Vec2d((self.width * 0.15, self.height * 0.85)))
        pos.append(Vec2d((self.width * 0.75, self.height * 0.35)))
        return pos

    def get_score(self):
        return self.score

    def game_over(self):
        return len(self.preys) == 0 or self.lives <= 0

    def reset(self):

        if self.init_flag:
            self.init_flag = False
            self.init()

        for agent in self.agent_deleted:
            if type(agent) is Prey:
                self.preys.add(agent)
            elif type(agent) is Toxin:
                self.toxins.add(agent)

        self.agent_deleted[:] = []

        for agent in self.AGENTS:
            agent.reset_pos()
            agent.reset_orientation()

        self.reward[:] = 0.0
        self.lives = self.TOXIN_NUM

    def _handle_player_events(self, hunters):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                for idx, actions in self.actions.iteritems():

                    if key == actions["left"]:
                        hunters[idx].direction.x = -hunters[idx].speed
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
    def step(self, dt):

        self.reward[:] = 0.0

        if self.game_over():
            return self.reward

        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)
        self._handle_player_events(self.HUNTERS)

        for prey in self.preys:
            count = 0
            hunter_pair = []
            for hunter in self.hunters:
                if count_distant(prey, hunter) <= (hunter.out_radius - prey.radius):
                    count += 1
                    hunter_pair.append(hunter.id)
            if count >= 2:
                self.reward[hunter_pair[0]] += self.rewards["positive"]
                self.reward[hunter_pair[1]] += self.rewards["positive"]
                self.preys.remove(prey)
                self.agent_deleted.append(prey)

        for hunter in self.hunters:
            for toxin in self.toxins:
                if count_distant(toxin, hunter) <= (hunter.radius + toxin.radius):
                    self.lives -= 1
                    self.toxins.remove(toxin)
                    self.agent_deleted.append(toxin)

        for i in range(0, len(self.HUNTERS)):
            for j in range(i + 1, len(self.HUNTERS)):
                if pygame.sprite.collide_circle(self.HUNTERS[i], self.HUNTERS[j]):
                    offset = Vec2d(self.HUNTERS[i].rect.center) - Vec2d(self.HUNTERS[j].rect.center)
                    self.HUNTERS[i].direction = offset
                    self.HUNTERS[j].direction = - offset

        self.hunters.update(dt)
        self.preys.update(dt)
        self.toxins.update(dt)

        self.hunters.draw(self.screen)
        self.preys.draw(self.screen)
        self.toxins.draw(self.screen)

        return self.reward


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = HunterWorld(width=500, height=500, num_preys=8, num_hunters=2)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.reset()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()
        reward = game.step(dt)
        # state = game.get_game_state()
        pygame.display.update()
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
