from env.pygamewrapper import PyGameWrapper
from env.v3.agent import Hunter, Prey, Toxin
from env.utils import *
from pygame.constants import K_w, K_a, K_s, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT


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
        self.BG_COLOR = (255, 255, 255)
        self.PREY_NUM = num_preys
        self.HUNTER_NUM = num_hunters
        self.TOXIN_NUM = num_toxins

        self.HUNTER_COLOR = (60, 60, 140)
        self.HUNTER_SPEED = width
        self.HUNTER_RADIUS = percent_round_int(width, 0.045)

        self.PREY_COLOR = (40, 140, 40)
        self.PREY_SPEED = 0.25 * width
        self.PREY_RADIUS = percent_round_int(width, 0.035)

        self.TOXIN_COLOR = (140, 40, 40)
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
                    np.round(self.HUNTER_RADIUS * (1 + i * 0.2)).astype(np.uint8),
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

    # @profile
    def step(self, dt):
        """
            Perform one step of v3-v0.01 emulation.
        """
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
                if count_distant(prey, hunter) < (hunter.out_radius - prey.radius):
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

        for hunter in self.hunters:
            for toxin in self.toxins:
                if count_distant(toxin, hunter) <= (hunter.radius + toxin.radius):
                    self.lives -= 1
                    self.toxins_list.remove(toxin)
                    self.toxins.remove(toxin)
                    self.agents.remove(toxin)
                    # self.score += self.rewards["negative"]

        self.hunters.update(dt, self.hunters)
        self.preys.update(dt)
        self.toxins.update(dt)

        self.hunters.draw(self.screen)
        self.preys.draw(self.screen)
        self.toxins.draw(self.screen)

        # print self.lives
        reward = self.score - self.previous_score
        self.previous_score = self.score
        return reward


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = HunterWorld(width=512, height=512, num_preys=8, num_hunters=2)
    game.screen = pygame.display.set_mode(game.get_screen_dims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.init()
        game.step(dt)
        print game.lives
        pygame.display.update()
        # if v3-v0.01.getScore() > 0:
        # print "Score: {:0.3f} ".format(v3-v0.01.getScore())
