from env.utils import *

class Agent(pygame.sprite.Sprite):
    def __init__(self, radius, color, speed, screen_width, screen_height):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.speed = speed
        self.radius = radius
        self.direction = None
        self.image = None
        self.rect = None
        self.pos = vec2d((0, 0))
        self.jitter_speed = random()

    def update(self, dt, agents):
        pass

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)

class Prey(Agent):
    def __init__(self, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, radius, color, speed, screen_width, screen_height)

        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))
        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )
        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.direction = vec2d((0, 0))
        # self.direction.normalize()

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = -1 * self.direction.x * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.direction.x = -1 * self.direction.x * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = -1 * self.direction.y * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
            self.direction.y = -1 * self.direction.y * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.direction.normalize()

        self.rect.center = ((self.pos.x, self.pos.y))


class Toxin(Agent):
    def __init__(self, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, radius, color, speed, screen_width, screen_height)
        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))
        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )
        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.direction = vec2d((random() - 0.5, random() - 0.5))
        # self.direction.normalize()

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = -1 * self.direction.x * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.direction.x = -1 * self.direction.x * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = -1 * self.direction.y * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
            self.direction.y = -1 * self.direction.y * \
                               (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.direction.normalize()

        self.rect.center = ((self.pos.x, self.pos.y))


class Hunter(Agent):
    def __init__(self, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, radius, color, speed, screen_width, screen_height)

        self.out_radius = radius * 4

        image = pygame.Surface([self.out_radius * 2, self.out_radius * 2])
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (150, 95, 95),
            (self.out_radius, self.out_radius),
            self.out_radius,
            percent_round_int(screen_width, 0.01)
        )

        image.set_alpha(int(255 * 0.75))

        pygame.draw.circle(
            image,
            color,
            (self.out_radius, self.out_radius),
            radius,
            0
        )

        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.direction = vec2d((0, 0))

    def update(self, dt, hunters):
        hunters_tmp = [x for x in hunters]
        hunters_tmp.remove(self)
        self.move(dt, hunters_tmp)

    def move(self, dt, other_hunters):
        # flag = True
        for i in range(0, len(other_hunters)):
            if pygame.sprite.collide_circle(self, other_hunters[i]):
                x = self.rect.center[0] - other_hunters[i].rect.center[0]
                y = self.rect.center[1] - other_hunters[i].rect.center[1]

                self.direction = vec2d((x, y))

        new_x = self.pos.x + self.direction.x * dt
        new_y = self.pos.y + self.direction.y * dt

        if new_x >= self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = 0
        elif new_x < self.radius:
            self.pos.x = self.radius
            self.direction.x = 0
        else:
            self.pos.x = new_x

        if new_y > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = 0

        elif new_y < self.radius:
            self.pos.y = self.radius
            self.direction.y = 0
        else:
            self.pos.y = new_y

        self.rect.center = (self.pos.x, self.pos.y)
