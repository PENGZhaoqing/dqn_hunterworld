from env.utils import *
from env.vec2d import Vec2d


class Agent(pygame.sprite.Sprite):
    def __init__(self, id, radius, speed, screen_width, screen_height):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.speed = speed
        self.radius = radius
        self.direction = None
        self.image = None
        self.rect = None
        self.pos = Vec2d((0, 0))
        self.id = id
        self.init_pos = None
        self.init_dir = None

    def init_positon(self, pos):
        self.init_pos = Vec2d((pos.x, pos.y))

    def init_direction(self, direction):
        self.init_dir = Vec2d((direction.x, direction.y))

    def reset_pos(self):
        self.pos.x = self.init_pos.x
        self.pos.y = self.init_pos.y

    def reset_orientation(self):
        self.direction.x = self.init_dir.x
        self.direction.y = self.init_dir.y

    def update(self, dt):
        pass

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Prey(Agent):
    def __init__(self, id, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, id, radius, speed, screen_width, screen_height)

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
        self.direction = Vec2d((0, 0))

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = - self.direction.x  # a little jitter
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.direction.x = - self.direction.x  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = - self.direction.y
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
            self.direction.y = - self.direction.y  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.rect.center = ((self.pos.x, self.pos.y))


class Toxin(Agent):
    def __init__(self, id, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, id, radius, speed, screen_width, screen_height)
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
        self.direction = Vec2d((0, 0))

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = - self.direction.x  # a little jitter
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.direction.x = - self.direction.x  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = - self.direction.y
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
            self.direction.y = - self.direction.y  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.rect.center = ((self.pos.x, self.pos.y))


class Hunter(Agent):
    def __init__(self, id, radius, color, speed, screen_width, screen_height):
        Agent.__init__(self, id, radius, speed, screen_width, screen_height)

        self.out_radius = radius * 4

        image = pygame.Surface([self.out_radius * 2, self.out_radius * 2])
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (1, 1, 1),
            (self.out_radius, self.out_radius),
            self.out_radius,
            percent_round_int(screen_width, 0.008)
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
        self.direction = Vec2d((0, 0))

    def update(self, dt):
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
