import pygame
from vec2d import *
from random import random, uniform

class Agent(pygame.sprite.Sprite):
    def __init__(self, radius, color, speed, screen_width, screen_height):
        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.speed = speed
        self.radius = radius
        self.dx = 0
        self.dy = 0
        self.direction = None
        self.image = None
        self.rect = None
        self.direction = None

    def set_pos(self, pos_init):
        self.pos = pos_init
        self.rect.center = (pos_init.x, pos_init.y)

    def update(self, dt, agents):
        pass

    def move(self, dt, other_agents):
        flag = True
        for i in range(0, len(other_agents)):
            if pygame.sprite.collide_circle(self, other_agents[i]):
                x = self.rect.center[0] - other_agents[i].rect.center[0]
                y = self.rect.center[1] - other_agents[i].rect.center[1]

                self.direction = vec2d((x, y))
                flag = False

        if flag:
            self.direction.x += self.dx
            self.direction.y += self.dy

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
        self.direction = vec2d((random() - 0.5, random() - 0.5))
        self.direction.normalize()

    def update(self, dt, agents):
        agents_tmp = [x for x in agents]
        agents_tmp.remove(self)
        self.dx = (random() - 0.5) * self.speed
        self.dy = (random() - 0.5) * self.speed
        self.move(dt, agents_tmp)


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
            2
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

    def update(self, dt, agents):
        agents_tmp = [x for x in agents]
        agents_tmp.remove(self)
        self.move(dt, agents_tmp)
