#! /usr/bin/env python

import os
import random as rand
import pygame
import numpy as np

size=16

# Class for the orange dude
class Agent(object):
    def __init__(self,color):

        height=len(map.layout)
        width=len(map.layout[0])

        self.rect = pygame.Rect(rand.randint(1,width-2)*size, rand.randint(1,height-2)*size, size, size)
        self.speed = 2
        self.color = color

    def move(self, dx, dy):
        if dx != 0:
            self.move_single_axis(dx*self.speed, 0)
        if dy != 0:
            self.move_single_axis(0, dy*self.speed)

    def draw(self,screen):
        pygame.draw.rect(screen, self.color, self.rect)

    def move_single_axis(self, dx, dy):

        # Move the rect
        self.rect.x += dx
        self.rect.y += dy

        # If you collide with a wall, move out based on velocity
        for wall in map.walls:
            if self.rect.colliderect(wall):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    self.rect.right = wall.left
                if dx < 0:  # Moving left; Hit the right side of the wall
                    self.rect.left = wall.right
                if dy > 0:  # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.top
                if dy < 0:  # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.bottom


# Nice class to hold a wall rect
class Map(object):

    def __init__(self, layout):
        self.layout=layout
        self.walls=[]
        x = y = 0
        for row in layout:
            for col in row:
                if col == "W":
                    self.walls.append(pygame.Rect(x, y, size, size))
                x += size
            y += size
            x = 0

    def draw(self,screen):
        for wall in self.walls:
            pygame.draw.rect(screen, (0, 0, 0), wall)

# Holds the level layout in a list of strings.
layout = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "W                  W",
    "WWWWWWWWWWWWWWWWWWWW",
]

map=Map(layout)
agent1 = Agent((255,0,0))  # Create the player
agent2 = Agent((255,0,0))  # Create the player
prey = Agent((0,255,0))  # Create the player


# Initialise pygame
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()

# Set up the display
pygame.display.set_caption("Get to the red square!")
screen = pygame.display.set_mode((320, 240))

clock = pygame.time.Clock()
running = True

episode=1
max_episode=1000
max_step=200

while running and episode<max_episode:

    step=1
    while running and step<max_step:
        clock.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                break
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
                break

        # Move the player if an arrow key is pressed
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            agent1.move(-1, 0)
        if key[pygame.K_RIGHT]:
            agent1.move(1, 0)
        if key[pygame.K_UP]:
            agent1.move(0, -1)
        if key[pygame.K_DOWN]:
            agent1.move(0, 1)

        # Just added this to make it slightly fun ;)
        if agent1.rect.colliderect(prey.rect):
            raise SystemExit, "You win!"

        # Draw the scene
        screen.fill((255, 255, 255))
        map.draw(screen)
        agent1.draw(screen)
        agent2.draw(screen)
        prey.draw(screen)

        pygame.display.flip()
        # print "steps: " + str(step)
        step=step+1


    # print "episode: "+str(episode)
    episode=episode+1


