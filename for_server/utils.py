import numpy as np
import math
import sys
from random import uniform, random
import pygame
from math import sqrt


class vec2d():
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def __add__(self, o):
        x = self.x + o.x
        y = self.y + o.y

        return vec2d((x, y))

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

    def normalize(self):
        norm = math.sqrt(self.x * self.x + self.y * self.y)
        self.x /= norm
        self.y /= norm


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


def count_distant(agent1, agent2):
    try:
        dis = math.sqrt((agent1.pos[0] - agent2.pos[0]) ** 2 + (agent1.pos[1] - agent2.pos[1]) ** 2)
    except (AttributeError, TypeError):
        raise AssertionError('Object should extend from Agent Class')
    return dis

def normalization(vector):
    norm = sqrt(vector[0] ** 2 + vector[1] ** 2)
    vector[0] /= norm
    vector[1] /= norm
    return vector
