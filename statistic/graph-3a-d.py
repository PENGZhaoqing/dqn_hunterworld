from io import open
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
import math
import os
import re


# print os.path.abspath(os.curdir)

class NetworkGraph(object):
    def __init__(self, file_name):
        self.steps = []
        self.score = []
        self.collision = []
        self.avg_loss = []
        self.avg_Q_value = []
        self.filename = file_name
        self.x_axis = None
        self.rewards_behavior = []
        self.mean_Qvalue = None
        self.mean_score = None
        self.mean_steps = None
        self.mean_collision = None
        self.mean_rewards = None
        self.mean_loss = None

    def load(self):
        with open(self.filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                columns = line.split(',')
                if len(columns) == 11:
                    self.steps.append(columns[3].split('/')[1])
                    self.score.append(columns[4].split(':')[1])
                    tmp = re.findall('\d+', columns[7].split(':')[1])
                    self.collision.append(map(float, tmp))
                    tmp = re.findall('\d+', columns[8].split(':')[1])
                    self.rewards_behavior.append(map(float, tmp))
                    self.avg_loss.append(columns[9].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[10].split(':')[1].split('/')[0])

        self.x_axis = np.array(range(len(self.score))) * 5
        self.steps = np.array(self.steps, dtype=int)
        self.score = np.array(self.score, dtype=float)
        self.rewards_behavior = np.array(self.rewards_behavior, dtype=float)
        self.collision = np.array(self.collision, dtype=float)
        self.avg_loss = np.array(self.avg_loss, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)


def draw(figure_num, network):
    fig = plt.figure(figure_num, figsize=(20, 10))
    fig.canvas.set_window_title(network.filename)
    plt.subplot(321)
    # plt.scatter(network.x_axis, network.score, s=4, c='g')
    # plt.plot(network.x_axis, network.score, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.score, 'b')
    plt.ylabel('scores')
    plt.grid(True)

    plt.subplot(322)
    # plt.plot(network.x_axis, network.score, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.score / network.steps, 'b')
    plt.ylabel('avg scores')
    plt.grid(True)

    plt.subplot(323)
    # plt.plot(network.x_axis, network.avg_loss, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.avg_loss, 'b')
    plt.ylabel('avg loss')
    plt.grid(True)

    plt.subplot(324)
    # plt.plot(network.x_axis, network.avg_Q_value, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.avg_Q_value, 'b')
    plt.ylabel('avg Q values')
    plt.grid(True)
    # fig.tight_layout()

    plt.subplot(325)
    # plt.plot(network.x_axis, network.collision, 'bo', markersize=4, mfc='b')
    # plt.plot(network.x_axis, network.collision[:, 1], 'b')/network.steps
    # plt.plot(network.x_axis, network.collision[:, 2], 'g')
    # plt.plot(network.x_axis, network.collision[:, 0], 'r')
    plt.plot(network.x_axis, np.sum(network.collision, axis=1) / network.steps, 'k')
    plt.ylabel('collisions')
    plt.grid(True)

    plt.subplot(326)
    # plt.plot(network.x_axis, network.collision, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.rewards_behavior[:, 0] / network.steps, 'r')
    plt.plot(network.x_axis, network.rewards_behavior[:, 1] / network.steps, 'b')
    plt.plot(network.x_axis, network.rewards_behavior[:, 2] / network.steps, 'g')
    plt.ylabel('cooperations')
    plt.grid(True)


    # avg_Q_value=np.array(avg_Q_value)
    # avg_Q_value = avg_Q_value.reshape(-1, 3).mean(axis=1)
    # x_range = len(avg_Q_value)
    # x_new = np.linspace(1, x_range, 100)
    # y_smooth = spline(range(x_range), avg_Q_value, x_new)
    # plt.plot(x_new, y_smooth)

    # plt.setp(lines, color='b', linewidth=1.0)
    # plt.title(r'$\sigma_i=15$')


no_com_network = NetworkGraph("3-agent/non_com_network_a3_04:16:12:39_adam-0.002.log")
no_com_network.load()
com_network = NetworkGraph("3-agent/com_network_a3_04:15:23:19_adam-0.002.log")
com_network.load()
cen_network = NetworkGraph("3-agent/centralized_network_a3_04:16:14:_adam-0.002.log")
cen_network.load()

draw(1, no_com_network)
draw(2, com_network)
draw(3, cen_network)
plt.show()
