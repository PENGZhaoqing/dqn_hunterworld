from io import open
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
import math
import os
from scipy import ndimage
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
        self.rewards = []
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
                    self.rewards.append(map(float, tmp))
                    self.avg_loss.append(columns[9].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[10].split(':')[1].split('/')[0])

        self.x_axis = np.array(range(len(self.score))) * 5
        self.steps = np.array(self.steps, dtype=int)
        self.score = np.array(self.score, dtype=float)
        self.rewards = np.array(self.rewards, dtype=float)
        self.collision = np.array(self.collision, dtype=float)
        self.avg_loss = np.array(self.avg_loss, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)

    def smooth(self, mean_num=3):
        self.mean_steps = self.steps.reshape(-1, mean_num).mean(axis=1)
        self.mean_score = self.score.reshape(-1, mean_num).mean(axis=1)

        self.mean_collision = np.zeros((self.collision.shape[0] / mean_num, mean_num))
        self.mean_collision[:, 0] = self.collision[:, 0].reshape(-1, mean_num).mean(axis=1)
        self.mean_collision[:, 1] = self.collision[:, 1].reshape(-1, mean_num).mean(axis=1)
        self.mean_collision[:, 2] = self.collision[:, 2].reshape(-1, mean_num).mean(axis=1)

        self.mean_rewards = np.zeros((self.rewards.shape[0] / mean_num, mean_num))
        self.mean_rewards[:, 0] = self.rewards[:, 0].reshape(-1, mean_num).mean(axis=1)
        self.mean_rewards[:, 1] = self.rewards[:, 1].reshape(-1, mean_num).mean(axis=1)
        self.mean_rewards[:, 2] = self.rewards[:, 2].reshape(-1, mean_num).mean(axis=1)

        self.mean_loss = self.avg_loss.reshape(-1, mean_num).mean(axis=1)
        self.mean_Qvalue = self.avg_Q_value.reshape(-1, mean_num).mean(axis=1)


def preprocess(y, split_num=300):
    x_range = len(y)
    x_new = np.linspace(1, x_range, split_num)
    y_smooth = spline(range(x_range), y, x_new)
    return x_new, y_smooth


def draw(mean_data1, mean_data2, mean_data3):
    x1, y1 = preprocess(mean_data1)
    x2, y2 = preprocess(mean_data2)
    x3, y3 = preprocess(mean_data3)
    plt.plot(x1, y1, 'b', label="none communication network")
    plt.plot(x2, y2, 'r', label="communicating network")
    plt.plot(x3, y3, 'g', label="centralized network")
    plt.legend(loc=2, labelspacing=0)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


no_com_network = NetworkGraph("3-agent/non_com_network_a3_04:16:12:39_adam-0.002.log")
no_com_network.load()
no_com_network.smooth(3)

com_network = NetworkGraph("3-agent/com_network_a3_04:15:23:19_adam-0.002.log")
com_network.load()
com_network.smooth(3)

cen_network = NetworkGraph("3-agent/centralized_network_a3_04:16:14:16_adam-0.002.log")
cen_network.load()
cen_network.smooth(3)

fig = plt.figure(1, figsize=(30, 10))
plt.subplot(221)
draw(no_com_network.mean_score, com_network.mean_score, cen_network.mean_score)
plt.ylabel('scores')
plt.grid(True)

plt.subplot(222)
reward_punish1 = no_com_network.mean_score / (np.sum(no_com_network.mean_collision, axis=1))
reward_punish2 = com_network.mean_score / (np.sum(com_network.mean_collision, axis=1))
reward_punish3 = cen_network.mean_score / (np.sum(cen_network.mean_collision, axis=1))
draw(reward_punish1, reward_punish2, reward_punish3)
plt.ylabel('scores/punishments')
plt.grid(True)

plt.subplot(223)
draw(no_com_network.mean_loss, com_network.mean_loss, cen_network.mean_loss)
plt.ylabel('avg loss')
plt.grid(True)

plt.subplot(224)
draw(no_com_network.mean_Qvalue, com_network.mean_Qvalue, cen_network.mean_Qvalue)
plt.ylabel('avg Q values')
plt.grid(True)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')

plt.show()
