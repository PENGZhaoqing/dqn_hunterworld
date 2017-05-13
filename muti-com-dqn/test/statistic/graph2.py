from io import open
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
import math
import os
from scipy import ndimage

class NetworkGraph(object):
    def __init__(self, file_name):
        self.steps = []
        self.reward = []
        self.collision = []
        self.avg_loss = []
        self.avg_Q_value = []
        self.filename = file_name
        self.x_axis = None
        self.mean_Qvalue = None
        self.mean_reward = None
        self.mean_steps = None
        self.mean_collision = None
        self.mean_loss = None

    def load(self, mean_num):
        with open(self.filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                columns = line.split(',')
                if len(columns) == 10:
                    self.steps.append(columns[3].split('/')[1])
                    self.reward.append(columns[4].split(':')[1])
                    self.collision.append(columns[7].split(':')[1].split('/')[1])
                    self.avg_loss.append(columns[8].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[9].split(':')[1].split('/')[0])

        self.x_axis = np.array(range(len(self.reward))) * 5
        self.steps = np.array(self.steps, dtype=int)
        self.reward = np.array(self.reward, dtype=float)
        self.collision = np.array(self.collision, dtype=float)
        self.avg_loss = np.array(self.avg_loss, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)

        if mean_num is not None:
            self.steps = self.steps.reshape(-1, mean_num).mean(axis=1)
            self.reward = self.reward.reshape(-1, mean_num).mean(axis=1)
            self.collision = self.collision.reshape(-1, mean_num).mean(axis=1)
            self.avg_loss = self.avg_loss.reshape(-1, mean_num).mean(axis=1)
            self.avg_Q_value = self.avg_Q_value.reshape(-1, mean_num).mean(axis=1)


def preprocess(y, split_num=300):
    x_range = len(y)
    x_new = np.linspace(1, x_range, split_num)
    y_smooth = spline(range(x_range), y, x_new)
    return x_new, y_smooth


def plot(x1, y1, x2, y2, x3, y3):
    plt.plot(x1, y1, 'b', label="none communication network")
    plt.plot(x2, y2, 'r', label="communicating network")
    plt.plot(x3, y3, 'g', label="centralized network")
    plt.legend(loc=2, labelspacing=0)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def draw(mean_data1, mean_data2, mean_data3, smooth=False, split_num=300):
    if smooth:
        x1, y1 = preprocess(mean_data1, split_num)
        x2, y2 = preprocess(mean_data2, split_num)
        x3, y3 = preprocess(mean_data3, split_num)
        plot(x1, y1, x2, y2, x3, y3)
    else:
        x1 = range(len(mean_data1))
        x2 = range(len(mean_data2))
        x3 = range(len(mean_data3))
        plot(x1, mean_data1, x2, mean_data2, x3, mean_data3)


smooth = False
mean_num = 3
split_num = 300

no_com_network = NetworkGraph("2-agent/non_com_network_a2_04:16:15:33_adam-0.002.log")
no_com_network.load(mean_num)
com_network = NetworkGraph("2-agent/com_network_a2_04:16:17:29_adam-0.002.log")
com_network.load(mean_num)
cen_network = NetworkGraph("2-agent/centralized_network_a2_04:16:18:38_adam-0.002.log")
cen_network.load(mean_num)

fig = plt.figure(1, figsize=(30, 10))
plt.subplot(221)
draw(no_com_network.reward, com_network.reward, cen_network.reward, smooth=smooth, split_num=split_num)
plt.ylabel('scores')
plt.grid(True)

plt.subplot(222)
draw(no_com_network.collision, com_network.collision, cen_network.collision, smooth=smooth, split_num=split_num)
plt.ylabel('punishments')
plt.grid(True)

plt.subplot(223)
draw(no_com_network.avg_loss, com_network.avg_loss, cen_network.avg_loss, smooth=smooth, split_num=split_num)
plt.ylabel('avg loss')
plt.grid(True)

plt.subplot(224)
draw(no_com_network.avg_Q_value, com_network.avg_Q_value, cen_network.avg_Q_value, smooth=smooth, split_num=split_num)
plt.ylabel('avg Q values')
plt.grid(True)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')

plt.show()
