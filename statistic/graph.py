from io import open
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
import math
import os


# print os.path.abspath(os.curdir)

class NetworkGraph(object):
    def __init__(self, file_name):
        self.steps = []
        self.reward = []
        self.collision = []
        self.avg_loss = []
        self.avg_Q_value = []
        self.filename = file_name

    def load(self):
        with open(self.filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                columns = line.split(',')
                if len(columns) == 10:
                    self.steps.append(columns[3].split('/')[1])
                    self.reward.append(columns[4].split(':')[1])
                    self.collision.append(columns[7].split(':')[1].split('/')[0])
                    self.avg_loss.append(columns[8].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[9].split(':')[1].split('/')[0])

        self.x_axis = np.array(range(len(self.reward))) * 5
        self.steps = np.array(self.steps, dtype=int)
        self.reward = np.array(self.reward, dtype=float)
        self.collision = np.array(self.collision, dtype=float)
        self.avg_loss = np.array(self.avg_loss, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)


def draw(figure_num, network):
    fig = plt.figure(figure_num, figsize=(20, 5))
    fig.canvas.set_window_title(network.filename)
    plt.subplot(221)
    # plt.scatter(network.x_axis, network.reward, s=4, c='g')
    plt.plot(network.x_axis, network.reward, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.reward, 'b')
    plt.ylabel('scores')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(network.x_axis, network.collision, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.collision, 'b')
    plt.ylabel('collision')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(network.x_axis, network.avg_loss, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.avg_loss, 'b')
    plt.ylabel('avg loss')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(network.x_axis, network.avg_Q_value, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.avg_Q_value, 'b')
    plt.ylabel('avg Q values')
    plt.grid(True)
    # fig.tight_layout()

    # avg_Q_value=np.array(avg_Q_value)
    # avg_Q_value = avg_Q_value.reshape(-1, 3).mean(axis=1)
    # x_range = len(avg_Q_value)
    # x_new = np.linspace(1, x_range, 100)
    # y_smooth = spline(range(x_range), avg_Q_value, x_new)
    # plt.plot(x_new, y_smooth)

    # plt.setp(lines, color='b', linewidth=1.0)
    # plt.title(r'$\sigma_i=15$')


no_com_network = NetworkGraph("2-agent/non_com_network_a2_04:16:15:33_adam-0.002.log")
no_com_network.load()
com_network = NetworkGraph("2-agent/com_network_a2_04:16:17:29_adam-0.002.log")
com_network.load()
cen_network = NetworkGraph("2-agent/centralized_network_a2_04:16:18:38_adam-0.002.log")
cen_network.load()

draw(1, no_com_network)
draw(2, com_network)
draw(3, cen_network)
plt.show()
