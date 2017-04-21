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
                    self.collision.append(columns[7].split(':')[1].split('/')[1])
                    self.avg_loss.append(columns[8].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[9].split(':')[1].split('/')[0])

        self.x_axis = np.array(range(len(self.reward))) * 5
        self.steps = np.array(self.steps, dtype=int)
        self.reward = np.array(self.reward, dtype=float)
        self.collision = np.array(self.collision, dtype=float)
        self.avg_loss = np.array(self.avg_loss, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)


def draw(network):
    plt.subplot(222)
    plt.plot(network.x_axis, network.collision, 'bo', markersize=4, mfc='b')
    plt.plot(network.x_axis, network.collision, 'b')
    plt.ylabel('punishments')
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


network1 = NetworkGraph("log/.log")
network1.load()
network2 = NetworkGraph("log/adam-0.0002-5.log")
network2.load()
network3 = NetworkGraph("log/single-custom_04:18:20:23_adam-0.002.log")
network3.load()

fig = plt.figure(1, figsize=(30, 10))
plt.subplot(221)
plt.plot(network1.x_axis, network1.reward, 'bo', markersize=4, mfc='b')
plt.plot(network1.x_axis, network1.reward, 'b', label="adam 0.0001 10")
plt.plot(network2.x_axis, network2.reward, 'ro', markersize=4, mfc='r')
plt.plot(network2.x_axis, network2.reward, 'r', label="adam 0.0002 5")
plt.plot(network3.x_axis, network3.reward, 'go', markersize=4, mfc='g')
plt.plot(network3.x_axis, network3.reward, 'g', label="adam 0.0002 10")
plt.legend(loc=2, labelspacing=0)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
plt.ylabel('scores')
plt.grid(True)

plt.subplot(222)
plt.plot(network1.x_axis, network1.collision, 'bo', markersize=4, mfc='b')
plt.plot(network1.x_axis, network1.collision, 'b', label="adam 0.0001 10")
plt.plot(network2.x_axis, network2.collision, 'ro', markersize=4, mfc='r')
plt.plot(network2.x_axis, network2.collision, 'r', label="adam 0.0002 5")
plt.plot(network3.x_axis, network3.collision, 'go', markersize=4, mfc='g')
plt.plot(network3.x_axis, network3.collision, 'g', label="adam 0.0002 10")
plt.legend(loc=2, labelspacing=0)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
plt.ylabel('rewards/collisons')
plt.grid(True)

plt.subplot(223)
plt.plot(network1.x_axis, network1.avg_loss, 'bo', markersize=4, mfc='b')
plt.plot(network1.x_axis, network1.avg_loss, 'b', label="adam 0.0001 10")
plt.plot(network2.x_axis, network2.avg_loss, 'ro', markersize=4, mfc='r')
plt.plot(network2.x_axis, network2.avg_loss, 'r', label="adam 0.0002 5")
plt.plot(network3.x_axis, network3.avg_loss, 'go', markersize=4, mfc='g')
plt.plot(network3.x_axis, network3.avg_loss, 'g', label="adam 0.0002 10")
plt.legend(loc=2, labelspacing=0)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
plt.ylabel('avg loss')
plt.grid(True)

plt.subplot(224)
plt.plot(network1.x_axis, network1.avg_Q_value, 'bo', markersize=4, mfc='b')
plt.plot(network1.x_axis, network1.avg_Q_value, 'b', label="adam 0.0001 5")
plt.plot(network2.x_axis, network2.avg_Q_value, 'ro', markersize=4, mfc='r')
plt.plot(network2.x_axis, network2.avg_Q_value, 'r', label="adam 0.0002 5")
plt.plot(network3.x_axis, network3.avg_Q_value, 'go', markersize=4, mfc='g')
plt.plot(network3.x_axis, network3.avg_Q_value, 'g', label="adam 0.0002 10")
plt.legend(loc=2, labelspacing=0)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
plt.ylabel('avg Q values')
plt.grid(True)

plt.show()
