import math

import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of actions
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  # 衰减系数
        self.epsilon = 0
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格
        # self.epsilon_end = cfg.epsilon_end
        # self.epsilon_start = cfg.epsilon_start
        # self.epsilon_decay = cfg.epsilon_decay

    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        # print(self.Q_table)
        self.sample_count += 1
        self.epsilon = 1 - math.exp(-1 * self.sample_count / 400)
        # print(self.epsilon)
        if np.random.uniform(0, 1) < self.epsilon:
            Q_list = self.Q_table[state, :]
            maxQ = np.max(Q_list)
            action_list = np.where(Q_list == maxQ)[0]
            action = np.random.choice(action_list)
        else:
            action = np.random.choice(self.action_dim)  # 随机选取一个动作探索
        return action

    def update(self, state, action, reward, next_state, done):
        ############################ Q表格的更新方法（需要完成）##################################
        Q_predict = self.Q_table[state, action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[next_state, :])
        self.Q_table[state, action] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
