import numpy as np
import random
from environment import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 采样 <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        """
        Q-Learning算法的学习函数，用于更新Q表中的值
        参数:
            state: 当前状态
            action: 在当前状态下执行的动作
            reward: 执行动作后获得的奖励
            next_state: 执行动作后转移到的下一个状态
        """
        # 获取当前状态-动作对在Q表中的Q值
        current_q = self.q_table[state][action]
        # 贝尔曼方程更新
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # 从Q-table中选取动作
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(self.actions)
        else:
            # 从q表中选择
            state_action = self.q_table[state]
            print(state_action)
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):

        """
        在给定的状态-动作值列表中，随机返回一个具有最大值的动作索引

        参数:
            state_action (list): 包含各个动作值的列表

        返回:
            int: 随机选择的一个具有最大值的动作索引
        """
        max_index_list = []  # 用于存储所有最大值对应的索引
        max_value = state_action[0]  # 初始化最大值为列表第一个元素
        for index, value in enumerate(state_action):  # 遍历列表，获取索引和值
            if value > max_value:  # 如果找到更大的值
                max_index_list.clear()  # 清空之前的索引列表
                max_value = value  # 更新最大值
                max_index_list.append(index)  # 将当前索引加入列表
            elif value == max_value:  # 如果找到相等的值
                max_index_list.append(index)  # 将当前索引加入列表
        return random.choice(max_index_list)  # 从所有最大值索引中随机返回一个


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    for episode in range(1000):
        state = env.reset()
        while True:
            env.render()
            # agent产生动作
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)
            # 更新Q表
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            env.print_value_all(agent.q_table)
            # 当到达终点就终止游戏开始新一轮训练
            if done:
                break
