import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from environment import Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        # 折扣
        self.gamma = 0.9
        # 探索率
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98  # 每5个episode更新一次，衰减率可适当调大
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 50
        self.update_count = 0

        self.model, self.optimizer = self._build_model()
        self.target_model, _ = self._build_model()
        self.update_target_model()

    def _build_model(self):
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, action_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = DQN(self.state_size, self.action_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    # 更新目标网络
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 在经验池中添加经验
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 随机对动作进行抽样
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    # 经验回放
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([s for s, a, r, ns, d in batch]).to(device)
        actions = torch.LongTensor([a for s, a, r, ns, d in batch]).to(device)
        rewards = torch.FloatTensor([r for s, a, r, ns, d in batch]).to(device)
        next_states = torch.FloatTensor([ns for s, a, r, ns, d in batch]).to(device)
        dones = torch.FloatTensor([d for s, a, r, ns, d in batch]).to(device)

        q_target_next = self.target_model(next_states).detach().max(1)[0]
        q_target = rewards + (self.gamma * q_target_next * (1 - dones))
        q_current = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_model()


if __name__ == "__main__":
    env = Env()
    state_size = 2
    action_size = env.n_actions
    agent = DQNAgent(state_size, action_size)
    episodes = 3000

    for e in range(episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0

        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            if time % 4 == 0:
                agent.replay()

            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, 步数: {time}, 总奖励: {total_reward}, ε: {agent.epsilon:.2f}")
                break

        # 每5个episode更新一次epsilon
        if (e + 1) % 15 == 0:  # 使用e+1确保第5、10、15...个episode结束后更新
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                print(f"第{e + 1}个episode后更新epsilon: {agent.epsilon:.4f}")
