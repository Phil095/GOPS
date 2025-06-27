import sys
sys.path.append("../")
sys.path.append("../gops/env/env_lasvsim/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import collections
import random

from gops.env.env_lasvsim.lasvsim_env import LASVSimEnv

import matplotlib.pyplot as plt

class ReplayBuffer:
    """
    经验回放池
    """
    def __init__(self, capacity, device, multy_lines_state_dim, multy_lines_selection_dim, track_state_dim, track_action_dim):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self._size = 0

        # 分配内存
        self.multy_lines_states = np.zeros((capacity, multy_lines_state_dim), dtype=np.float32)
        self.multy_lines_selections = np.zeros((capacity, multy_lines_selection_dim), dtype=np.float32)
        self.track_states = np.zeros((capacity, track_state_dim), dtype=np.float32)
        self.track_actions = np.zeros((capacity, track_action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_multy_lines_states = np.zeros((capacity, multy_lines_state_dim), dtype=np.float32)
        self.next_track_states = np.zeros((capacity, track_state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, multy_lines_state, multy_lines_selection, track_state, track_action, reward, next_multy_lines_state, next_track_state, done):
        self.multy_lines_states[self.ptr] = multy_lines_state
        self.multy_lines_selections[self.ptr] = multy_lines_selection.detach().numpy()
        self.track_states[self.ptr] = track_state
        self.track_actions[self.ptr] = track_action.detach().numpy()
        self.rewards[self.ptr] = reward
        self.next_multy_lines_states[self.ptr] = next_multy_lines_state
        self.next_track_states[self.ptr] = next_track_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size): 
        """随机采样一个 batch"""
        idx = np.random.randint(0, self._size, size=batch_size)
        multy_lines_states = torch.tensor(self.multy_lines_states[idx], device=self.device, dtype=torch.float32)
        multy_lines_selections = torch.tensor(self.multy_lines_selections[idx], device=self.device, dtype=torch.float32)
        track_states = torch.tensor(self.track_states[idx], device=self.device, dtype=torch.float32)
        track_actions = torch.tensor(self.track_actions[idx], device=self.device, dtype=torch.float32)
        rewards = torch.tensor(self.rewards[idx], device=self.device, dtype=torch.float32)
        next_multy_lines_states = torch.tensor(self.next_multy_lines_states[idx], device=self.device, dtype=torch.float32)
        next_track_states = torch.tensor(self.next_track_states[idx], device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.dones[idx], device=self.device, dtype=torch.float32)

        return multy_lines_states, multy_lines_selections, track_states, track_actions, rewards, next_multy_lines_states, next_track_states, dones

    def load_dataset(self, dataset):
        N = min(len(dataset['multy_lines_states']), self.capacity)
        self.multy_lines_states[:N] = dataset['multy_lines_states'][:N]
        self.multy_lines_selections[:N] = dataset['multy_lines_selections'][:N]
        self.track_states[:N] = dataset['track_states'][:N]
        self.track_actions[:N] = dataset['track_actions'][:N]
        self.rewards[:N] = dataset['rewards'][:N]
        self.next_multy_lines_states[:N] = dataset['next_multy_lines_states'][:N]
        self.next_track_states[:N] = dataset['next_track_states'][:N]
        self.dones[:N] = dataset['dones'][:N]
        self._size = N
        self.ptr = N % self.capacity

    def size(self): 
        return self._size


class HighLevelSACActorMLP(nn.Module):
    def __init__(self, input_dim=2406, hidden_sizes=[1024, 512], output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim)
        )
    
    def forward(self, x):
        # x: [B, 2400]
        logits = self.net(x)            # [B, 3]
        probs = F.softmax(logits, dim=1)  # 转为概率
        return probs, logits  # SAC通常使用 log probs

# # 示例使用
# model = SACActorMLP()
# sample = torch.randn(16, 2400)
# probs, logits = model(sample)
# print(probs.shape)  # torch.Size([16, 3])

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class LowLevelSACActorMLP(nn.Module):
    def __init__(self,
                 input_dim=2400,
                 hidden_sizes=[1024, 512],
                 action_dim=2,  # 前轮转角 + 加速度
                 action_limit=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.action_limit = action_limit  # 用于限制输出范围

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()  # reparameterization trick
        action = torch.tanh(z)  # 限制输出在 [-1, 1]
        if self.action_limit is not None:
            action = action * self.action_limit  # scale to实际范围
        # 计算 log_prob 时考虑 tanh 效果
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.fc2 = torch.nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc_out = torch.nn.Linear(hidden_dim[1], 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class HighLevelSAC:
    """
    负责参考线选择的SAC
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.actor = HighLevelSACActorMLP(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob.sum(dim=-1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards'].view(-1, 1)
        next_states = transition_dict['next_states']
        dones = transition_dict['dones'].view(-1, 1)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


class LowLevelSAC:
    """
    负责具体控制指令的SAC
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.actor = LowLevelSACActorMLP(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action, probs = self.actor.sample(state)
        return action, probs

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor.sample(next_states)
        entropy = -log_prob.sum(dim=-1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards'].view(-1, 1)
        next_states = transition_dict['next_states']
        dones = transition_dict['dones'].view(-1, 1)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


"""
training
"""
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjIxLCJvaWQiOjEwMiwibmFtZSI6IueOi-WGoOiHoyIsImlkZW50aXR5Ijoibm9ybWFsIiwicGVybWlzc2lvbnMiOltdLCJpc3MiOiJ1c2VyIiwic3ViIjoiTGFzVlNpbSIsImV4cCI6MTc1MTI3MjkwMiwibmJmIjoxNzUwNjY4MTAyLCJpYXQiOjE3NTA2NjgxMDIsImp0aSI6IjIxIn0.knJCYLxY3mBaVTju9MJgIJ1ReqcgUzdWSvP_ZpLI6Ws"
task_id = 12703
record_id = 39267
endpoint = "https://qianxing-api.risenlighten.com"

if __name__ == "__main__":
    env = LASVSimEnv(token=token, endpoint=endpoint, task_id=task_id, record_id=record_id)
    # for i in range(100):
    #     ego_state = env.state
    #     refline_obs = env.reference_lines
    #     state = np.concatenate([ego_state, refline_obs.flatten()])
    #     print("state", state.shape)
    # env.stop()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100

    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64

    # target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ego_state_dim = 6
    multi_lines_dim = 2400
    ref_line_dim = 800

    # 共用一个回放池
    replay_buffer = ReplayBuffer(buffer_size, device, ego_state_dim+multi_lines_dim, 3, ego_state_dim+ref_line_dim, 2)

    high_level_agent = HighLevelSAC(ego_state_dim+multi_lines_dim, [1024, 512], 3, None, actor_lr, critic_lr, alpha_lr, -3, tau, gamma, device)
    low_level_agent = LowLevelSAC(ego_state_dim+ref_line_dim, [512, 256], 2, None, actor_lr, critic_lr, alpha_lr, -2, tau, gamma, device)

    # action limit
    scale = torch.tensor([0.6, 6.1], device=device)
    bias = torch.tensor([0.0, -1.9], device=device)

    return_list = []
    for i in range(1):
        episode_return = 0.0
        env.reset()
        ego_state = env.state
        refline_obs = env.reference_lines
        done = False
        while not done:
            multy_lines_state = np.concatenate([ego_state, refline_obs.flatten()])
            lines_probs = high_level_agent.take_action(multy_lines_state)
            track_idx = np.argmax(lines_probs.detach().numpy())
            track_line = refline_obs[track_idx]
            track_line_state = np.concatenate([ego_state, track_line.flatten()])
            action, _ = low_level_agent.take_action(track_line_state)
            actual_action = action * scale + bias
            next_ego_state, next_refline_obs, reward, done, _ = env.step(refline_idx=track_idx, action=actual_action.detach().numpy()[0])
            next_multy_lines_state = np.concatenate([next_ego_state, next_refline_obs.flatten()])
            next_track_line_state = np.concatenate([next_ego_state, next_refline_obs[track_idx].flatten()])
            replay_buffer.add(multy_lines_state, lines_probs, track_line_state, action, reward, next_multy_lines_state, next_track_line_state, done)
            ego_state = next_ego_state
            refline_obs = next_refline_obs
            episode_return = episode_return + reward
            if replay_buffer.size() > minimal_size:
                # update low level sac
                _, _, b_s, b_a, b_r, _, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                low_level_agent.update(transition_dict)
        
        return_list.append(episode_return)

        if i % 10 == 0 and replay_buffer.size() > minimal_size:
            # update high level sac
            b_s, b_a, _, _, b_r, b_ns, _, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            high_level_agent.update(transition_dict)

    # save
    torch.save(high_level_agent.actor.state_dict(), "high_level_agent.pth")
    torch.save(low_level_agent.actor.state_dict(), "low_level_agent.pth")

    plt.plot(return_list)
    plt.show()
