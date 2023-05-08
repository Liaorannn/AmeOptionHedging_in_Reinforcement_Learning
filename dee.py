"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/5/7 9:36
@File : dee.py
"""
import gym
import numpy as np
from collections import namedtuple
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random

import matplotlib.pyplot as plt

Node = namedtuple('Node', ['t', 'm'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BinomialTree:
    def __init__(self, s0=100, K=100, r=0.02, sigma=0.1, maturity=1, n=10, D=1):
        self.s0 = s0
        self.K = K  # ATM strike price
        self.r = r  # riskless interest rate
        self.sigma = sigma  # stocks vol
        self.T = maturity  # option's maturity
        self.n = n  # total steps of Tree
        self.delta_t = self.T / self.n

        ## Calculate key elements in Tree generalization
        self.u = np.exp(self.sigma * np.sqrt(self.delta_t))  # Move up steps
        self.d = 1 / self.u
        self.R = np.exp(self.r * self.delta_t)  # Discount ratio of each step
        self.p = (self.R - self.d) / (self.u - self.d)  # Prob of moving up

        ## Self state
        self.node = None  # current state: Node(t, m)
        self.hedging_pos = 0  # Started hedging position
        self.D = D  # Holding position of derivatives

        ## Env state
        self.grid = None  # Store the tree's Node:  {time t: [Node(t, m1), Node(t, m2) ...]}
        self.observation_space = None  # Dict: {Node(t, m): (stock_price, option_price, {'exercise': bool})}
        self.action_space = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.state_dim = 2  # State: (stock_price, option_value)
        self.action_dim = len(self.action_space)

    def get_price(self, node):  # Return the stock price at Node(i,j)
        return self.s0 * np.power(self.u, node.m)

    def _generate(self):
        grid = {}
        for t in range(self.n + 1):
            grid[t] = []
            for i in range(-t, t + 1, 2):
                grid[t].append(Node(t, i))  # Node at time t
        self.grid = grid

        # return grid

    def _back_propagation(self):
        assert self.grid, 'Grid must be generated first!'
        obs_space = {}  # Restore data at each node {Node : (stock price, option value)

        for t in range(self.n, 0 - 1, -1):

            if t == self.n:  # Value at maturity
                for node in self.grid[t]:
                    stock_p = self.get_price(node)
                    opt_v = max(self.K - stock_p, 0)  # Put option value at maturity
                    obs_space[node] = (stock_p, opt_v, {'exercise': True})  # Restore current price & value
            else:  # Value before maturity
                for node in self.grid[t]:
                    m = node.m
                    stock_p = self.get_price(node)
                    opt_exe_v = max(self.K - stock_p, 0)
                    opt_continue_v = (self.p * (obs_space[Node(t + 1, m + 1)][1]) + (1 - self.p) * (
                        obs_space[Node(t + 1, m - 1)][1])) / self.R

                    if opt_exe_v > opt_continue_v:
                        opt_v = opt_exe_v
                        obs_space[node] = (stock_p, opt_v, {'exercise': True})
                    else:
                        opt_v = opt_continue_v
                        obs_space[node] = (stock_p, opt_v, {'exercise': False})
        self.observation_space = obs_space
        # return obs_space

    def fit(self):  # Initialized the environment
        self._generate()
        self._back_propagation()
        print('Environment initialization completed!')

    def get_holding_value(self, stock_p, opt_v, hedge_pos):
        return hedge_pos * stock_p - opt_v * self.D

    def reset(self):
        assert self.observation_space, 'Env must be fited first!'
        self.node = Node(0, 0)  # Initial state
        self.hedging_pos = 0
        stock_p, opt_v, _ = self.observation_space[self.node]

        # holding_value = self.get_holding_value(stock_p, opt_v, self.hedging_pos)
        # state = (stock_p, holding_value)
        state = (stock_p, opt_v)
        return np.array(state, dtype=np.float32)

    def reward(self, stock_p, opt_v, hedging_pos_new, w=0.001):
        """
        Considering the transaction cost in each step and the final payoff
        :param stock_p: current stock price
        :param opt_v: current option value
        :param hedging_pos_new:  current action
        :param w: trading cost coefficient
        :return:
        """
        trading_cost = - stock_p * (hedging_pos_new - self.hedging_pos) * w  # Negative trading reward
        hedging_diff = - np.square(self.get_holding_value(stock_p, opt_v, hedging_pos_new))  # Negative hedging diff
        return hedging_diff + trading_cost

    def step(self, action):
        """
        :param action: hedging amount (continuous num)
        :return: stock_price, reward, done, {"exercise": bool}
        """
        assert self.node, 'Must reset the environment first'
        assert action in self.action_space, "Input action is not in action space"
        t, m = self.node

        if t == self.n:
            stock_p, opt_v, exercise = self.observation_space[self.node]
            # holding_val = self.get_holding_value(stock_p, opt_v, action)
            # state = (stock_p, holding_val)
            state = (stock_p, opt_v)

            done = True
            reward = self.reward(stock_p, opt_v, action)

            return state, reward, done, {}

        else:
            if np.random.binomial(1, self.p):  # Binomial transition prob
                node = Node(t + 1, m + 1)
            else:
                node = Node(t + 1, m - 1)

            stock_p, opt_v, exercise = self.observation_space[node]

            # holding_val = self.get_holding_value(stock_p, opt_v, action)
            # state = (stock_p, holding_val)
            state = (stock_p, opt_v)

            done = exercise['exercise']
            reward = self.reward(stock_p, opt_v, action)

            ## Update self state
            self.hedging_pos = action
            self.node = node

            return np.array(state, dtype=np.float32), reward, done, {}


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x.shape = (batch, state_dim)  batch:一次trajectory中的steps
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, 0).to(device)

        probs = self.forward(state).to(device)
        # highest_prob = np.random.choice(self.action_dim,  p=np.squeeze(prob.detach().numpy()))
        m = Categorical(probs)
        action = m.sample()
        # log_prob = torch.log(prob.squeeze(0)[highest_prob])

        return action.item() / 10, m.log_prob(action)


class AgentVPG:
    GAMMA = 0.9
    max_episodes = 2000

    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net

    @staticmethod
    def discounted_future_reward(rewards: list):
        discounted_r = [rewards[-1]]

        for r in rewards[-2::-1]:
            rr = AgentVPG.GAMMA * discounted_r[-1]
            Gt = r + rr
            discounted_r.append(Gt)
        discounted_r = discounted_r[::-1]
        return discounted_r

    def update_policy(self, rewards, log_probs):
        discounted_rewards = self.discounted_future_reward(rewards)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_grads = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_grads.append(-log_prob * Gt)

        self.policy_net.optimizer.zero_grad()
        policy_grad = torch.stack(policy_grads).sum()
        policy_grad.backward()
        self.policy_net.optimizer.step()

    def fit(self):
        # num_steps = []
        # avg_num_steps = []
        all_rewards = []
        mean_rewards = []

        for episode in range(AgentVPG.max_episodes):
            state = self.env.reset()
            log_probs = []
            rewards = []
            for step in range(15):  # total steps is 10

                action, log_prob = self.policy_net.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    # 完成一次 episode/rollout，得到一次完整的 trajectory
                    self.update_policy(rewards, log_probs)
                    # num_steps.append(step)
                    # avg_num_steps.append(np.mean(num_steps[-3:]))

                    all_rewards.append(sum(rewards))
                    mean_rewards.append(np.mean(rewards))
                    if episode % 100 == 0:
                        print(
                            f'episode: {episode}, '
                            f'total reward: {sum(rewards)}, '
                            f'mean_reward: {np.mean(rewards)}, '
                            f'length: {step}')
                    break

                state = next_state

        plt.plot(all_rewards)
        plt.plot(mean_rewards)
        plt.legend(['all_rewards', 'mean_rewards'])
        plt.xlabel('episode')
        plt.show()


binomial_env = BinomialTree()
binomial_env.fit()

# VPG_policy = PolicyNetwork(binomial_env.state_dim, binomial_env.action_dim, 16)
# VPG_policy.to(device)
# agent = AgentVPG(binomial_env, VPG_policy)
#
# agent.fit()


Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    GAMMA = 0.9
    LearningRate = 3e-4
    BATCH_SIZE = 32
    # EPSILON = 0.9
    Max_Episode = 500
    Capacity = 2000
    Target_network_update = 30

    def __init__(self, env=BinomialTree()):
        self.env = env
        self.eval_net = Net(self.env.state_dim, self.env.action_dim).to(device)
        self.target_net = Net(self.env.state_dim, self.env.action_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=DQNAgent.LearningRate)

        self.memory = ReplayBuffer(DQNAgent.Capacity)
        # self.loss = nn.MSELoss()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def choose_action(self, state, episode):
        state = torch.unsqueeze(torch.from_numpy(state).float(), dim=0).to(device)
        # epsilon = 0.5*1 / (1 + episode)  # Focus on exploration first and do exploitation next
        epsilon_threshold = 0.05 + (0.9 - 0.05) * np.exp(- episode / 1000)

        if np.random.uniform() < epsilon_threshold:
            # action = [random.sample(self.env.action_space, 1)]
            action = np.random.choice(self.env.action_space)
            # return torch.tensor(action, device=device, dtype=torch.long)
            return action
        else:
            with torch.no_grad():
                return self.eval_net.forward(state).max(1)[1].item() / 10  # TODO: / 10?

    def update_q_function(self):
        if len(self.memory) < DQNAgent.BATCH_SIZE:
            return

        batch = self.memory.sample(DQNAgent.BATCH_SIZE)
        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_terminal_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_terminal_next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.eval_net(state_batch).gather(dim=1, index=action_batch)
        next_state_values = torch.zeros(DQNAgent.BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_terminal_mask] = self.target_net(non_terminal_next_state_batch).max(dim=1)[0]

        expected_state_action_values = reward_batch + (next_state_values * DQNAgent.GAMMA)
        expected_state_action_values = torch.unsqueeze(expected_state_action_values, 1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 100)
        self.optimizer.step()

    def fit(self):
        reward_sum = []
        reward_mean = []

        for eps in range(DQNAgent.Max_Episode):  # 700
            state = self.env.reset()
            rewards = []
            # state = torch.unsqueeze(torch.from_numpy(state).float(), 0).to(device)

            for step in range(15):
                action = self.choose_action(state, eps)

                next_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)

                if done:
                    next_state_m = None
                else:
                    next_state_m = torch.unsqueeze(torch.from_numpy(next_state).float(), dim=0).to(device)
                reward_m = torch.tensor([reward], device=device)
                state_m = torch.unsqueeze(torch.from_numpy(state).float(), dim=0).to(device)
                action_m = torch.tensor([[int(action * 10)]], device=device)
                self.memorize(state_m, action_m, next_state_m, reward_m)

                state = next_state
                self.update_q_function()

                if done:
                    print(f'episode: {eps}, steps: {step}')
                    break

            if eps % DQNAgent.Target_network_update == 0:  # Update Network parameters after 30 times
                self.target_net.load_state_dict(self.eval_net.state_dict())
            reward_sum.append(np.sum(rewards))
            reward_mean.append(np.mean(rewards))

        plt.plot(reward_sum)
        plt.plot(reward_mean)
        plt.legend(['sum_rewards', 'mean_rewards'])
        plt.xlabel('episode')
        plt.show()
        return


DQN_policy = DQNAgent(binomial_env)

DQN_policy.fit()

#
# import torch                                    # 导入torch
# import torch.nn as nn                           # 导入torch.nn
# import torch.nn.functional as F                 # 导入torch.nn.functional
# import numpy as np                              # 导入numpy
# import gym                                      # 导入gym
#
# # 超参数
# BATCH_SIZE = 32                                 # 样本数量
# LR = 0.01                                       # 学习率
# EPSILON = 0.9                                   # greedy policy
# GAMMA = 0.9                                     # reward discount
# TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
# MEMORY_CAPACITY = 2000                          # 记忆库容量
# env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
# N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
# N_STATES = env.observation_space.shape[0]       # 杆子状态个数 (4个)
#
#
# """
# torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
# nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
# 定义网络：
#     需要继承nn.Module类，并实现forward方法。
#     一般把网络中具有可学习参数的层放在构造函数__init__()中。
#     只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
# """
#
#
# # 定义Net类 (定义网络)
# class Net(nn.Module):
#     def __init__(self):                                                         # 定义Net的一系列属性
#         # nn.Module的子类函数必须在构造函数中执行父类的构造函数
#         super(Net, self).__init__()                                             # 等价与nn.Module.__init__()
#
#         self.fc1 = nn.Linear(N_STATES, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
#         self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
#         self.out = nn.Linear(50, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
#         self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
#
#     def forward(self, x):                                                       # 定义forward函数 (x为状态)
#         x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
#         actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
#         return actions_value                                                    # 返回动作值
#
#
# # 定义DQN类 (定义两个网络)
# class DQN(object):
#     def __init__(self):                                                         # 定义DQN的一系列属性
#         self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
#         self.learn_step_counter = 0                                             # for target updating
#         self.memory_counter = 0                                                 # for storing memory
#         self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
#         self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
#
#     def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
#         if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
#             actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
#             action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
#             action = action[0]                                                  # 输出action的第一个数
#         else:                                                                   # 随机选择动作
#             action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
#         return action                                                           # 返回选择的动作 (0或1)
#
#     def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
#         transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
#         # 如果记忆库满了，便覆盖旧的数据
#         index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
#         self.memory[index, :] = transition                                      # 置入transition
#         self.memory_counter += 1                                                # memory_counter自加1
#
#     def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
#         # 目标网络参数更新
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
#             self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
#         self.learn_step_counter += 1                                            # 学习步数自加1
#
#         # 抽取记忆库中的批数据
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
#         b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
#         b_s = torch.FloatTensor(b_memory[:, :N_STATES])
#         # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
#         b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
#         # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
#         b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
#         # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
#         b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
#         # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列
#
#         # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
#         q_eval = self.eval_net(b_s).gather(1, b_a)
#         # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
#         q_next = self.target_net(b_s_).detach()
#         # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
#         # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
#         loss = self.loss_func(q_eval, q_target)
#         # 输入32个评估值和32个目标值，使用均方损失函数
#         self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
#         loss.backward()                                                 # 误差反向传播, 计算参数更新值
#         self.optimizer.step()                                           # 更新评估网络的所有参数
#
#
# dqn = DQN()                                                             # 令dqn=DQN类
#
# for i in range(400):                                                    # 400个episode循环
#     print('<<<<<<<<<Episode: %s' % i)
#     s = env.reset()                                                     # 重置环境
#     episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
#
#     while True:                                                         # 开始一个episode (每一个循环代表一步)
#         env.render()                                                    # 显示实验动画
#         a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
#         s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
#
#         # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
#         x, x_dot, theta, theta_dot = s_
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         new_r = r1 + r2
#
#         dqn.store_transition(s, a, new_r, s_)                 # 存储样本
#         episode_reward_sum += new_r                           # 逐步加上一个episode内每个step的reward
#
#         s = s_                                                # 更新状态
#
#         if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000
#             # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
#             dqn.learn()
#
#         if done:       # 如果done为True
#             # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
#             print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
#             break                                             # 该episode结束
