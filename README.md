# AmeOptionHedging_in_Reinforcement_Learning
HKUST MAFM course project for American Option Hedging in Reinforcement Learning


# Introduction
This project is one of my class assignments completed at HKUST M.Sc. in 
Financial Mathematics (MAFM) program , 2023 Spring, for the course MAFM 5370 
Reinforcement Learning with Financial Applications.
  
This project consider the simplest situation in American Option Hedging and 
using several fundamental method in RL to achieve the optimal investment policy. 
  
The project contains the code of: Environment settings, Policy Gradient(REINFORCE), DQN

**Reference:**

- Learning Materials is from my class teacher: Chak Wong.  
- Most of the Algorithme Pseudocode is from Bilibili course: 
[Mathematical Foundations of Reinforcement Learning](https://www.bilibili.com/video/BV1KY4y1N7H8/?spm_id_from=333.788&vd_source=c6859ec5158d515b50f001aba53cc8f9)
and its Book GitHub link: 
[Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning)

- The code of the algorithms reference following sources:
  - [Pytorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
  - [OpenAI Spinning up Algorithms Doc](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

_Attention: there may be some errors in the algorithm code, since I'm still a beginner
in RL, please raise issue if you find sth wrong._

# Problem Setting
> In a binomial model of a single stock with non-zero interest rate, assume that we can hedge any fraction of a stock,
> use policy gradient to train the optimal policy of hedging an ATM American put option with maturity T = 10.

The project's objective is to find out the optimal hedging quantity the issuer need in order to minimize the potential loss
from selling an American put option. The hedging quantity should consider the early exercise situation.

# Model Assumption & Environment Setting
According to the Black-Scholes Model, our hedging decision is effected by the underlying price and 
the option price. Therefore, I set (stock price, option price) as the state in the RL model. As for the reward
setting, considering the fact that the trader wants to minimize the hedging difference as well as the trading cost, 
I simply set the reward is function of those two factors. Due to the constraint of my computer running speed, 
I have to discretize the action space so the agent's hedging quantities can only by (0.1, 0.2...1). So 
the problem can be seen as a discrete-state & discrete action problem.

The Env setting is as below:
- State: (stock price, option price)
- Action: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
- Reward: Trading Cost + Hedging Difference  
  $$ w_{trading cost} * S_t * Hedging Position change + (Hedging value - option value)^2 $$
- Done: Arrives at maturity or Early exercise
- Stock price follows the binomial tree distribution

interest rate, vol, p and other factors can be checked in my code.
In order to compute the option price and early exercise's time, I use back propagation in binomial tree 
model, which is written in the Env setting code. After fitting the Env model, one can apply agent algorithm.

# Algorithm Code

## REINFORCE (VPG)
I use simple neural network as the policy function approximation. The algorithm contains two parts: 
Network setting and VPG agent.
The VPG algorithm seen to converge first after 200 episodes.

## DQN
The DQN code contains three parts: Replay buffer, Network setting and DQN agent.
The DQN running result converges badly, I guess it's because the batch size is too small and 
the episode num is not big enough, or just because the reward setting is not wise enough. Therefore, the
 DQN algorithm still needs to improve.

*Plz Check [ipynb](test0507.ipynb) file to see the algorithm running results.*

# Future Works:
Apply other Algorithms like:
- [ ] Actor Critic
- [ ] PPO
- [ ] DDPG