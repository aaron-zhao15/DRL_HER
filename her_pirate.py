import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from environment.pirate_env import PirateEnv
import numpy as np
import cv2
import copy

Experience = namedtuple("Experience", field_names="state action reward next_state done")


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Critic(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dim, hidden_dim=64, device="cpu"):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + discrete_dim + continuous_dim, hidden_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, x):
        return self.model(x)
    
class Actor(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dim, hidden_dim=64, device="cpu"):
        super(Actor, self).__init__()
        latent_dim = 32
        # self.discrete_model = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim, device=device),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, latent_dim, device=device),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, discrete_dim)
        # )
        # self.continuous_model = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim, device=device),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, latent_dim, device=device),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, continuous_dim)
        # )
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, discrete_dim+continuous_dim)
        )
        self.squash = nn.Tanh()
        # should be decoding Q values

        self.continuous_passthrough_layer = nn.Linear(state_dim, continuous_dim)
        nn.init.zeros_(self.continuous_passthrough_layer.weight)
        nn.init.zeros_(self.continuous_passthrough_layer.bias)
        self.continuous_passthrough_layer.weight.requires_grad = False
        self.continuous_passthrough_layer.bias.requires_grad = False

    def forward(self, x):
        # discrete_qs = self.discrete_model(x)
        # continuous = self.continuous_model(x) #+ self.continuous_passthrough_layer(x)
        # return discrete_qs, continuous
        return self.model(x)

class PDDPGAgent:
    def __init__(self, state_dim, discrete_dim=1, continuous_dim=2, device="cpu"):
        self.device = device
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        self.gamma = 0.99
        self.batch_size = 128
        self.train_start = 1000
        self.tau = 0.01

        self.memory = ReplayMemory(int(1e6))

        self.actor = Actor(state_dim, discrete_dim, continuous_dim, device=device)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, discrete_dim, continuous_dim, device=device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001, weight_decay=0.1)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0.1)

        # Copy parameters to target networks
        self.update_target_hard()

    def push_experience(self, state, action, reward, next_state, done):
        self.memory.push(Experience(state, action, reward, next_state, done))

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def update_target_hard(self):
        self.hard_update(self.actor, self.target_actor)
        self.hard_update(self.critic, self.target_critic)
    
    def update_target_soft(self):
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    # continuous approach to movement (dx, dy, act)
    def act(self, state, epsilon=0.0):
        state = torch.tensor(state, device=self.device, dtype=torch.float32) if type(state) == np.ndarray else state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state)
        q_values, continuous_action = action[:, :self.discrete_dim], action[:, self.discrete_dim:]
        if random.random() > epsilon:
            tgt_action = q_values.argmax(dim=1).unsqueeze(1)
        else:
            tgt_action = torch.randint(self.discrete_dim, size=(state.shape[0],1))
        action_mask = (tgt_action != 0).expand(-1, 2)
        continuous_action = continuous_action.clone()
        continuous_action[action_mask] = 0
        env_action = torch.cat((tgt_action, continuous_action), dim=1)
        lp_action = torch.cat((q_values, continuous_action), dim=1)
        return torch.squeeze(env_action).detach().cpu().numpy(), lp_action
    
    def act_target(self, state, epsilon=0.0):
        state = torch.tensor(state, device=self.device, dtype=torch.float32) if type(state) == np.ndarray else state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.target_actor(state)
        q_values, continuous_action = action[:, :self.discrete_dim], action[:, self.discrete_dim:]
        if random.random() > epsilon:
            tgt_action = q_values.argmax(dim=1).unsqueeze(1)
        else:
            tgt_action = torch.randint(self.discrete_dim, size=(state.shape[0],1))
        action_mask = (tgt_action != 0).expand(-1, 2)
        continuous_action = continuous_action.clone()
        continuous_action[action_mask] = 0
        env_action = torch.cat((tgt_action, continuous_action), dim=1)
        lp_action = torch.cat((q_values, continuous_action), dim=1)
        return torch.squeeze(env_action).detach().cpu().numpy(), lp_action

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        next_state_batch = torch.cat(batch.next_state)
        
        self.update(state=state_batch, action=action_batch, reward=reward_batch, done=done_batch, next_state=next_state_batch)

    def update(self, state, action, reward, next_state, done, episode=0):
        
        # Critic update
        wait_step = 5000
        
        with torch.no_grad():
            # discrete_qs, continuous = self.target_actor(state)
            # next_state_action = torch.cat((next_state, discrete_qs, continuous), dim=1)
            _, next_actions = self.act_target(next_state)
            next_state_action = torch.cat((next_state, next_actions), dim=1)
            target_q = reward.unsqueeze(1) + self.gamma * (1 - done).unsqueeze(1) * self.target_critic(next_state_action)
        
        state_action = torch.cat((state, action), dim=1)
        q = self.critic(state_action.detach())
        critic_loss = nn.MSELoss()(q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # if episode < wait_step:
        #     return

        # Actor update
        _, actions = self.act(state)
        state_action = torch.cat((state, actions), dim=1)
        actor_loss = -self.critic(state_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self.update_target_soft()


def train(num_bits=10, num_epochs=10, hindsight_replay=True,
          eps_max=0.2, eps_min=0.0, exploration_fraction=0.5):

    """
    Training loop for the bit flip experiment introduced in https://arxiv.org/pdf/1707.01495.pdf using DQN or DQN with
    hindsight experience replay. Exploration is decayed linearly from eps_max to eps_min over a fraction of the total
    number of epochs according to the parameter exploration_fraction. Returns a list of the success rates over the
    epochs.
    """

    # Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
    future_k = 4
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40
    max_steps = 20

    # env = BitFlipEnvironment(num_bits)
    num_agents = 1
    env = PirateEnv(num_agents=num_agents, max_steps=max_steps)

    state_dim = env.observation_space[0].shape[0]
    goal_dim = env.goal_space[0].shape[0]
    discrete_dim = 3
    continuous_dim = 2
    agent = PDDPGAgent(state_dim+goal_dim, discrete_dim, continuous_dim)

    # state, _ = env.reset()
    # done = False
    # while not done:
    #     random_actions = env.action_space[0].sample()  # Take random actions
    #     state, rewards, done, _ = env.step([(random_actions[0], random_actions[1][0], random_actions[1][1])])
    #     env.capture_frame()  # Store frames for playback

    # Now we have frames to show before training
    # env.show_video(title="Before training")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    success_rate = 0.0
    success_rates = []
    for epoch in range(num_epochs):

        # Decay epsilon linearly from eps_max to eps_min
        eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)
        print("Epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * eps, success_rate))

        successes = 0
        for cycle in range(num_cycles):

            for episode in range(num_episodes):

                # Run episode and cache trajectory
                episode_trajectory = []
                state, goal = env.reset()

                state = torch.tensor(state, dtype=torch.float32)
                goal = torch.tensor(goal, dtype=torch.float32)

                for step in range(max_steps):

                    state_ = torch.cat((state, goal), dim=-1)
                    action, lp_action = agent.act(state_, eps)
                    next_state, reward, done, _ = env.step(action) # Do this because environment is single agent (for now)

                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    reward = torch.tensor(reward, dtype=torch.float32)
                    done = torch.tensor(done, dtype=torch.float32)

                    episode_trajectory.append(Experience(state, lp_action, reward, next_state, done))
                    state = next_state
                    if reward > 0:
                        successes += 1
                        break

                # Fill up replay memory
                steps_taken = step
                for t in range(steps_taken):

                    # Standard experience replay
                    state, action, reward, next_state, done = episode_trajectory[t]
                    state_, next_state_ = torch.cat((state, goal), dim=-1), torch.cat((next_state, goal), dim=-1)
                    agent.push_experience(state_, action, reward, next_state_, done)

                    # Hindsight experience replay
                    if hindsight_replay:
                        for _ in range(future_k):
                            future = random.randint(t, steps_taken)  # index of future time step
                            new_goal = torch.cat((episode_trajectory[future].state[:,0:2], torch.zeros((num_agents, 1))), dim=1)  # take future next_state and set as goal
                            new_reward, new_done = env.compute_reward(state, action, new_goal)
                            
                            new_reward = torch.tensor(new_reward, dtype=torch.float32)
                            new_done = torch.tensor(new_done, dtype=torch.float32)
                            
                            state_, next_state_ = torch.cat((state, new_goal), dim=1), torch.cat((next_state, new_goal), dim=1)
                            agent.push_experience(state_, action, new_reward, next_state_, new_done)
                            # if new_reward > 0:
                            #     print(next_state, new_goal)

            # Optimize DQN
            # for opt_step in range(num_opt_steps):
            #     agent.optimize_model()
            agent.optimize_model()
            agent.update_target_hard()

        success_rate = successes / (num_episodes * num_cycles)
        success_rates.append(success_rate)

    env.show_video(title="After Training")
    env.generate_video("pirate_rollout.mp4")

    return success_rates


if __name__ == "__main__":
    bits = 50  # more than 10^15 states
    epochs = 40


    for her in [True, False]:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        success = train(bits, epochs, her)
        plt.plot(success, label="HER-DQN" if her else "DQN")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title("Number of bits: {}".format(bits))
    plt.savefig("{}_bits.png".format(bits), dpi=1000)
    plt.show()
