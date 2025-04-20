import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
from collections import namedtuple
import copy

import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from pirate_env import PirateEnv

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
log_dir = f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=log_dir)
bc_log = []
bc_log_path = os.path.join(log_dir, "bc_training_log.json")
training_log = []
json_log_path = os.path.join(log_dir, "training_log.json")


class Actor(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dim, hidden_dim=64, device="cpu"):
        super(Actor, self).__init__()
        latent_dim = 32
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim, device=device),
            nn.LeakyReLU(),
        )
        self.discrete_layer = nn.Linear(latent_dim, discrete_dim)
        self.continuous_layer = nn.Linear(latent_dim, continuous_dim)
        self.squash = nn.Tanh()
        # should be decoding Q values

        self.continuous_passthrough_layer = nn.Linear(state_dim, continuous_dim)
        nn.init.zeros_(self.continuous_passthrough_layer.weight)
        nn.init.zeros_(self.continuous_passthrough_layer.bias)
        self.continuous_passthrough_layer.weight.requires_grad = False
        self.continuous_passthrough_layer.bias.requires_grad = False

    def forward(self, x):
        latent = self.model(x)
        discrete_qs = self.discrete_layer(latent)
        continuous = self.squash(self.continuous_layer(latent) + self.continuous_passthrough_layer(x))
        return discrete_qs, continuous

# Collect dataset
def collect_data_uniform(env, n_data=70000):
    state_pos = np.random.uniform(0, env.grid_size, (n_data, 2))
    state_cap_count = np.random.uniform(0, 1, (n_data, 1))
    state_cap_range = 3*np.ones((n_data, 1))
    # state_cap_range = np.random.uniform(0, env.grid_size, (n_data, 1))
    states = np.concatenate((state_pos, state_cap_count, state_cap_range), axis=1)

    goal_pos = np.random.uniform(0, env.grid_size, (n_data, 2))
    goal_captured = np.random.randint(2, size=(n_data, 1))
    goals = np.concatenate((goal_pos, goal_captured), axis=1)

    # labels = Heuristic.determine_capture(states, goals)
    # labels = Heuristic.determine_move(states, goals)
    labels = Heuristic.determine_move_and_capture(states, goals)
    states = np.concatenate((states, goals), axis=1)

    return states.astype(np.float32), labels.astype(np.float32)

def collect_data_from_env(env, num_episodes=1000):
    state_list, label_list = [], []
    
    for _ in range(num_episodes):
        state, goal = env.reset()
        done = np.zeros(env.num_agents)
        
        while not done.all():
            actions = [env.action_space[i].sample() for i in range(env.num_agents)]
            actions = np.array([[actions[i][0], actions[i][1][0], actions[i][1][1]] for i in range(env.num_agents)])
            next_state, rewards, done, goal = env.step(actions)
            
            # label = Heuristic.determine_capture(state, goal)
            # label = Heuristic.determine_move(state, goal)
            label = Heuristic.determine_move_and_capture(state, goal)
            state_ = np.concatenate((state, goal), axis=1)

            for i in range(state_.shape[0]):
                state_list.append(state_[i].astype(np.float32))
                label_list.append(label[i].astype(np.float32))
                
            state = next_state
    
    return np.array(state_list), np.array(label_list)

def adjust_label_distribution(X, y, p=0.5):
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    min_label = min(idx_0.shape[0], idx_1.shape[0])
    num_1 = int(p * min_label)
    num_0 = min_label - num_1
    
    chosen_1s = np.random.choice(idx_1, num_1, replace=False)
    chosen_0s = np.random.choice(idx_0, num_0, replace=False)
    selected_indices = np.concatenate((chosen_1s, chosen_0s))
    np.random.shuffle(selected_indices)

    X_new = X[selected_indices]
    y_new = y[selected_indices]
    return X_new, y_new

class Heuristic:
    @staticmethod
    def move_act(state):
        agent_pos = state[:, 0:2]
        goal_pos = state[:, 4:6]
        direction = goal_pos - agent_pos
        direction = direction/np.linalg.norm(direction, axis=1, keepdims=True)
        return direction
    
    @staticmethod
    def capture_act(state):
        state_pos = state[:, 0:2]
        state_cap_count = state[:, 2]
        state_cap_range = state[:, 3]

        goal_pos = state[:, 4:6]
        goal_captured = state[:, 6]

        agent_goal_dist = np.linalg.norm(state_pos-goal_pos, axis=1)
        dist_capturable = agent_goal_dist < state_cap_range
        resource_capturable = state_cap_count < 1
        active_capturable = 1-goal_captured

        heuristic_capture = dist_capturable * resource_capturable * active_capturable

        return heuristic_capture[:, None]
    
    @staticmethod
    def determine_capture(state, goal):
        state_pos = state[:, 0:2]
        state_cap_count = state[:, 2]
        state_cap_range = state[:, 3]

        goal_pos = goal[:, 0:2]
        goal_captured = goal[:, 2]

        agent_goal_dist = np.linalg.norm(state_pos-goal_pos, axis=1)
        dist_capturable = agent_goal_dist < state_cap_range
        resource_capturable = state_cap_count < 1
        active_capturable = 1-goal_captured

        heuristic_capture = dist_capturable * resource_capturable * active_capturable

        return heuristic_capture
    
    @staticmethod
    def determine_move(state, goal):
        state_pos = state[:, 0:2]
        goal_pos = goal[:, 0:2]
        direction = goal_pos - state_pos
        direction = direction/np.linalg.norm(direction, axis=1, keepdims=True)
        return direction
    
    @staticmethod
    def determine_move_and_capture(state, goal):
        movement = Heuristic.determine_move(state, goal)
        capture = Heuristic.determine_capture(state, goal)[:, None]
        return np.concatenate((capture, movement), axis=1)
    
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    TP, TN, FP, FN = 0, 0, 0, 0  # Initialize confusion matrix values

    with torch.no_grad():
        for batch_states, batch_labels in test_loader:
            batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)

            outputs = model(batch_states).squeeze()
            loss = criterion(outputs, batch_labels)

            test_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            # Compute TP, TN, FP, FN
            TP += ((preds == 1) & (batch_labels == 1)).sum().item()
            TN += ((preds == 0) & (batch_labels == 0)).sum().item()
            FP += ((preds == 1) & (batch_labels == 0)).sum().item()
            FN += ((preds == 0) & (batch_labels == 1)).sum().item()

    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity / Recall
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0  # False Positive Rate
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False Negative Rate

    return avg_test_loss, accuracy, TPR, TNR, FPR, FNR

def plot_label_distribution(labels):
    plt.figure(figsize=(6, 4))
    plt.hist(labels, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color=['blue'])
    plt.xticks([0, 1])
    plt.xlabel("Label (0: Negative, 1: Positive)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Labels sampled Uniformly")
    plt.show()

# Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = PirateEnv(num_agents=1)

# Collect dataset
X, y = collect_data_from_env(env)
# X, y = adjust_label_distribution(X, y, p=0.5)

train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_data, test_data = random_split(TensorDataset(torch.tensor(X), torch.tensor(y)), [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Call function to visualize label distribution
# plot_label_distribution(y)
# print("Percentage of labels that are 0:", 1-(np.sum(y)/len(y)))

# Convert to PyTorch tensors
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
# model = StateClassifier(input_dim=env.observation_space[0].shape[0]+env.goal_space[0].shape[0]).to(device)
# model = StateMover(input_dim=env.observation_space[0].shape[0]+env.goal_space[0].shape[0]).to(device)
model = Actor(state_dim=env.observation_space[0].shape[0]+env.goal_space[0].shape[0], discrete_dim=2, continuous_dim=2).to(device)
disc_criterion, cont_criterion = nn.CrossEntropyLoss(), nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
# train(model, train_loader, test_loader, optimizer, criterion, num_epochs=20)
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_states, batch_labels in dataloader:
        batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        disc_out, cont_out = model(batch_states)
        loss = disc_criterion(disc_out, batch_labels[:, 0].long()) + cont_criterion(cont_out, batch_labels[:, 1:3])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        preds = disc_out.argmax(dim=1).float()
        correct += (preds == batch_labels[:, 0]).sum().item()
        total += batch_labels.size(0)

    accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    writer.add_scalar("BC/Loss", total_loss, epoch)
    writer.add_scalar("BC/Accuracy", accuracy, epoch)

    epoch_data = {
        "epoch": epoch + 1,
        "loss": total_loss,
        "accuracy": accuracy,
    }
    bc_log.append(epoch_data)
    with open(bc_log_path, "w") as f:
        json.dump(bc_log, f, indent=4)

state, goal = env.reset(seed=1)
successes = 0
for _ in range(64):
    env.reset()
    for step in range(env.max_steps):
        state_ = np.concatenate((state, goal), axis=1)

        # Get heuristic-based move action
        # move_action = torch.tensor(Heuristic.move_act(state_), dtype=torch.float32).to(device)
        # capture_action = Heuristic.capture_act(state_)
        
        # disc_out = model(torch.tensor(state_, dtype=torch.float32))
        disc_out, cont_out = model(torch.tensor(state_, dtype=torch.float32))
        move_action = cont_out.detach().numpy()
        capture_action = disc_out.argmax(dim=1).unsqueeze(1)
        # Combine actions
        action = np.concatenate((capture_action, move_action), axis=1)

        # Step environment
        next_state, reward, done, goal = env.step(action)

        # Convert to tensors
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        state = next_state
        if torch.sum(reward) > 0:
            successes += 1

        if done.all():
            break
print(successes/64)
# Play and save rollout video
# env.show_video(fps=10)  # Show the rollout
env.generate_video(f"{log_dir}/pirate_rollout.mp4", fps=10)  # Save as a video file

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

class PDDPGAgent:
    def __init__(self, state_dim, discrete_dim=1, continuous_dim=2, device="cpu", actor=None):
        self.device = device
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        self.gamma = 0.99
        self.batch_size = 128
        self.train_start = 1000
        self.tau = 0.01

        self.memory = ReplayMemory(int(1e6))

        if actor:
            self.actor = actor
        else:
            self.actor = Actor(state_dim, discrete_dim, continuous_dim, device=device)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, discrete_dim, continuous_dim, device=device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-5)

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
        # action = self.actor(state)
        # q_values, continuous_action = action[:, :self.discrete_dim], action[:, self.discrete_dim:]
        q_values, continuous_action = self.actor(state)
        epsilon=0
        if random.random() > epsilon:
            tgt_action = q_values.argmax(dim=1).unsqueeze(1)
        else:
            tgt_action = torch.randint(self.discrete_dim, size=(state.shape[0],1))
        action_mask = (tgt_action != 0).expand(-1, 2)
        continuous_action = torch.where(action_mask, torch.zeros_like(continuous_action), continuous_action)
        env_action = torch.cat((tgt_action, continuous_action), dim=1)
        lp_action = torch.cat((q_values, continuous_action), dim=1)
        return env_action.detach().cpu().numpy(), lp_action
    
    def act_target(self, state, epsilon=0.0):
        state = torch.tensor(state, device=self.device, dtype=torch.float32) if type(state) == np.ndarray else state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # action = self.target_actor(state)
        # q_values, continuous_action = action[:, :self.discrete_dim], action[:, self.discrete_dim:]
        q_values, continuous_action = self.target_actor(state)
        epsilon=0
        if random.random() > epsilon:
            tgt_action = q_values.argmax(dim=1).unsqueeze(1)
        else:
            tgt_action = torch.randint(self.discrete_dim, size=(state.shape[0],1))
        action_mask = (tgt_action != 0).expand(-1, 2)
        continuous_action = torch.where(action_mask, torch.zeros_like(continuous_action), continuous_action)
        env_action = torch.cat((tgt_action, continuous_action), dim=1)
        lp_action = torch.cat((q_values, continuous_action), dim=1)
        return env_action.detach().cpu().numpy(), lp_action

    def optimize_model(self, episode=0):
        if len(self.memory) < self.train_start:
            return 0, 0

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        next_state_batch = torch.cat(batch.next_state)
        
        return self.update(state=state_batch, action=action_batch, reward=reward_batch, done=done_batch, next_state=next_state_batch, episode=episode)

    def update(self, state, action, reward, next_state, done, episode=0):
        
        # Critic update
        actor_wait_step = 100
        
        with torch.no_grad():
            # discrete_qs, continuous = self.target_actor(state)
            # next_state_action = torch.cat((next_state, discrete_qs, continuous), dim=1)
            _, next_actions = self.act_target(next_state)
            next_state_action = torch.cat((next_state, next_actions), dim=1)
            target_q = reward.unsqueeze(1) + self.gamma * (1 - done).unsqueeze(1) * self.target_critic(next_state_action)
        
        state_action = torch.cat((state, action), dim=1)
        q = self.critic(state_action.detach())
        critic_loss = nn.MSELoss()(q, target_q.detach())

        # if episode < actor_wait_step:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        _, actions = self.act(state)
        state_action = torch.cat((state, actions), dim=1)
        actor_loss = -self.target_critic(state_action).mean()

        if episode > actor_wait_step:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Soft update of target networks
        self.update_target_soft()

        return critic_loss.item(), actor_loss.item()


hindsight_replay=False
num_epochs=4000
eps_max=0.2
eps_min=0.0
exploration_fraction=0.5
visualize_interval = 50

"""
Training loop for the bit flip experiment introduced in https://arxiv.org/pdf/1707.01495.pdf using DQN or DQN with
hindsight experience replay. Exploration is decayed linearly from eps_max to eps_min over a fraction of the total
number of epochs according to the parameter exploration_fraction. Returns a list of the success rates over the
epochs.
"""

# Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
future_k = 8
num_cycles = 1
num_episodes = 16
num_opt_steps = 40
max_steps = 200
experiences_per_epoch = 5000

# env = BitFlipEnvironment(num_bits)
num_agents = 1
env = PirateEnv(num_agents=num_agents, max_steps=max_steps)

state_dim = env.observation_space[0].shape[0]
goal_dim = env.goal_space[0].shape[0]
discrete_dim = 2
continuous_dim = 2
agent = PDDPGAgent(state_dim+goal_dim, discrete_dim, continuous_dim, actor=model)
# agent = DQNAgent(state_dim+goal_dim, action_dim=10)
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
    for episode in range(num_episodes):

        # Run episode and cache trajectory
        episode_trajectory = []
        state, goal = env.reset()

        state = torch.tensor(state, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        for step in range(max_steps):

            state_ = torch.cat((state, goal), dim=-1)
            action, lp_action = agent.act(state_, eps)
            next_state, reward, done, goal = env.step(action) # Do this because environment is single agent (for now)

            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
            goal = torch.tensor(goal, dtype=torch.float32)

            episode_trajectory.append(Experience(state, lp_action, reward, next_state, done))
            state = next_state
            if reward > 0:
                successes += 1
                break
            if done.all():
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
                # modified with G as final state
                # new_goal = torch.cat((episode_trajectory[steps_taken].state[:, 0:2], torch.zeros((num_agents, 1))), dim=1)
                # new_reward, new_done = env.compute_reward(state, action, new_goal)
                # new_reward = torch.tensor(new_reward, dtype=torch.float32)
                # new_done = torch.tensor(new_done, dtype=torch.float32)
                # state_, next_state_ = torch.cat((state, new_goal), dim=1), torch.cat((next_state, new_goal), dim=1)
                # agent.push_experience(state_, action, new_reward, next_state_, new_done)
                for _ in range(future_k):

                    future = random.randint(t, steps_taken)  # index of future time step
                    # new_goal = torch.cat((torch.randint(0, env.grid_size, size=(num_agents, 2)), torch.zeros((num_agents, 1))), dim=1)
                    new_goal = torch.cat((episode_trajectory[future].state[:,0:2], torch.zeros((num_agents, 1))), dim=1)  # take future next_state and set as goal
                    new_reward, new_done = env.compute_reward(state, action, new_goal)
                    
                    new_reward = torch.tensor(new_reward, dtype=torch.float32)
                    new_done = torch.tensor(new_done, dtype=torch.float32)
                    
                    state_, next_state_ = torch.cat((state, new_goal), dim=1), torch.cat((next_state, new_goal), dim=1)
                    agent.push_experience(state_, action, new_reward, next_state_, new_done)
                    # if new_reward > 0:
                    #     print(next_state, new_goal)

    # Optimize
    for opt_step in range(num_opt_steps-1):
        agent.optimize_model(episode=epoch)
    critic_loss, actor_loss = agent.optimize_model(episode=epoch)
    # actor_loss = agent.optimize_model(episode=epoch)
    agent.update_target_soft()

    success_rate = successes / num_episodes
    success_rates.append(success_rate)

    writer.add_scalar("Loss/Critic", critic_loss, epoch)
    writer.add_scalar("Loss/Actor", actor_loss, epoch)
    writer.add_scalar("Success Rate", success_rate, epoch)

    epoch_data = {
        "epoch": epoch + 1,
        "epsilon": eps,
        "success_rate": success_rate,
        "critic_loss": critic_loss.item() if torch.is_tensor(critic_loss) else critic_loss,
        "actor_loss": actor_loss.item() if torch.is_tensor(actor_loss) else actor_loss,
    }

    training_log.append(epoch_data)

    # Write to file (overwrite each time to keep a clean log)
    with open(json_log_path, "w") as f:
        json.dump(training_log, f, indent=4)

    if epoch % visualize_interval == 0:
        env.generate_video(f"{log_dir}/rollout_{epoch}.mp4")

# env.show_video(title="After Training")
# env.generate_video(f"{log_dir}/pirate_rollout_{hindsight_replay}.mp4")

print(success_rates)

