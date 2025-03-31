import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import cv2
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2

class ContinuousMovableObject:
    def __init__(self, x, y, grid_size, speed=1, occupancy_radius=1):
        self.position = np.array([x, y])
        self.grid_size = grid_size
        self.occupancy_radius = occupancy_radius
        self.speed = speed

    def move(self, movement, other_agents=None):
        new_position = self.position + self.speed * np.array(movement)
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        
        # for other_agent in other_agents:
        #     distance = np.linalg.norm(new_position - other_agent.position)
        #     if distance < self.occupancy_radius + other_agent.occupancy_radius:
        #         return  # Collision detected, cancel the move

        self.position = new_position

class PirateEnv(gym.Env):
    def __init__(self, num_agents=3, grid_size=20, capture_distance=3, occupancy_radius=1, max_steps=1000):
        super(PirateEnv, self).__init__()

        self.num_agents = num_agents
        self.n = num_agents
        self.num_targets = num_agents
        self.grid_size = grid_size
        self.capture_distance = capture_distance
        self.max_steps = max_steps
        self.steps = 0
        self.occupancy_radius = occupancy_radius
        self.capture_limit = 10

        self.action_space = [spaces.Tuple((
            spaces.Discrete(2),  # Action type: 0 (move), 1 (capture)
            spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]))
        )) for _ in range(self.num_agents)]

        self.observation_space = [spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(4,),
        ) for _ in range(self.num_agents)]

        self.goal_space = [spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(3,),
        ) for _ in range(self.num_agents)]

        self.reset()

    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)

        self.agents = [ContinuousMovableObject(
            np.random.uniform(0, self.grid_size),
            np.random.uniform(0, self.grid_size),
            self.grid_size,
            speed=0.5 * i + 0.5,
            occupancy_radius=self.occupancy_radius
        ) for i in range(self.num_agents)]

        self.targets = [ContinuousMovableObject(
            np.random.uniform(0, self.grid_size),
            np.random.uniform(0, self.grid_size),
            self.grid_size
        ) for _ in range(self.num_targets)]

        self.targets_disabled = [0] * self.num_targets
        self.agent_capture_count = [0] * self.num_agents
        self.steps = 0

        self.frames = []

        states, goals = self.observations(), self.goals()

        # return np.concat((states, goals), axis=1)
        return states, goals


    def step(self, actions):
        self.steps += 1
        dones = np.zeros(self.num_agents)
        rewards = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            discrete_action, continuous_action = action[0], action[1:3]
            moving = (discrete_action == 0)
            capturing = (discrete_action == 1)
            if moving:  # Move action
                other_agents = [agent for j, agent in enumerate(self.agents) if j != i]
                # theta = action[1]
                movement = np.clip(continuous_action, -1, 1)
                self.agents[i].move(movement, other_agents)
            
            if capturing and self.agent_capture_count[i] < self.capture_limit:
                target = self.targets[i]
                distance = np.linalg.norm(self.agents[i].position - target.position)
                if distance <= self.capture_distance and not self.targets_disabled[i]:
                    self.targets_disabled[i] = 1
                self.agent_capture_count[i] += 1
        
        for i, target in enumerate(self.targets):
            if not self.targets_disabled[i]:
                target.move(np.random.uniform(-1, 1, size=2), self.agents)

        states, goals = self.observations(), self.goals()
        rewards, dones = self.compute_reward(states, actions, goals)
        for i in range(self.num_agents):
            if self.agent_capture_count[i] >= self.capture_limit:
                dones[i] = 1

        if all(self.targets_disabled):
            dones = np.ones_like(dones)

        if self.steps >= self.max_steps:
            dones = np.ones_like(dones)

        self.capture_frame()
        # states = np.concat((states, goals), axis=1)

        return states, rewards, dones, goals
    
    def compute_reward(self, state, action, goal):
        agent_pos = state[:, 0:2]
        capture_count = state[:, 2]
        capture_threshold = state[:, 3]

        agent_capturing = action[:, 0]

        target_pos = goal[:, 0:2]
        target_disabled = goal[:, 2]

        agent_target_distance = np.linalg.norm(agent_pos - target_pos, axis=1)
        rewards, dones = np.zeros_like(agent_target_distance), np.zeros_like(agent_target_distance)
        for i in range(rewards.shape[0]):
            if agent_target_distance[i] < capture_threshold[i] and agent_capturing[i] and capture_count[i] < 1 and target_disabled[i] == 0:
                rewards[i] = 1
                dones[i] = 1
            # elif agent_capturing[i]:
            #     rewards[i] = -1
            # elif agent_target_distance[i] > capture_threshold:
            #     rewards[i] = -agent_target_distance[i]/self.grid_size
        return rewards, dones


    def dist_rewards(self):
        return np.array([-np.linalg.norm(agent.position - target.position) for agent, target in zip(self.agents, self.targets)])

    def observations(self):
        obs = [[] for _ in self.agents]
        for agent_id, agent in enumerate(self.agents):
            obs[agent_id].extend(agent.position)
            obs[agent_id].append(self.agent_capture_count[agent_id]/self.capture_limit)
            obs[agent_id].append(self.capture_distance)

        return np.array(obs)
    
    def goals(self):
        goals = [[] for _ in self.agents]
        for agent_id, agent in enumerate(self.agents):
            goals[agent_id].extend(self.targets[agent_id].position)
            goals[agent_id].append(self.targets_disabled[agent_id])
        return np.array(goals)
    
    def render(self, mode='human'):
        canvas_size = 500
        scale = canvas_size / self.grid_size
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            start_point = (int(i * scale), 0)
            end_point = (int(i * scale), canvas_size)
            cv2.line(canvas, start_point, end_point, (200, 200, 200), 1)  # Light gray grid lines
            # Horizontal lines
            start_point = (0, int(i * scale))
            end_point = (canvas_size, int(i * scale))
            cv2.line(canvas, start_point, end_point, (200, 200, 200), 1)  # Light gray grid lines

        # Draw agents with continuous positions
        for i, agent in enumerate(self.agents):
            green_intensity = max(0, 144 - int((self.agent_capture_count[i] / 10) * 144))
            color = (0, green_intensity, 0)
            agent_pos = np.clip(agent.position, 0, self.grid_size - 1) * scale
            center = tuple(agent_pos.astype(int))
            cv2.circle(canvas, center, int(scale // 3), color, -1)

        # Draw the target with continuous position
        for j, target in enumerate(self.targets):
            target_color = (0, 0, 255) if not self.targets_disabled[j] else (128, 128, 128)
            # Scale continuous positions for rendering
            target_pos = np.clip(target.position, 0, self.grid_size - 1) * scale
            target_center = tuple(target_pos.astype(int))
            cv2.circle(canvas, target_center, int(scale // 3), target_color, -1)

        return canvas
    
    def save_state(self):
        """
        Returns the current state as a dictionary.
        """
        state = {
            'agents': [(agent.position, agent.speed) for agent in self.agents],
            'targets': [(target.position, target.speed) for target in self.targets],
            'targets_disabled': self.targets_disabled,
            'agent_launched': self.agent_launched,
            'steps': self.steps
        }
        return state

    def load_state(self, state):
        """
        Loads the state from a dictionary.
        """
        self.agents = [
            ContinuousMovableObject(pos[0], pos[1], self.grid_size, speed=speed)
            for pos, speed in state['agents']
        ]
        self.targets = [
            ContinuousMovableObject(pos[0], pos[1], self.grid_size, speed=speed)
            for pos, speed in state['targets']
        ]
        self.targets_disabled = state['targets_disabled']
        self.agent_launched = state['agent_launched']
        self.steps = state['steps']

    def capture_frame(self):
        """ Store the current environment state as an image for playback. """
        frame = self.render()  # Get the current frame from render()
        self.frames.append(frame)

    def show_video(self, title="Environment Rollout", fps=10):
        """ Play an animation of the captured trajectory using OpenCV. """
        if not self.frames:
            print("No frames to show!")
            return
        
        for frame in self.frames:
            cv2.imshow(title, frame)  
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()

    def generate_video(self, filename="pirate_rollout.mp4", fps=10):
        """ Save the captured rollout as an MP4 video using OpenCV. """
        if not self.frames:
            print("No frames to save!")
            return

        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in self.frames:
            out.write(frame)  # Write frame to video

        out.release()
        print(f"Video saved as {filename}")

    def close(self):
        cv2.destroyAllWindows()

def run_environment(env, num_steps=20):
    for _ in range(num_steps):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        env.step(actions)
        env.render()

from matplotlib.animation import FuncAnimation
def run_and_visualize(env, agents):
    frames = []

    # Initialize state and pre-allocate frame storage if possible
    state = env.reset(seed=1)

    # Step through the environment and collect frames
    for _ in range(env.max_steps):
        # Optimize action computation
        actions = []
        for i, agent in enumerate(agents):
            action, _ = agent.act(state[i])
            actions.append(action)
        
        state, rewards, done, _ = env.step(actions)

        # Collect frame only if rendering is enabled
        frame = env.render()
        frames.append(frame)
        if done.all():
            break

    # Create a video and visualize it in the Jupyter notebook cell
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')  # Turn off axis for better visualization
    ax_img = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))  # Display the first frame immediately

    def update_frame(i):
        ax_img.set_data(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        return [ax_img]

    # Use FuncAnimation to create an animation with a lower interval
    anim = FuncAnimation(fig, update_frame, frames=len(frames), interval=100, repeat=False)
    plt.close(fig)  # Prevent duplicate static display of the plot

    # Display the animation in the notebook
    from IPython.display import HTML
    return HTML(anim.to_jshtml())
