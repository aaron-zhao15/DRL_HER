

# DRL_HER

This is an implementation of **Hindsight Experience Replay (HER)** in a custom gridworld-like environment called `PirateEnv`, along with behavior cloning (BC) pretraining and a continuous + discrete hybrid actor-critic architecture using PyTorch.

It extends [Alex Hermansson’s HER in BitFlip](https://github.com/AlexHermansson/hindsight-experience-replay) and adapts it for continuous action environments.


## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/aaron-zhao15/DRL_HER.git
cd DRL_HER
```

### Step 2: Setup Environment

Ensure `conda` is installed, then run:

```bash
conda env create -f environment.yml
conda activate HER-env
```

> ⚠️ If you're on macOS and encounter issues with specific `torch` versions, check compatibility with your Python and system architecture.

## Running the Code

### Pretrain the Actor using Behavior Cloning

```bash
python bc_rl_finetune.py
```

- Trains the actor to mimic a heuristic.
- Logs loss/accuracy to TensorBoard.
- Generates a rollout video at the end.


### 🔧 Custom Configuration Instructions

To enable or tweak **Hindsight Experience Replay (HER)** and other training parameters, open the file:

```bash
bc_rl_finetune.py
```

Scroll to line 484 in bc_rl_finetune.py, you can these these code:

```python
"""
hindsight_replay=True
num_epochs=4000
eps_max=0.2
eps_min=0.0
exploration_fraction=0.5

future_k = 8
num_cycles = 1
num_episodes = 16
num_opt_steps = 40
max_steps = 200
experiences_per_epoch = 5000

# env = BitFlipEnvironment(num_bits)
num_agents = 1
"""
```

Update the following variables as needed:

| Variable | Description |
|----------|-------------|
| `hindsight_replay=True` | Set to `True` to enable HER |
| `num_epochs=4000` | Total number of training epochs |
| `eps_max=0.2` | Initial exploration rate (epsilon-greedy) |
| `eps_min=0.0` | Final exploration rate |
| `exploration_fraction=0.5` | Fraction of epochs used for epsilon decay |
| `future_k=8` | Number of future goals sampled in HER |
| `num_episodes=16` | Number of episodes per epoch |
| `num_opt_steps=40` | Gradient updates per epoch |
| `max_steps=200` | Max steps per episode |
| `num_agents=1` | Number of agents in the environment |

Once updated, rerun:

```bash
python bc_rl_finetune.py
```


### HER Training with PDDPG

HER and actor-critic fine-tuning will automatically follow after the behavior cloning pretraining within the same script.

```bash
# From inside bc_rl_finetune.py (runs after BC phase)
# Trains agent using PDDPG + HER
# Generates a rollout video for each epoch
```

## Logs and Videos

- TensorBoard logs saved to: `runs/YYYY-MM-DD_HH-MM-SS`
- Rollout videos saved per epoch: `runs/YYYY-MM-DD_HH-MM-SS/rollout_*.mp4`

To view logs:

```bash
tensorboard --logdir runs/
```

## Custom Environment: PirateEnv

The `PirateEnv` is a multi-agent grid world environment where each agent tries to:
- **Move** toward a goal
- **Capture** the goal under specific constraints

The environment provides:
- `observation_space`, `goal_space`
- `reset()`, `step()`, `compute_reward()`
- video generation via `generate_video()`

## File Overview

| File | Description |
|------|-------------|
| `bc_rl_finetune.py` | Main training script: behavior cloning, HER training, logging |
| `pirate_env.py` | Custom Gym-like environment for multi-agent task |
| `environment.yml` | Conda environment setup file |
| `runs/` | Auto-generated log and video directory |

## Acknowledgements

- [OpenAI HER Paper (Andrychowicz et al., 2017)](https://arxiv.org/pdf/1707.01495.pdf)
- [Alex Hermansson’s HER BitFlip Repo](https://github.com/AlexHermansson/hindsight-experience-replay)
- PyTorch + Stable Baselines3 inspiration

