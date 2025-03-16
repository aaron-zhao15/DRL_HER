# DRL_HER

This is an implementation of Hindsight Experience Replay (HER) in the BitFlip environment. It is based on the implementation by Alex Hermansson in the following repository: https://github.com/AlexHermansson/hindsight-experience-replay

## Requirements

To run the code, you'll need the following dependencies:

- Python >= 3.5 (Python 3.8+ preferred)
- `torch`
- `matplotlib`
- `seaborn`
- `numpy`
- `random`
- `collections`

You can install the required packages using `pip`:

```bash
pip install torch matplotlib seaborn numpy
```

To run the training loop, simply execute:

```bash
python bitflip.py
```

By default, this will run two training loops, first with HER and second without HER. Then it will display the two training curves.


### Key Concepts

- **State Representation**: The state is a tensor of bits, each of which is either 0 or 1. The goal is to transform this state into another random bit string (goal) by flipping individual bits.
  
- **Actions**: The action is simply an index corresponding to the bit that will be flipped in the state.
  
- **Reward Function**: The reward is `1` when the agent's state matches the goal. If the states do not match, a small negative reward of `-0.1` is given to encourage efficient progress towards the goal.

### Bit Flip Environment Flow

1. **Initialization**: The environment starts with a random state and goal.
2. **Reset**: The environment can be reset, which re-randomizes the state and goal.
3. **Action**: The agent selects a bit (by its index) to flip, changing the state.
4. **Reward**: The reward is computed based on the proximity of the state to the goal.
5. **Completion**: The environment is considered "done" when the agent's state matches the goal.


