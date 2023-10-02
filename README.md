# Reinforcement Learning Snake Game

This repository showcases a reinforcement learning approach to mastering the classic Snake game. The game environment is built using the Pygame library, and the agent is trained using Q-learning with a neural network model built on PyTorch.

## Structure of the Repository:

- **agent.py**: Defines the reinforcement learning agent responsible for training and playing the Snake game. The agent utilizes a neural network model and Q-learning to make decisions.
  
- **library.py**: A collection of utility functions, including:
  - Plotting functions to visualize the agent's performance.
  - Functions to ensure the required directories exist.

- **model.py**: Describes the neural network architecture (`Linear_QNet`) that the agent uses to estimate Q-values. It also contains the `QTrainer` class which handles the training of the model.

- **snake.py**: Contains the game logic and visuals for the Snake game. This file uses Pygame to render the game and handle its mechanics.
