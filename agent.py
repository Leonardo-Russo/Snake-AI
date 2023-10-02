import torch
import random
import numpy as np
from collections import deque
import os
from IPython import display
from matplotlib import pyplot as plt

from snake import SnakeGameRL, Direction, Point
from snake import BLOCK_SIZE, INNER_BLOCK_SIZE, SPEED

from model import Linear_QNet, QTrainer

from library import plot, check_paths, get_record

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TRAIN_LENGTH = 100

reset_model = False
if reset_model:
    eps0 = 80
else:
    eps0 = 0


check_paths()       # create Saved Videos and Saved Models folders if they do not exist yet
model_path = os.path.join('Saved Models', 'model.pth')


class Agent:

    def __init__(self):
        self.n_games = 0        # store total number of games played
        self.epsilon = 0        # to control randomness
        self.gamma = 0.9        # discount rate -> how much we want to care about future steps
        self.memory = deque(maxlen=MAX_MEMORY)        # popleft() if memory is full -> will start removing elements from the left
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)         # neural network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)         # optimizer

        if not reset_model:
            # Load the state_dict of our saved model
            self.model.load_state_dict(torch.load(f=model_path))


    def get_state(self, game):
        head = game.snake[0]

        pt_l = Point(head.x - BLOCK_SIZE, head.y)
        pt_r = Point(head.x + BLOCK_SIZE, head.y)
        pt_u = Point(head.x, head.y - BLOCK_SIZE)       # remember that y+ points downwards
        pt_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(pt_r)) or
            (dir_l and game.is_collision(pt_l)) or
            (dir_u and game.is_collision(pt_u)) or
            (dir_d and game.is_collision(pt_d)),

            # Danger Right
            (dir_u and game.is_collision(pt_r)) or
            (dir_r and game.is_collision(pt_d)) or
            (dir_d and game.is_collision(pt_l)) or
            (dir_l and game.is_collision(pt_u)),

            # Danger Left
            (dir_u and game.is_collision(pt_l)) or
            (dir_l and game.is_collision(pt_d)) or
            (dir_d and game.is_collision(pt_r)) or
            (dir_r and game.is_collision(pt_u)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,      # food is on the left
            game.food.x > game.head.x,      # food is on the right
            game.food.y < game.head.y,      # food is above
            game.food.y > game.head.y       # food is below
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))       # append to the right -> will start removing elements from the left


    def train_long_memory(self):
        # this training is on a batch of data from our memory with batch size as hyperparameter
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)        # sample a batch of data from memory
        else:
            mini_sample = self.memory       # if memory would not fill batch size yet, we just use the whole memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)     # zip(*...) is the opposite of zip(...)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # # Alternatively, one could also use a for loop to iterate over the memory and train on each sample individually
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # We want random moves at the beginning, then more and more often the best move -> Exploration vs. Exploitation
        self.epsilon = eps0 - self.n_games        # epsilon decays over time -> hence it is called decay rate
        
        current_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:    # the move is randomly randomized
            move_idx = random.randint(0, 2)     # 0: straight, 1: right, 2: left
            current_move[move_idx] = 1
        else:
            state_T = torch.tensor(state, dtype=torch.float)    # convert state to tensor
            prediction = self.model(state_T)            # predict the best move using our model
            move = torch.argmax(prediction).item()      # get the index of the best move
            current_move[move] = 1

        return current_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = get_record()
    agent = Agent()
    game = SnakeGameRL()

    while True:
        # Get current state
        state_current = agent.get_state(game)

        # Get move
        current_move = agent.get_action(state_current)

        # Perform move and get new state
        reward, done, score = game.play_step(current_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_current, current_move, reward, state_new, done)

        # Remember by Store in memory
        agent.remember(state_current, current_move, reward, state_new, done)

        if done:
            
            print(f'Game: {agent.n_games}\t|\tScore: {score}\t|\tRecord: {record}')

            if score > record:
                record = score
                print('Saving Video...')
                game.save_video(record)

                # Save the plots when a new record is reached
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title('Training...')
                ax.set_xlabel('Number of Games')
                ax.set_ylabel('Score')
                ax.plot(plot_scores, label='Score per Game', color='blue')
                ax.plot(plot_mean_scores, label='Mean Scores', color='red')
                ax.legend()
                ax.text(len(plot_scores)-1, plot_scores[-1], str(plot_scores[-1]))
                ax.text(len(plot_mean_scores)-1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
                plot_save_path = os.path.join('Saved Plots', f'scores_record_{record}.png')
                fig.savefig(plot_save_path)
                plt.close(fig)

            if score > record or len(plot_scores) % TRAIN_LENGTH == 0:
                agent.model.save('model.pth')
            
            # Train long memory (-> experience replay) and Plot Results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()