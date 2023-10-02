import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import imageio
import cv2
from library import get_record
import os

pygame.init()
font = pygame.font.Font(os.path.join('Archive', 'arial.ttf'), 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (23, 232, 86)
GREEN2 = (146, 232, 172)
BLACK = (0,0,0)

BLOCK_SIZE = 20
INNER_BLOCK_SIZE = 12
SPEED = 100

class SnakeGameRL:

    def __init__(self, w=640, h=480, record=get_record()):
        self.w = w
        self.h = h
        self.record = record
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):        # Initialize game state

        self.direction = Direction.RIGHT
        self.frame_iteration = 0
        self.head = Point(self.w/2, self.h/2)       # initialize snake head in the middle of the screen
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]       # initialize snake body as 3 blocks
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.frames = []                # to store captured frames


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():        # detects if user closes window, then quits
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        # Move Step
        self._move(action)  # update the snake head
        self.snake.insert(0, self.head)    # insert new head position into snake list
        
        # Check for Game Over
        reward = 0      # reward is 0 by default

        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10        # give negative reward if game is over
            return reward, game_over, self.score
            
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10         # give positive reward if food is eaten
            self._place_food()
        else:
            self.snake.pop()        # remove last block of snake if food is not eaten
        
        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return Game Over and Score
        return reward, game_over, self.score
    

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        

    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, INNER_BLOCK_SIZE, INNER_BLOCK_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])     # display score in top left corner
        pygame.display.flip()       # update the display -> makes changes visible

        # Capture the current frame
        frame_data = pygame.surfarray.array3d(pygame.display.get_surface()).transpose([1, 0, 2])
        self.frames.append(frame_data)




    def save_gif(self, record):
        filename="snake-" + str(record) + ".gif"
        filename = os.path.join('Saved Videos', filename)
        duration = 1000 // 30  # For 30 fps, each frame should last 1000/30 milliseconds
        if not self.frames:
            print("No frames captured!")
            return
        imageio.mimsave(filename, self.frames, duration=duration)


    def save_video(self, record):
        filename = "snake-" + str(record) + ".mp4"
        filename = os.path.join('Saved Videos', filename)
        
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        fps = 30
        height, width, _ = self.frames[0].shape

        # Upscale resolution (optional)
        upscale_factor = 2  # Adjust this value as needed
        width *= upscale_factor
        height *= upscale_factor

        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in self.frames:
            # Convert from RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Upscale frame (optional)
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            
            out.write(frame_bgr)
        
        out.release()

        

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Update the Direction
        if np.array_equal(action, [1, 0, 0]):       # go straight
            new_dir = clock_wise[idx]
        if np.array_equal(action, [0, 1, 0]):      
            next_idx = (idx + 1) % 4         # we periodically move in a circle (clockwise)
            new_dir = clock_wise[next_idx]       # right turn r -> d -> l -> u
        if np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4         # we periodically move in a circle (clockwise)
            new_dir = clock_wise[next_idx]       # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:      # note: the y-axis is flipped and starts from 0 at the top
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

            
