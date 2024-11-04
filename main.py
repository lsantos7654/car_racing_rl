import gymnasium as gym
import pygame
import numpy as np
from pygame.locals import *

# Initialize Pygame for keyboard capture
pygame.init()
screen = pygame.display.set_mode((600, 400))

# Initialize the environment
env = gym.make("CarRacing-v3", render_mode="human")
observation, info = env.reset()

# Control variables
steering = 0.0
gas = 0.0
brake = 0.0

running = True
while running:
    # Process Pygame events for keyboard input
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Get the current keyboard state
    keys = pygame.key.get_pressed()

    # Update controls based on keyboard input
    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[K_LEFT]:
        steering = -1.0
    if keys[K_RIGHT]:
        steering = 1.0
    if keys[K_UP]:
        gas = 1.0
    if keys[K_DOWN]:
        brake = 1.0

    # Create action array and step the environment
    action = np.array([steering, gas, brake])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
pygame.quit()
