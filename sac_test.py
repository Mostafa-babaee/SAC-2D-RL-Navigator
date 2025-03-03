import torch
import numpy as np
import time
import pybullet as p
from sac_navigator import Custom2DEnvironment, SACAgent  

env = Custom2DEnvironment(space_size=10, max_steps=200, visualize=True)

# Instantiate the agent
agent = SACAgent(state_dim=4, action_dim=2, action_bound=1.0)

# Load the saved model weights
checkpoint = torch.load("sac_trained_model.pth")
agent.actor.load_state_dict(checkpoint['actor'])
agent.critic_1.load_state_dict(checkpoint['critic_1'])
agent.critic_2.load_state_dict(checkpoint['critic_2'])
agent.log_alpha = checkpoint['log_alpha']
agent.alpha = agent.log_alpha.exp()

# Test loop: 
num_test_episodes = 10
for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
    print(f"Test Episode {episode+1}: Total Reward = {total_reward:.2f}")

env.close()