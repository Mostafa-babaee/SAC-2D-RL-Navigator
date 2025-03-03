import pybullet as p
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F

# ========================================================
# Custom 2D Environment
# ========================================================
class Custom2DEnvironment:
    def __init__(self, space_size=10, max_steps=200, visualize=True):
        self.space_size = space_size
        self.max_steps = max_steps
        self.current_step = 0
        self.trajectory_lines = []  # Store trajectory line visual IDs
        if visualize:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        if visualize:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=space_size, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0]
            )
        # Create a ground plane
        ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[space_size/2, space_size/2, 0.1])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[space_size/2, space_size/2, 0.1], rgbaColor=[1, 1, 1, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_collision, baseVisualShapeIndex=ground_visual, basePosition=[0, 0, -0.1])
        
        self.robot = None
        self.target = None
        self.robot_position = None
        self.target_position = None
        self.reset()

    def random_position(self):
        return np.random.uniform(-self.space_size/2, self.space_size/2, size=2)

    def create_point(self, position, color, size=0.2):
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=[position[0], position[1], 0.1])

    def reset(self):
        self.current_step = 0
        if self.robot:
            p.removeBody(self.robot)
        if self.target:
            p.removeBody(self.target)
        for line_id in self.trajectory_lines:
            p.removeUserDebugItem(line_id)
        self.trajectory_lines = []
        self.robot_position = self.random_position()
        self.target_position = self.random_position()
        self.robot = self.create_point(self.robot_position, color=[1, 0, 0, 1])  # Red robot
        self.target = self.create_point(self.target_position, color=[0, 0, 1, 1])  # Blue target
        return np.concatenate([self.robot_position, self.target_position])

    def step(self, action):
        # Save previous position to compute progress
        prev_position = self.robot_position.copy()
        # Update robot position with clipped action
        self.robot_position += np.clip(action, -0.2, 0.2)
        self.robot_position = np.clip(self.robot_position, -self.space_size/2, self.space_size/2)
        p.resetBasePositionAndOrientation(self.robot, [self.robot_position[0], self.robot_position[1], 0.1], [0, 0, 0, 1])
        
        # Draw trajectory line for visualization
        line_id = p.addUserDebugLine(
            lineFromXYZ=[prev_position[0], prev_position[1], 0.1],
            lineToXYZ=[self.robot_position[0], self.robot_position[1], 0.1],
            lineColorRGB=[1, 0.5, 0],
            lineWidth=2
        )
        self.trajectory_lines.append(line_id)
        
        # Calculate progress reward
        prev_distance = np.linalg.norm(prev_position - self.target_position)
        current_distance = np.linalg.norm(self.robot_position - self.target_position)
        progress_reward = (prev_distance - current_distance) * 10  # scaled progress reward
        
        # Small penalty for each step to encourage faster reaching
        step_penalty = -0.1
        reward = progress_reward + step_penalty
        
        done = False
        # Bonus for reaching the target
        if current_distance < 0.1:
            reward += 100
            done = True
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            reward -= 10
            done = True
        
        next_state = np.concatenate([self.robot_position, self.target_position])
        return next_state, reward, done

    def render(self):
        p.stepSimulation()
        time.sleep(0.01)

    def close(self):
        p.disconnect()

# ========================================================
# Replay Buffer
# ========================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards).unsqueeze(1),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1))
    def __len__(self):
        return len(self.buffer)

# ========================================================
# Actor Network
# ========================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

# ========================================================
# Critic Network
# ========================================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# ========================================================
# SAC Agent with Target Critics
# ========================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        # Target networks for critics
        self.critic_target_1 = Critic(state_dim, action_dim)
        self.critic_target_2 = Critic(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=1e-4)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp()

        self.gamma = 0.95
        self.tau = 0.005 
        self.action_bound = action_bound

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        if evaluate:
            action = torch.tanh(mu)
        else:
            action_sample = normal.rsample()
            action = torch.tanh(action_sample)
        return action.detach().squeeze(0).numpy() * self.action_bound

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # ----------------------------
        # Update Critic Networks
        # ----------------------------
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            normal = torch.distributions.Normal(next_mu, next_std)
            next_action_sample = normal.rsample()
            next_action = torch.tanh(next_action_sample) * self.action_bound
            log_prob = normal.log_prob(next_action_sample).sum(dim=1, keepdim=True)
            log_prob -= torch.sum(torch.log(1 - torch.tanh(next_action_sample)**2 + 1e-6), dim=1, keepdim=True)
            next_q1 = self.critic_target_1(next_states, next_action)
            next_q2 = self.critic_target_2(next_states, next_action)
            min_next_q = torch.min(next_q1, next_q2) - self.alpha * log_prob
            target_q = rewards + (1 - dones) * self.gamma * min_next_q
        
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Soft-update target networks
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # ----------------------------
        # Update Actor Network
        # ----------------------------
        mu, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action_sample = normal.rsample()
        log_prob = normal.log_prob(action_sample).sum(dim=1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - torch.tanh(action_sample)**2 + 1e-6), dim=1, keepdim=True)
        actor_loss = (self.alpha * log_prob - self.critic_1(states, torch.tanh(action_sample) * self.action_bound)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------------
        # Update Alpha (Entropy Temperature)
        # ----------------------------
        alpha_loss = -(self.log_alpha * (log_prob + 0.2).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

# ========================================================
# Training Loop
# ========================================================
if __name__ == "__main__":
    env = Custom2DEnvironment(visualize=True)
    replay_buffer = ReplayBuffer(capacity=100000)
    agent = SACAgent(state_dim=4, action_dim=2, action_bound=1.0)
    
    episodes = 5000
    batch_size = 64  # Adjusted batch size for stability
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            env.render()
            
            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
        
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
    
    env.close()
    
    # ========================================================
    # Save the Trained Model
    # ========================================================
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic_1': agent.critic_1.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'log_alpha': agent.log_alpha
    }, "sac_trained_model.pth")
    
    print("Training Complete!")
