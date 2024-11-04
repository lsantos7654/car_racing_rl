# Import required libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import wandb
import time
from datetime import datetime
from collections import namedtuple, deque

# Device configuration
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.running_mean = 0
        self.running_std = 1
        self.epsilon = epsilon

    def normalize(self, reward):
        self.running_mean = 0.99 * self.running_mean + 0.01 * reward
        self.running_std = 0.99 * self.running_std + 0.01 * abs(
            reward - self.running_mean
        )
        return (reward - self.running_mean) / (self.running_std + self.epsilon)


class CurriculumLearning:
    def __init__(
        self, initial_max_steps=200, final_max_steps=1000, episodes_to_max=100
    ):
        self.initial_max_steps = initial_max_steps
        self.final_max_steps = final_max_steps
        self.episodes_to_max = episodes_to_max

    def get_max_steps(self, episode):
        return int(
            min(
                self.initial_max_steps
                + (self.final_max_steps - self.initial_max_steps)
                * episode
                / self.episodes_to_max,
                self.final_max_steps,
            )
        )


# Define discrete actions for the car
class DiscreteActions:
    def __init__(self):
        self.actions = [
            np.array([-1.0, 0.0, 0.0]),  # Full left
            np.array([-0.5, 0.0, 0.0]),  # Half left
            np.array([0.0, 0.0, 0.0]),  # Straight
            np.array([0.5, 0.0, 0.0]),  # Half right
            np.array([1.0, 0.0, 0.0]),  # Full right
            np.array([0.0, 1.0, 0.0]),  # Full gas
            np.array([0.0, 0.5, 0.0]),  # Half gas
            np.array([0.0, 0.0, 0.5]),  # Light brake
            np.array([0.0, 0.0, 1.0]),  # Full brake
            np.array([-0.5, 0.5, 0.0]),  # Left + gas
            np.array([0.5, 0.5, 0.0]),  # Right + gas
        ]
        self.n_actions = len(self.actions)

    def get_action(self, index):
        return self.actions[index]


# Neural Network Model
class DQN(nn.Module):
    def __init__(self, n_actions, input_channels=1):
        super(DQN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Calculate fc input dimension
        dummy_input = torch.zeros(1, input_channels, 96, 96)
        conv_out = self.conv_layers(dummy_input)
        self.fc_input_dim = conv_out.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        conv_out = self.conv_layers(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        return self.fc_layers(flattened)


# Experience Replay Buffer
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# Modify the ReplayBuffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.recent_rewards = deque(maxlen=1000)  # Keep track of recent rewards

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)
        self.recent_rewards.append(transition.reward)  # Store reward

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_average_reward(self):
        if len(self.recent_rewards) == 0:
            return 0.0
        return np.mean(self.recent_rewards)

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)

# # Modify the DQNAgent class update_epsilon method
# def update_epsilon(self):
#     self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# Alternative version of update_epsilon that considers performance
def update_epsilon(self):
    avg_reward = self.memory.get_average_reward()
    # Decay epsilon faster if performing well
    decay = self.epsilon_decay * (1.0 + max(0, avg_reward) * 0.01)
    self.epsilon = max(self.epsilon_end, self.epsilon * decay)


# Preprocessing function
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (96, 96))
    frame = frame / 255.0
    return torch.FloatTensor(frame).unsqueeze(0)  # Shape: [1, 96, 96]


# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim  # Add this line to store action_dim

        self.model = DQN(action_dim).to(device)
        self.target_model = DQN(action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(300000)
        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.997
        self.gamma = 0.99
        self.reward_normalizer = RewardNormalizer()
        self.train_steps = 0

    def warmup(self, env, warmup_steps=10000):
        print("Starting warmup phase...")
        state, _ = env.reset()
        state = preprocess_frame(state)

        for step in range(warmup_steps):
            action_idx = random.randrange(self.action_dim)
            action = discrete_actions.get_action(action_idx)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_frame(next_state)

            # Normalize reward
            normalized_reward = self.reward_normalizer.normalize(reward)

            self.memory.push(state, action_idx, normalized_reward, next_state, done)

            if done or truncated:
                state, _ = env.reset()
                state = preprocess_frame(state)
            else:
                state = next_state

            if step % 1000 == 0:
                print(f"Warmup step {step}/{warmup_steps}")

        print("Warmup complete!")

    def get_target_q_values(self, next_states, rewards, dones):
        with torch.no_grad():
            # Use online network to select actions (Double DQN)
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            # Use target network to evaluate actions
            next_q_values = self.target_model(next_states).gather(1, next_actions)
        return rewards + (1 - dones) * self.gamma * next_q_values

    def update_epsilon(self):
        if len(self.memory) > self.batch_size:
            recent_rewards = [t.reward for t in list(self.memory)[-1000:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            # Slow down decay if performance is poor
            decay_rate = (
                self.epsilon_decay if avg_reward > -30 else self.epsilon_decay * 0.5
            )
            self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = state.unsqueeze(0).to(device)
                q_values = self.model(state)
                return q_values.max(1)[1].item()
        return random.randrange(self.action_dim)

    def save_checkpoint(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.tensor(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.stack(batch.next_state).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

        # Use Double DQN for target computation
        target_q = self.get_target_q_values(
            next_state_batch, reward_batch.unsqueeze(1), done_batch.unsqueeze(1)
        )

        current_q = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        self.train_steps += 1
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Training function
def train_agent(env, agent, num_episodes, max_steps):
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="car-racing-dqn",
        config={
            "architecture": "DQN",
            "learning_rate": 3e-4,
            "episodes": num_episodes,
            "max_steps": max_steps,
            "epsilon_decay": agent.epsilon_decay,
            "gamma": agent.gamma,
        },
    )

    best_reward = float("-inf")

    # Add curriculum learning
    curriculum = CurriculumLearning()

    # Add warmup phase
    agent.warmup(env, warmup_steps=10000)

    # Statistics tracking
    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        episode_reward = 0
        steps_taken = 0
        losses = []
        start_time = time.time()

        # Get max steps from curriculum
        current_max_steps = curriculum.get_max_steps(episode)

        for step in range(current_max_steps):
            steps_taken += 1
            action_idx = agent.select_action(state)
            action = discrete_actions.get_action(action_idx)
            next_state, reward, done, truncated, _ = env.step(action)

            next_state = preprocess_frame(next_state)
            episode_reward += reward
            normalized_reward = agent.reward_normalizer.normalize(reward)

            # Store transition and train
            agent.memory.push(
                state.clone(),
                action_idx,
                normalized_reward,  # Store normalized reward
                next_state.clone(),
                done,
            )
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state = next_state

            # Print real-time statistics
            if step % 10 == 0:  # Update every 10 steps
                print(
                    f"\rEpisode {episode}/{num_episodes} | "
                    f"Step {step}/{max_steps} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}",
                    end="",
                )

            if done or truncated:
                break

        # Episode complete - update statistics
        episode_time = time.time() - start_time
        all_rewards.append(episode_reward)
        all_lengths.append(steps_taken)
        avg_loss = np.mean(losses) if losses else 0
        avg_reward = np.mean(all_rewards[-100:])  # Moving average of last 100 episodes

        # Update agent
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()

        # Log to wandb
        wandb.log(
            {
                "episode": episode,
                "reward": episode_reward,
                "average_reward": avg_reward,
                "steps": steps_taken,
                "epsilon": agent.epsilon,
                "loss": avg_loss,
                "episode_time": episode_time,
            }
        )

        # Print episode summary
        print(f"\nEpisode {episode} Complete:")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Average Reward (last 100): {avg_reward:.2f}")
        print(f"Steps Taken: {steps_taken}")
        print(f"Episode Time: {episode_time:.2f}s")
        print(f"Average Loss: {avg_loss:.5f}")
        print("-" * 50)

        # After each episode, check if we should save a checkpoint
        if episode % 50 == 0:  # Save every 50 episodes
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_episode_{episode}.pth"
            )
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model based on performance
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            agent.save_checkpoint(best_model_path)
            print(f"New best model saved with average reward: {best_reward:.2f}")

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save_checkpoint(final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")

    wandb.finish()
    return checkpoint_dir


# Main execution
if __name__ == "__main__":
    # Initialize wandb
    wandb.login()

    # Create environment with human render mode
    env = gym.make("CarRacing-v3", render_mode="human")
    discrete_actions = DiscreteActions()

    state_dim = (1, 96, 96)
    action_dim = discrete_actions.n_actions

    # Initialize agent
    agent = DQNAgent(state_dim, action_dim)

    # If you want to load from a previous checkpoint:
    # checkpoint_path = "path_to_your_checkpoint.pth"
    # agent.load_checkpoint(checkpoint_path)

    try:
        checkpoint_dir = train_agent(env, agent, num_episodes=500, max_steps=1000)
        print(f"Training completed. All checkpoints saved in: {checkpoint_dir}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint on interrupt
        interrupt_checkpoint_path = os.path.join(
            "checkpoints",
            f"interrupt_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        )
        agent.save_checkpoint(interrupt_checkpoint_path)
        print(f"Interrupt checkpoint saved: {interrupt_checkpoint_path}")
    finally:
        env.close()
