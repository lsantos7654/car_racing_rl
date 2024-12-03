import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler  # Updated import
import cv2

# Device and CUDA optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
print(f"Using device: {device}")


class ParallelEnvs:
    def __init__(self, num_envs=8):
        self.envs = [gym.make("CarRacing-v3") for _ in range(num_envs)]

    def reset(self):
        states = [env.reset()[0] for env in self.envs]
        return torch.stack([preprocess_frame(state) for state in states])

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_states = torch.stack([preprocess_frame(s) for s, _, _, _, _ in results])
        rewards = torch.tensor([r for _, r, _, _, _ in results], device=device)
        dones = torch.tensor(
            [float(t or tr) for _, _, t, tr, _ in results], device=device
        )
        return next_states, rewards, dones

    def close(self):
        for env in self.envs:
            env.close()


class DiscreteActions:
    def __init__(self):
        self.actions = [
            np.array([-1.0, 0.0, 0.0]),  # LEFT
            np.array([1.0, 0.0, 0.0]),  # RIGHT
            np.array([0.0, 1.0, 0.0]),  # GAS
            np.array([0.0, 0.0, 1.0]),  # BRAKE
            np.array([-1.0, 1.0, 0.0]),  # LEFT+GAS
            np.array([1.0, 1.0, 0.0]),  # RIGHT+GAS
            np.array([0.0, 0.0, 0.0]),  # NEUTRAL
            np.array([0.0, 0.0, 1.0]),  # BRAKE (same as 3)
        ]
        self.n_actions = len(self.actions)

    def get_action(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return np.array([self.actions[i] for i in index])


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate feature shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            feature_size = self.features(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state, deterministic=False):
        action_probs, value = self(state)

        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action_idx = dist.sample()

        return action_idx, dist.log_prob(action_idx), value


class PPOMemory:
    def __init__(self, num_envs, capacity=2048):
        self.states = []
        self.action_idxs = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.num_envs = num_envs
        self.capacity = capacity

    def push(self, state, action_idx, reward, value, log_prob, done):
        self.states.append(state)
        self.action_idxs.append(action_idx)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done.float())  # Convert to float

    def get(self):
        states = torch.cat(self.states)
        action_idxs = torch.cat(self.action_idxs)
        rewards = torch.cat(self.rewards)
        values = torch.cat(self.values)
        log_probs = torch.cat(self.log_probs)
        dones = torch.cat(self.dones)

        self.clear()
        return states, action_idxs, rewards, values, log_probs, dones

    def clear(self):
        self.states = []
        self.action_idxs = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def is_full(self):
        return len(self.states) * self.num_envs >= self.capacity


def preprocess_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (96, 96))
        frame = torch.from_numpy(frame).to(device)
    frame = frame.float() / 255.0
    return frame.unsqueeze(0)


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete_actions,
        num_envs,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.actor_critic = torch.compile(ActorCritic(state_dim, action_dim)).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.scaler = GradScaler(device='cuda')  # Corrected initialization

        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.memory = PPOMemory(num_envs)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            action_idx, log_prob, value = self.actor_critic.get_action(
                state, deterministic
            )
            action = self.discrete_actions.get_action(action_idx)
            return action, action_idx, log_prob, value

    def train(self, batch_size=256, epochs=10):
        states, action_idxs, rewards, values, old_log_probs, dones = self.memory.get()

        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            running_advantage = running_return - values[t].item()
            returns[t] = running_return
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                if end > len(states):
                    end = len(states)
                batch_indices = slice(start, end)

                with autocast(device_type="cuda"):
                    action_probs, current_values = self.actor_critic(
                        states[batch_indices]
                    )
                    dist = Categorical(action_probs)

                    current_log_probs = dist.log_prob(action_idxs[batch_indices])
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(current_log_probs - old_log_probs[batch_indices])
                    surr1 = ratios * advantages[batch_indices]
                    surr2 = (
                        torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                        * advantages[batch_indices]
                    )

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(
                        current_values.squeeze(), returns[batch_indices]
                    )
                    loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()


def train_agent():
    wandb.init(
        project="car-racing-ppo-8actions",
        config={
            "num_envs": 8,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_episodes": 1000,
            "steps_per_update": 2048,
            "batch_size": 64,
            "epochs": 10,
        },
    )

    num_envs = wandb.config.num_envs
    parallel_envs = ParallelEnvs(num_envs)
    state_dim = (1, 96, 96)
    discrete_actions = DiscreteActions()

    agent = PPO(
        state_dim=state_dim,
        action_dim=discrete_actions.n_actions,
        discrete_actions=discrete_actions,
        num_envs=num_envs,
        lr=wandb.config.learning_rate,
        gamma=wandb.config.gamma,
        epsilon=wandb.config.epsilon,
        value_coef=wandb.config.value_coef,
        entropy_coef=wandb.config.entropy_coef,
    )

    best_reward = float("-inf")

    for episode in range(wandb.config.max_episodes):
        states = parallel_envs.reset()
        episode_rewards = torch.zeros(num_envs, device=device)
        steps = 0

        while True:
            actions, action_idxs, log_probs, values = agent.select_action(states)
            next_states, rewards, dones = parallel_envs.step(actions)

            agent.memory.push(states, action_idxs, rewards, values, log_probs, dones)
            states = next_states
            episode_rewards += rewards
            steps += 1

            if agent.memory.is_full():
                agent.train(
                    batch_size=wandb.config.batch_size, epochs=wandb.config.epochs
                )

            if dones.all() or steps >= wandb.config.steps_per_update:
                break

        mean_reward = episode_rewards.mean().item()
        wandb.log({"episode": episode, "reward": mean_reward, "steps": steps})

        print(f"Episode {episode}, Mean Reward: {mean_reward:.2f}, Steps: {steps}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(agent.actor_critic.state_dict(), f"best_model_{episode}.pth")

    parallel_envs.close()
    wandb.finish()


if __name__ == "__main__":
    wandb.login()
    train_agent()
