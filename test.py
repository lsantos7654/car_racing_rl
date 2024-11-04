import gymnasium as gym
import torch
import numpy as np
import cv2
import argparse
from rl_implementation import (
    DQN,
    DiscreteActions,
    preprocess_frame,
)  # Import your model classes


def evaluate_agent(args):
    # Initialize environment
    env = gym.make("CarRacing-v3", render_mode="human" if not args.no_render else None)

    # Initialize device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize model and actions
    discrete_actions = DiscreteActions()
    state_dim = (1, 96, 96)
    action_dim = discrete_actions.n_actions

    # Create model and load checkpoint
    model = DQN(action_dim).to(device)
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded checkpoint from {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()  # Set to evaluation mode

    total_rewards = []
    for episode in range(args.num_episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < args.max_steps:
            steps += 1

            # Get action from model
            with torch.no_grad():
                state_tensor = state.unsqueeze(0).to(device)
                q_values = model(state_tensor)
                if args.deterministic:
                    action_idx = q_values.max(1)[1].item()
                else:
                    # Add some exploration during evaluation
                    if np.random.random() < args.eval_epsilon:
                        action_idx = np.random.randint(action_dim)
                    else:
                        action_idx = q_values.max(1)[1].item()

            # Convert to continuous action and step
            action = discrete_actions.get_action(action_idx)
            next_state, reward, done, truncated, _ = env.step(action)

            episode_reward += reward
            state = preprocess_frame(next_state)

            if args.verbose and steps % 100 == 0:
                print(
                    f"Episode {episode + 1}, Step {steps}, Current Reward: {episode_reward:.2f}"
                )

            if done or truncated:
                break

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1} finished with reward: {episode_reward:.2f} in {steps} steps"
        )

        if args.save_rewards:
            np.save(
                f"evaluation_rewards_{args.checkpoint_path.split('/')[-1]}.npy",
                np.array(total_rewards),
            )

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on CarRacing-v3"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate (default: 5)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )

    parser.add_argument(
        "--no_render", action="store_true", help="Disable rendering of the environment"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration)",
    )

    parser.add_argument(
        "--eval_epsilon",
        type=float,
        default=0.05,
        help="Epsilon value for evaluation exploration (default: 0.05)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    parser.add_argument(
        "--save_rewards", action="store_true", help="Save rewards to a numpy file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(args)
