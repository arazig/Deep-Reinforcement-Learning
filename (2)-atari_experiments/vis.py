import torch
import gymnasium as gym
from tianshou.data import Collector
from atari_network import Rainbow
from atari_wrapper import wrap_deepmind
from tianshou.policy import RainbowPolicy
from tianshou.env import DummyVectorEnv

# Define the task
task = "MsPacmanNoFrameskip-v4"

# Path to your saved policy
policy_path = "/home/abdelilah-younsi/Desktop/master X-2024-2025/RL/project2/log/MsPacmanNoFrameskip-v4/rainbow/0/250228-030104/policy.pth"

# Create the environment with render_mode explicitly set
raw_env = gym.make(task, render_mode="human")
env = wrap_deepmind(
    raw_env,
    episode_life=False,  # No episode life for evaluation
    clip_rewards=False,  # No reward clipping for evaluation
    frame_stack=4,
    scale=False,
)

# Get the state and action shapes from the environment before wrapping
state_shape = env.observation_space.shape or env.observation_space.n
action_space = env.action_space
action_shape = action_space.shape or action_space.n
print(f"Observation shape: {state_shape}")
print(f"Action shape: {action_shape}")

# Create the same network architecture as used during training
device = "cuda" if torch.cuda.is_available() else "cpu"
net = Rainbow(
    *state_shape,
    action_shape,
    num_atoms=51,
    noisy_std=0.1,
    device=device,
    is_dueling=True,
    is_noisy=True
)

# Create optimizer (won't be used for evaluation, but needed for policy initialization)
optim = torch.optim.Adam(net.parameters(), lr=0.0000625)

# Initialize the Rainbow policy with the same parameters as during training
policy = RainbowPolicy(
    model=net,
    optim=optim,
    discount_factor=0.99,
    action_space=action_space,  # Use the original action space
    num_atoms=51,
    v_min=-10.0,
    v_max=10.0,
    estimation_step=3,
    target_update_freq=500
).to(device)

# Load the saved policy
try:
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    print(f"Successfully loaded policy from {policy_path}")
except Exception as e:
    print(f"Error loading policy: {e}")
    exit(1)

# Set policy to evaluation mode
policy.eval()

# Set exploration epsilon for testing
eps_test = 0.005
policy.set_eps(eps_test)

# Wrap in DummyVectorEnv AFTER creating the policy
env = DummyVectorEnv([lambda: env])

# Create a collector and visualize the agent's behavior
collector = Collector(policy, env, exploration_noise=True)

# Reset the collector and collect episodes
print("Starting visualization...")
collector.reset()
try:
    result = collector.collect(n_episode=1, render=0.03)  # ~33 FPS
    print("Visualization completed successfully")
    
    # Print some statistics about the episode
    print("\nEpisode statistics:")
    print(f"Length: {result.lens[0]} steps")
    print(f"Reward: {result.rews[0]}")
except Exception as e:
    print(f"Error during visualization: {e}")
    # Try an alternative approach if rendering fails
    print("Trying an alternative approach...")
    # Create a new environment for manual stepping
    manual_env = gym.make(task, render_mode="human")
    manual_env = wrap_deepmind(
        manual_env,
        episode_life=False,
        clip_rewards=False,
        frame_stack=4,
        scale=False,
    )
    
    # Just step through the environment manually
    obs, _ = manual_env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 10000:  # Set a reasonable limit
        act = policy.forward(obs)[0]
        obs, rew, terminated, truncated, _ = manual_env.step(act)
        done = terminated or truncated
        total_reward += rew
        step_count += 1
        print(f"Step {step_count}, Reward: {rew}", end="\r")
    print(f"\nManual run completed. Steps: {step_count}, Total reward: {total_reward}")
