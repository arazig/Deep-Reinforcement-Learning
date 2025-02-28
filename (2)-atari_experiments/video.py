import torch
import gymnasium as gym
import numpy as np
import cv2
from tianshou.data import Batch
from atari_network import Rainbow
from atari_wrapper import wrap_deepmind
from tianshou.policy import RainbowPolicy
import os
from datetime import datetime

# Define the task
task = "MsPacmanNoFrameskip-v4"

# Path to your saved policy
policy_path = "/home/abdelilah-younsi/Desktop/master X-2024-2025/RL/project2/log/MsPacmanNoFrameskip-v4/rainbow/0/250228-030104/policy.pth"

# Video settings
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(video_dir, f"mspacman_agent_{timestamp}.mp4")
fps = 30  # Frames per second for the output video
record_frames = []

# Create the environment with render_mode set to rgb_array to capture frames
raw_env = gym.make(task, render_mode="rgb_array")
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

# Manual rendering and recording approach
def manual_run_with_recording():
    # Reset the environment
    print("Starting recording...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    # Get initial render
    frame = env.render()
    record_frames.append(frame)
    
    while not done and step_count < 10000:  # Set a reasonable limit
        # Create a proper Batch for the policy
        batch = Batch(obs=[obs], info={})
        
        # Get action from policy
        with torch.no_grad():
            result = policy(batch)
            act = result.act[0]
        
        # Step the environment
        obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        total_reward += rew
        step_count += 1
        
        # Render and record
        frame = env.render()
        record_frames.append(frame)
        
        # Display progress
        if step_count % 10 == 0:  # Update less frequently to improve performance
            print(f"Step {step_count}, Reward: {total_reward:.1f}", end="\r")
    
    print(f"\nRecording completed. Steps: {step_count}, Total reward: {total_reward:.1f}")
    return step_count, total_reward

# Run the agent and record frames
step_count, total_reward = manual_run_with_recording()

# Save recorded frames as video
if record_frames:
    print(f"Saving video to {video_path}...")
    
    # Get dimensions of frames
    height, width, layers = record_frames[0].shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' for .avi
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame in record_frames:
        # OpenCV uses BGR format, convert if necessary
        if frame.shape[2] == 3:  # RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    
    # Release video writer
    video.release()
    print(f"Video saved successfully! {len(record_frames)} frames, {len(record_frames)/fps:.1f} seconds")
    print(f"Video path: {os.path.abspath(video_path)}")
    print(f"Episode stats: {step_count} steps, reward: {total_reward:.1f}")
else:
    print("No frames were recorded!")
