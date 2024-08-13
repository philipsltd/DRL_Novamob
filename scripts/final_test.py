import os
import sys
import gymnasium as gym
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
torch.set_default_tensor_type('torch.FloatTensor')  # Force all tensors to be on CPU


# Assuming the NovamobGym environment is already defined in your module or you import it from where it's defined
import novamob_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/novamob_gym/novamob_gym'))

def evaluate_model(env, model, num_episodes=10):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards
            print(f"Step: {obs}, Action: {action}, Reward: {rewards}")
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == '__main__':
    rclpy.init(args=None)  # Initialize ROS 2 if needed
    # Load the trained model
    model = PPO.load("/home/filipe/thesis/drl_novamob_alt/src/novamob_gym/novamob_gym/ppo_novamob_model.zip")

    # Create the environment
    env = gym.make('NovamobGym-v0')
    env = DummyVecEnv([lambda: env])  # Wrap in a dummy VecEnv if necessary

    # Evaluate the model
    evaluate_model(env, model, num_episodes=10)

    # Shutdown ROS 2 if necessary
    if rclpy.ok():
        rclpy.shutdown()
