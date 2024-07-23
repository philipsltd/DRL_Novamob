import sys
import os
import gymnasium as gym
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import novamob_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/novamob_gym/novamob_gym'))

def train_environment():
    # Initialize the environment using gym.make
    env = DummyVecEnv([lambda: gym.make('NovamobGym-v0')])

    model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log="./sac_novamob_tensorboard/", device='cpu')
    
    # Train the agent
    model.learn(total_timesteps=1000000)  # Adjust the number of timesteps as needed
    
    # Save the trained model
    model.save("sac_novamob_model")

    # Load the trained model
    model = SAC.load("sac_novamob_model")

    # Evaluate the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

if __name__ == '__main__':
    rclpy.init(args=None)
    try:
        train_environment()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
