import sys
import os
import gymnasium as gym
import rclpy
import novamob_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/novamob_gym/novamob_gym'))

def test_environment():
    # Initialize the environment using gym.make
    env = gym.make('NovamobGym-v0')
    
    num_episodes = 10
    max_steps_per_episode = 50

    for episode in range(num_episodes):
        # Reset the environment and get the initial observation
        obs = env.reset()
        print(f"Episode {episode + 1}: Initial Observation: {obs}")

        done = False
        step_count = 0
        cumulative_reward = 0
        
        # Test loop - you can define the number of steps you want to test
        while not done and step_count < max_steps_per_episode:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward
            done = terminated

            step_count += 1

        print(f"Episode {episode + 1} finished. Total steps: {step_count}, Cumulative reward: {cumulative_reward}")

    # Close the environment
    env.close()

if __name__ == '__main__':
    rclpy.init(args=None)
    try:
        test_environment()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
