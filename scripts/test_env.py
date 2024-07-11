import sys
import os
import gymnasium as gym
import rclpy
import novamob_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/novamob_gym/novamob_gym'))

def test_environment():
    # Initialize the environment using gym.make
    env = gym.make('NovamobGym-v0')
    
    # Reset the environment and get the initial observation
    obs = env.reset()
    print(f"Initial Observation: {obs}")
    
    done = False
    step_count = 0
    
    # Test loop - you can define the number of steps you want to test
    while not done and step_count < 50:  # Test for 5 steps
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print the results of the step
        print(f"Step {step_count}: Observation: {obs}, Reward: {reward}, Done: {terminated}")
        # print(f"Step: {step_count}, Reward: {reward}, Done: {terminated}")
        
        step_count += 1
    
    # Close the environment
    env.close()

if __name__ == '__main__':
    rclpy.init(args=None)
    try:
        test_environment()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
