import gymnasium as gym
import novamob_gym

env = gym.make('NovamobGym-v0')

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Replace with your action selection logic
    obs, reward, done, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

env.close()
