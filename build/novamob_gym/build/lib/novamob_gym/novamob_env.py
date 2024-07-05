import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node

# ROS 2 message imports
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class NovamobGym(gym.Env):
    def __init__(self):
        super(NovamobGym, self).__init__()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('gym_robot_env')

        # Define action and observation space
        # Example for a robot with two continuous actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Publishers and subscribers
        self.cmd_vel_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher = self.node.create_publisher(String, '/robot/command', 10)
        self.subscription = self.node.create_subscription(
            String,
            '/robot/state',
            self.listener_callback,
            10
        )
        self.current_state = np.zeros(2)

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def listener_callback(self, msg):
        self.current_state = np.array([float(val) for val in msg.data.split()])
        print(f"listenining to: {self.current_state}")

    def lidar_callback(self, msg):
        print(f"lidar data: {msg.ranges}")

    def step(self, action):
        # Send action to robot
        twist = Twist()
        twist.linear.x = 0.5 # action[0]  # Linear velocity
        twist.angular.z = 0.0 # action[1]  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Wait for the next state
        rclpy.spin_once(self.node)

        # Example reward calculation
        reward = -np.sum(np.square(self.current_state))

        done = self.is_done(self.current_state)

        return self.current_state, reward, done, {}

    def reset(self, seed=None, options=None):
        # Handle the seed for random number generation
        super().reset(seed=seed)

        # Reset the robot and simulation
        self.publisher.publish(String(data='reset'))
        rclpy.spin_once(self.node)

        return self.current_state, {}

    def is_done(self, state):
        # Define a condition to end the episode
        # return np.linalg.norm(state) < 0.1
        return False

    def close(self):
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def render(self, mode='human'):
        # Render the environment for visualization
        pass

    def seed(self, seed=None):
        # Handle the seed for random number generation
        super().seed(seed=seed)

    def __del__(self):
        self.close()

def main(args=None):
    env = NovamobGym()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your action selection logic
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

    env.close()

if __name__ == '__main__':
    main()