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
        self.node = rclpy.create_node('gym_novamob_env')

        # * Define action and observation space
        # the action space is a dictionary with two keys: linear_x and angular_z velocities
        self.action_space = spaces.Dict({'linear_x': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                                         'angular_z': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)})
        
        # the observation space is a continuous space with 1080 values from the lidar sensor
        # ? still need to consider the state of the robot (position and orientation)
        # ? do i need to consider the odom and the map data?
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32)

        # Publishers 
        self.cmd_vel_publisher = self.node.create_publisher(Twist, 
                                                            '/cmd_vel',
                                                            10)
        self.publisher = self.node.create_publisher(String, 
                                                    '/robot/command',
                                                    10)

        # Subscribers
        self.subscription = self.node.create_subscription(String,
                                                          '/robot/state',
                                                          self.listener_callback,
                                                          10)
        self.lidar_sub = self.node.create_subscription(LaserScan,
                                                       '/scan',
                                                       self.lidar_callback,
                                                       10)
        
        # Place holder and initialization for data
        self.lidar_data = np.zeros(1080)
        self.current_state = np.zeros(2)

    def listener_callback(self, msg):
        self.current_state = np.array([float(val) for val in msg.data.split()])
        print(f"listenining to: {self.current_state}")

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)

    def step(self, action):
        # Send action to robot
        twist = Twist()
        twist.linear.x = action['linear_x']  # Linear velocity
        twist.angular.z = action['angular_z']  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Wait for the next state
        rclpy.spin_once(self.node)

        # Example reward calculation
        reward = -np.sum(np.square(self.current_state))

        done = self.is_done(self.current_state)

        return self.lidar_data, reward, done, {}

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