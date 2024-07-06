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
        self.observation_space = spaces.Dict({'lidar': spaces.Box(low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32),
                                              'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                                              'orientation': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                                              })

        # Publishers 
        self.cmd_vel_publisher = self.node.create_publisher(Twist, 
                                                            '/cmd_vel',
                                                            1)
        self.publisher = self.node.create_publisher(String, 
                                                    '/robot/command',
                                                    10)

        # Subscribers
        self.odom_subscription = self.node.create_subscription(String,
                                                          '/odom',
                                                          self.odom_callback,
                                                          10)
        self.lidar_subscription = self.node.create_subscription(LaserScan,
                                                       '/scan',
                                                       self.lidar_callback,
                                                       10)
        
        # Place holder and initialization for data
        self.lidar_data = np.zeros(1080)
        self.robot_state = np.zeros(2)


    def odom_callback(self, msg):
        # Store the robot state in a 6 element numpy array (first 3 elements are position, last 4 are orientation)
        self.robot_state = np.array([float(val) for val in msg.data.split()])
        print(f"listenining to: {self.robot_state}")


    def lidar_callback(self, msg):
        # Process LiDAR data (extract X, Y, Z and potentially preprocess)
        lidar_readings = list(msg.ranges)
        # Handle infinite values returned by the LiDAR sensor
        lidar_readings = [1e6 if x == float('inf') else x for x in lidar_readings]
        # Convert the readings to a string and remove brackets
        lidar_readings_str = str(lidar_readings).replace('[', '').replace(']', '')
        # Split the string into separate elements and convert them to floats
        lidar_readings_columns = [float(x) for x in lidar_readings_str.split(',')]
        # Store the LiDAR data in a numpy array
        self.lidar_data = np.array(lidar_readings_columns)



    def step(self, action):
        # Send action to robot
        twist = Twist()
        twist.linear.x = action['linear_x']  # Linear velocity
        twist.angular.z = action['angular_z']  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Wait for the next state
        rclpy.spin_once(self.node)

        # Example reward calculation
        reward = -np.sum(np.square(self.robot_state))

        done = self.is_done(self.robot_state)

        state = {'lidar': self.lidar_data, 'position': self.robot_state[:3], 'orientation': self.robot_state[3:]}

        return state, reward, done, {}

    def reset(self, seed=None, options=None):
        # Handle the seed for random number generation
        super().reset(seed=seed)

        # Reset the robot and simulation
        self.publisher.publish(String(data='reset'))
        rclpy.spin_once(self.node)

        return self.robot_state, {}

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