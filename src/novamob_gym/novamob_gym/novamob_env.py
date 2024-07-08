import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node

# ROS 2 message imports
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

# ROS 2 Service imports
from std_srvs.srv import Empty

# -- Environment Constants --
from .common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT
# -- Possible Outcomes --
from .common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER

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

        # Clients
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')

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
        self.clock_subscription = self.node.create_subscription(Clock,
                                                                '/clock',
                                                                self.clock_callback,
                                                                10)
        
        # Place holder and initialization for data
        self.lidar_data = np.zeros(1080)
        self.obstacle_distance = np.inf
        self.robot_state = np.zeros(2)
        self.robot_tilt = np.zeros(2)
        self.current_time = 0
        self.episode_deadline = np.inf


    def odom_callback(self, msg):
        # Store the robot state and tilt
        self.robot_state[0] = msg.pose.pose.position.x
        self.robot_state[1] = msg.pose.pose.position.y
        self.robot_tilt[0] = msg.pose.pose.orientation.x
        self.robot_tilt[1] = msg.pose.pose.orientation.y
        print(f"Listening to position: {self.robot_state} and tilt: {self.robot_tilt}")


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
        self.obstacle_distance = np.min(self.lidar_data)


    def clock_callback(self, msg):
        # Store the time elapsed
        self.current_time = msg.clock.sec


    def step(self, action):
        # Send action to robot
        twist = Twist()
        twist.linear.x = action['linear_x'][0]  # Linear velocity
        twist.angular.z = action['angular_z'][0]  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Check the status of the robot
        # Verifies the time elapsed, distance to the goal, distance to obstacles, and robot tilt and concludes if the episode is done
        done = self.is_done()

        # Calculate the reward
        # TODO - Implement the reward function
        reward = -np.sum(np.square(self.robot_state))

        state = {'lidar': self.lidar_data, 'position': self.robot_state, 'robot_tilt': self.robot_tilt}

        return state, reward, done, {}

    def reset(self):
        # Reset the gazebo simulation
        rclpy.spin_once(self.node)
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            reset_req = Empty.Request()
            future = self.reset_client.call_async(reset_req)
            rclpy.spin_until_future_complete(self.node, future)
            if future.result() is not None:
                self.node.get_logger().info('Simulation reset completed')
            else:
                self.node.get_logger().error('Failed to reset simulation')
        else:
            self.node.get_logger().error('Reset service not available')

        # Stop the robot and reset the episode
        self.cmd_vel_publisher.publish(Twist())  # stop robot
        self.episode_deadline = self.current_time + MAX_EPISODE_TIME
        self.done = False

        self.robot_state = np.zeros(2)
        self.robot_tilt = np.zeros(2)
        self.obstacle_distance = np.inf

        return {'position': self.robot_state, 'robot_tilt': self.robot_tilt, 'lidar': self.lidar_data}

    def is_done(self):
        # Define a condition to end the episode
        return False

    def close(self):
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # def render(self, mode='human'):
    #     # Render the environment for visualization
    #     pass

    def __del__(self):
        self.close()

def get_status(self):
    self.robot_status = UNKNOWN

    if self.goal_distance < GOAL_THRESHOLD:
        self.robot_status = GOAL_REACHED
    elif self.obstacle_distance < COLLISION_DISTANCE:
        self.robot_status = COLLISION
    elif self.current_time >= self.episode_deadline:
        self.robot_status = TIMEOUT
    elif self.robot_tilt > MAX_TILT or self.robot_tilt < MAX_TILT:
        self.robot_status = ROLLED_OVER

    if self.robot_status != UNKNOWN:
        done = True
    return done

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
