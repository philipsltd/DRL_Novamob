import time
import math
import threading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from tf_transformations import euler_from_quaternion

# ROS 2 message imports
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

# ROS 2 Service imports
from std_srvs.srv import Empty

# -- Environment Constants --
from common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT, TIME_DELTA
# -- Possible Outcomes --
from common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER
# -- Topic Names --
from common.settings import VEL_TOPIC, ODOM_TOPIC, LIDAR_TOPIC

# -- Reward Functions --
import reward as rw


class NovamobGym(gym.Env):
    def __init__(self):
        super(NovamobGym, self).__init__()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('gym_novamob_env')

        # * Define action and observation space
        # the action space is a box with two elements: linear_x and angular_z velocities
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # the observation space is a continuous space with 1080 values from the lidar sensor, a continuous position space with 2 values (x, y), and a continuous robot tilt space with 2 values (roll, pitch)
        self.observation_space = spaces.Dict({'lidar': spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32),
                                              'position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                                              'robot_tilt': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
                                              })

        # Clients
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        self.pause_client = self.node.create_client(Empty, '/gazebo/pause_physics')
        self.unpause_client = self.node.create_client(Empty, '/gazebo/unpause_physics')

        # QoS profile for clock subscriber
        clock_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Publishers 
        self.cmd_vel_publisher = self.node.create_publisher(Twist, VEL_TOPIC, 1)

        # Subscribers
        self.odom_subscription = self.node.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.lidar_subscription = self.node.create_subscription(LaserScan, LIDAR_TOPIC, self.lidar_callback, 1)
        self.clock_subscription = self.node.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=clock_qos_profile)

        # Place holder and initialization for data
        self.lidar_data = np.zeros(360, dtype=np.float32)
        self.goal_distance = np.inf
        self.obstacle_distance = np.inf
        self.robot_state = np.zeros(2, dtype=np.float32)
        self.robot_tilt = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.current_time = 0
        self.episode_deadline = np.inf
        self.robot_status = UNKNOWN
        self.cummulative_reward = 0.0

        self.reset_flag = False

        # Flags to check if data is updated
        self.lidar_updated = threading.Event()
        self.odom_updated = threading.Event()
        self.clock_updated = threading.Event()

        # Initialize the random number generator
        self.np_random = np.random.RandomState(42)

        # Start the MultiThreadedExecutor
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        # Initialize the goal position and reward function
        self.goal_index = 0
        self.goal_array = [(3.0, 0.0), (3.0, 1.82), (0.0, 1.0), (0.0, 0.0)]
        self.goal_x = 3.0
        self.goal_y = 0.0
        # self.change_goal()
        self.goal_distance = np.sqrt((self.robot_state[0] - self.goal_x) ** 2 + (self.robot_state[1] - self.goal_y) ** 2)
        rw.reward_init(self.goal_distance)


    def odom_callback(self, msg):
        # Store the robot state and tilt
        self.robot_state[0] = msg.pose.pose.position.x
        self.robot_state[1] = msg.pose.pose.position.y

        # Extract quaternion from odometry message
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.robot_tilt[0] = roll
        self.robot_tilt[1] = pitch

        # Convert yaw from radians to degrees
        self.heading = math.degrees(yaw) % 360
        if self.heading >= 180:
            self.heading -= 360

        self.odom_updated.set()


    def lidar_callback(self, msg):
        lidar_readings = [min(x, 10.0) if x == float('inf') else x for x in msg.ranges]  # Downsampling
        lidar_features = self.process_lidar_readings(lidar_readings)
        self.lidar_data = np.array(list(lidar_features.values()), dtype=np.float32)
        self.obstacle_distance = lidar_features['min_distance']
        self.lidar_updated.set()


    def process_lidar_readings(self, readings):
        features = {
            'min_distance': float(np.min(readings)),
            'max_distance': float(np.max(readings)),
            'average_distance': float(np.mean(readings)),
            'front_distance': float(np.min(readings[len(readings)//2 - 5:len(readings)//2 + 5])),  # Distance directly in front
            'left_distance': float(np.min(readings[:10])),  # Distance to the left
            'right_distance': float(np.min(readings[-10:]))  # Distance to the right
        }
        return features


    def clock_callback(self, msg):
        # Store the time elapsed
        self.current_time = msg.clock.sec
        self.clock_updated.set()


    def step(self, action):
        # Wait until data from all topics has been read at least once
        while not (self.lidar_updated.is_set() and self.odom_updated.is_set() and self.clock_updated.is_set()):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Reset the update flags
        self.lidar_updated.clear()
        self.odom_updated.clear()
        self.clock_updated.clear()

        # Send action to robot
        twist = Twist()
        twist.linear.x = float(action[0])  # Linear velocity
        twist.angular.z = float(action[1])  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Unpause the simulation to propagate the state
        self.unpause_simulation()

        # propagate state for TIME_DELTA seconds
        for _ in range(int(TIME_DELTA)):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Pause the simulation after propagating state
        self.pause_simulation()


        # Check the status of the robot
        done = self.get_status()

        print(f"[DEBUG] goal distance: {self.goal_distance}")

        self.goal_distance = np.sqrt((self.robot_state[0] - self.goal_x) ** 2 + (self.robot_state[1] - self.goal_y) ** 2)

        print(f"[DEBUG] new goal distance: {self.goal_distance} and previous reward: {self.cummulative_reward}")

        reward = rw.get_reward(self.cummulative_reward, self.robot_status, self.obstacle_distance, self.heading, twist.linear.x, self.goal_distance, self.reset_flag)
        self.cummulative_reward = reward

        self.reset_flag = False

        state = {'lidar': self.lidar_data.copy(), 'position': self.robot_state.copy(), 'robot_tilt': self.robot_tilt.copy()}

        # Ensure the state is within the observation space and has the correct dtype
        state = {k: np.asarray(v, dtype=self.observation_space[k].dtype) for k, v in state.items()}

        # New API requires `terminated` and `truncated` flags
        terminated = done
        truncated = False

        print(f"[DEBUG] cummulative: {self.cummulative_reward}")

        return state, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)

        # Stop the robot and reset the episode
        self.cmd_vel_publisher.publish(Twist())  # stop robot
        self.episode_deadline = self.current_time + MAX_EPISODE_TIME

        # Reset the gazebo simulation
        self.reset_simulation()

        # Wait until data from all topics has been read at least once
        while not (self.lidar_updated.is_set() and self.odom_updated.is_set() and self.clock_updated.is_set()):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Reset the update flags
        self.lidar_updated.clear()
        self.odom_updated.clear()
        self.clock_updated.clear()

        self.current_time = 0
        self.goal_index = 0
        # self.change_goal()
        rw.reward_init(self.goal_distance)

        self.reset_flag = True

        self.robot_status = UNKNOWN
        self.goal_distance = np.sqrt((self.robot_state[0] - self.goal_x) ** 2 + (self.robot_state[1] - self.goal_y) ** 2)

        state = {'lidar': self.lidar_data.copy(), 'position': self.robot_state.copy(), 'robot_tilt': self.robot_tilt.copy()}

        # Ensure the state is within the observation space and has the correct dtype
        state = {k: np.asarray(v, dtype=self.observation_space[k].dtype) for k, v in state.items()}

        return state, {}


    def get_status(self):
        self.robot_status = UNKNOWN

        if self.goal_distance < GOAL_THRESHOLD:
            # if self.goal_index != 3:
            #     self.change_goal()
            # else:
                self.robot_status = GOAL_REACHED
        elif self.obstacle_distance < COLLISION_DISTANCE:
            self.robot_status = COLLISION
        elif self.current_time >= self.episode_deadline:
            self.robot_status = TIMEOUT
        elif (self.robot_tilt > MAX_TILT).any() or (self.robot_tilt < -MAX_TILT).any():
            self.robot_status = ROLLED_OVER

        if self.robot_status != UNKNOWN:
            self.cmd_vel_publisher.publish(Twist())
            return True
        return False


    # def change_goal(self):
    #     if self.goal_index == len(self.goal_array):
    #         self.goal_index = 0
    #     self.goal_x = self.goal_array[self.goal_index][0]
    #     self.goal_y = self.goal_array[self.goal_index][1]
    #     self.goal_index += 1

    #     print(f"[DEBUG] New goal: ({self.goal_x}, {self.goal_y})")


    def __del__(self):
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread.is_alive():
            self.executor.shutdown()
            self.executor_thread.join()


    def unpause_simulation(self):
        if self.unpause_client.wait_for_service(timeout_sec=1.0):
            future = self.unpause_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            if not future.done():
                self.node.get_logger().error('Failed to unpause simulation')


    def pause_simulation(self):
        if self.pause_client.wait_for_service(timeout_sec=1.0):
            future = self.pause_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            if not future.done():
                self.node.get_logger().error('Failed to pause simulation')


    def reset_simulation(self):
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