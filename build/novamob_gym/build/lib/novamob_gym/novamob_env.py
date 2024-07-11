import time
import threading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

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
        # the action space is a dictionary with two keys: linear_x and angular_z velocities
        self.action_space = spaces.Dict({'linear_x': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                                         'angular_z': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)})
        
        # the observation space is a continuous space with 1080 values from the lidar sensor
        self.observation_space = spaces.Dict({'lidar': spaces.Box(low=-np.inf, high=np.inf, shape=(360,), dtype=np.float32),
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
        self.cmd_vel_publisher = self.node.create_publisher(Twist, 
                                                            VEL_TOPIC,
                                                            1)

        # Subscribers
        self.odom_subscription = self.node.create_subscription(Odometry,
                                                               ODOM_TOPIC,
                                                               self.odom_callback,
                                                               10)
        self.lidar_subscription = self.node.create_subscription(LaserScan,
                                                                LIDAR_TOPIC,
                                                                self.lidar_callback,
                                                                1)
        self.clock_subscription = self.node.create_subscription(Clock,
                                                                '/clock',
                                                                self.clock_callback,
                                                                qos_profile=clock_qos_profile)
        
        # Place holder and initialization for data
        self.lidar_data = np.zeros(360, dtype=np.float32)
        self.goal_distance = np.inf
        self.obstacle_distance = np.inf
        self.robot_state = np.zeros(2, dtype=np.float32)
        self.robot_tilt = np.zeros(2, dtype=np.float32)
        self.current_time = 0
        self.episode_deadline = np.inf
        self.robot_status = UNKNOWN
        self.cummulative_reward = 0.0

        self.lidar_read = 0
        self.odom_read = 0
        self.clock_read = 0

        # Flags to check if data is updated
        self.lidar_updated = False
        self.odom_updated = False
        self.clock_updated = False

        # Initialize the random number generator
        self.np_random = np.random.RandomState(42)

        # Initialize threading locks
        self.lidar_lock = threading.Lock()
        self.odom_lock = threading.Lock()
        self.clock_lock = threading.Lock()

        # Start the MultiThreadedExecutor
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()


    def spin_odom(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def spin_lidar(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def spin_clock(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)


    def odom_callback(self, msg):
        with self.odom_lock:
            # Store the robot state and tilt
            self.robot_state[0] = msg.pose.pose.position.x
            self.robot_state[1] = msg.pose.pose.position.y
            self.robot_tilt[0] = msg.pose.pose.orientation.x
            self.robot_tilt[1] = msg.pose.pose.orientation.y
            self.odom_updated = True

        self.odom_read += 1


    def lidar_callback(self, msg):
        with self.lidar_lock:
            # Process LiDAR data (extract X, Y, Z and potentially preprocess)
            lidar_readings = list(msg.ranges)
            # Handle infinite values returned by the LiDAR sensor
            lidar_readings = [1e6 if x == float('inf') else x for x in lidar_readings]
            # Convert the readings to a string and remove brackets
            lidar_readings_str = str(lidar_readings).replace('[', '').replace(']', '')
            # Split the string into separate elements and convert them to floats
            lidar_readings_columns = [np.float32(x) for x in lidar_readings_str.split(',')]
            # Store the LiDAR data in a numpy array
            self.lidar_data = np.array(lidar_readings_columns)
            self.obstacle_distance = np.min(self.lidar_data)
            self.lidar_updated = True

        self.lidar_read += 1


    def clock_callback(self, msg):
        with self.clock_lock:
            # Store the time elapsed
            self.current_time = msg.clock.sec
            self.clock_updated = True

        self.clock_read += 1


    def step(self, action):
        # Wait until data from all topics has been read at least once
        while not (self.lidar_updated and self.odom_updated and self.clock_updated):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        # Reset the update flags
        self.lidar_updated = False
        # self.data_ready.wait()
        self.odom_updated = False
        # self.data_ready.clear()
        self.clock_updated = False

        # Send action to robot
        twist = Twist()
        twist.linear.x = float(action['linear_x'][0])  # Linear velocity
        twist.angular.z = float(action['angular_z'][0])  # Angular velocity
        self.cmd_vel_publisher.publish(twist)

        # Unpause the simulation to propagate the state
        # This is necessary to ensure the robot moves then we pause the simulation to calculate the reward
        if self.unpause_client.wait_for_service(timeout_sec=1.0):
            try:
                self.unpause_client.call(Empty.Request())
            except (rclpy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        for _ in range(int(TIME_DELTA * 10)):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self.pause_client.wait_for_service(timeout_sec=1.0):
            try:
                self.pause_client.call(Empty.Request())
            except (rclpy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")


        # Check the status of the robot
        # Verifies the time elapsed, distance to the goal, distance to obstacles, and robot tilt to conclude the episode status
        print("Checking status...")
        print(f"Obstacle distance: {self.obstacle_distance}")
        self.get_status()
        print(f"Robot status: {self.robot_status}")

        # Calculate the reward and check if the episode is done
        done = self.is_done()
        reward = rw.get_reward(self.cummulative_reward, self.robot_status)
        self.cummulative_reward = reward

        # Acquire the locks to read the data safely
        with self.lidar_lock:
            lidar_data = self.lidar_data.copy()
        with self.odom_lock:
            robot_state = self.robot_state.copy()
            robot_tilt = self.robot_tilt.copy()

        state = {'lidar': lidar_data, 'position': robot_state, 'robot_tilt': robot_tilt}

        # Ensure the state is within the observation space and has the correct dtype
        state = {k: np.asarray(v, dtype=self.observation_space[k].dtype) for k, v in state.items()}

        # New API requires `terminated` and `truncated` flags
        terminated = done
        truncated = False

        return state, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)

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

        if self.unpause_client.wait_for_service(timeout_sec=1.0):
            try:
                self.unpause_client.call(Empty.Request())
            except (rclpy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        for _ in range(int(TIME_DELTA * 10)):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self.pause_client.wait_for_service(timeout_sec=1.0):
            try:
                self.pause_client.call(Empty.Request())
            except (rclpy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")

        with self.lidar_lock:
            lidar_data = self.lidar_data.copy()
        with self.odom_lock:
            robot_state = self.robot_state.copy()
            robot_tilt = self.robot_tilt.copy()

        rw.reward_init(self.goal_distance)
        state = {'position': robot_state, 'robot_tilt': robot_tilt, 'lidar': lidar_data}

        return state, {}


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def is_done(self):
        # Define a condition to end the episode
        if self.robot_status != UNKNOWN:
            return True
        return False


    def close(self):
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread.is_alive():
            self.executor.shutdown()
            self.executor_thread.join()


    def __del__(self):
        self.close()


    def get_status(self):
        done = False

        if self.goal_distance < GOAL_THRESHOLD:
            self.robot_status = GOAL_REACHED
        elif self.obstacle_distance < COLLISION_DISTANCE:
            self.robot_status = COLLISION
        elif self.current_time >= self.episode_deadline:
            self.robot_status = TIMEOUT
        elif (self.robot_tilt > MAX_TILT).any() or (self.robot_tilt < -MAX_TILT).any():
            self.robot_status = ROLLED_OVER

        if self.robot_status != UNKNOWN:
            done = True
        return done


def main(args=None):
    env = NovamobGym()
    # Reset the environment and get the initial observation
    obs = env.reset()
    print(f"Initial Observation: {obs}")
    
    done = False
    step_count = 0
    
    # Test loop - you can define the number of steps you want to test
    while not done and step_count < 50:  # Test for 5 steps
        # Sample a random action
        action = env.action_space.sample()
        # print(f"Step {step_count}: Action: {action}")
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print the results of the step
        # print(f"Step {step_count}: Observation: {obs}, Reward: {reward}, Done: {terminated}")
        print(f"Step: {step_count}, Reward: {reward}, Done: {terminated}")
        
        step_count += 1
    
    # Close the environment
    env.close()

if __name__ == '__main__':
    main()