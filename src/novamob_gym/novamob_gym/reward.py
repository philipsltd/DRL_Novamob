import numpy as np
from common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT, TIME_DELTA, TRACK_WIDTH
from common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER, REWARD_FUNCTION

# Initialize global variable
initial_goal_distance = 0.0

# TODO - Implement reward functions

# perfrom reward when the robot is close to the middle of the road through the measurement of the lidar distance. add 3 levels of obstacle distance that represent the desired distance to the walls.

# use the angle of the robot to slow down during turning and increase speed when going straight


# Step 1: Define your functions
def get_reward_1(cummulative_reward, robot_status, obstacle_distance, heading_angle, linear_speed, distance_to_goal):
    global initial_goal_distance

    # marker_1 = 0.1 * TRACK_WIDTH
    # marker_2 = 0.25 * TRACK_WIDTH
    # marker_3 = 0.45 * TRACK_WIDTH   # since the obstable_distance is the distance between the robot and the wall the maximum reward comes when it is driving between 0.45 and 0.5 of the track width. obstacle distance is always smaller or equal to the half the track width

    # Define the ideal distance to the walls (half the track width)
    ideal_distance = 0.45 * TRACK_WIDTH

    print(f"[DEBUG] initial goal distance: {initial_goal_distance}")
    print(f"[DEBUG] distance to goal: {distance_to_goal}")
    print(f"[DEBUG] difference: {initial_goal_distance- distance_to_goal}")

    # Distance to goal reward
    if distance_to_goal < initial_goal_distance:
        cummulative_reward += 2.0
    elif distance_to_goal >= initial_goal_distance:
        cummulative_reward -= 1.0

    # Reward based on how close the robot is to the ideal distance
    distance_error = abs(obstacle_distance - ideal_distance)
    distance_reward = max(0, 5.0 - (distance_error / ideal_distance) * 5.0)  # Reward decreases as error increases
    cummulative_reward += distance_reward

    if heading_angle >= -5 and heading_angle <= 5:
        if linear_speed > 0.8:
            cummulative_reward += 1.0
        elif linear_speed > 0.5:
            cummulative_reward += 0.5
    elif abs(heading_angle) > 20:
        if linear_speed < 0.8:
            cummulative_reward += 0.5
        elif linear_speed < 0.6:
            cummulative_reward += 1.0

    if linear_speed < 0.0:
        cummulative_reward -= 5.0

    if robot_status == GOAL_REACHED:
        cummulative_reward += 100.0
    elif robot_status == TIMEOUT:
        cummulative_reward -= 20.0
    elif robot_status == COLLISION or robot_status == ROLLED_OVER:
        cummulative_reward -= 15.0

    # Baseline reward for each step
    cummulative_reward += 0.1
    initial_goal_distance = distance_to_goal

    return float(cummulative_reward)

def get_reward_2(robot_status):
    reward -= 20.0
    return float(reward)

def get_reward_3(robot_status):
    reward = -30.0
    return float(reward)

# Step 2: Create a dictionary mapping function names to function objects
function_dict = {
    "1": get_reward_1,
    "2": get_reward_2,
    "3": get_reward_3
}

# Step 3: Implement the function selector based on the constant from the settings file
def get_reward(cummulative_reward, robot_status, obstacle_distance, heading_angle, linear_speed, distance_to_goal):
    # Get the choice from the settings file
    function_to_use = REWARD_FUNCTION
    
    # Check if the choice exists in the dictionary
    if function_to_use in function_dict:
        # Call the selected function
        reward = function_dict[function_to_use](cummulative_reward, robot_status, obstacle_distance, heading_angle, linear_speed, distance_to_goal)
        return reward
    else:
        raise ValueError("Invalid choice in settings! Please select a valid function name.")

def reward_init(init_goal_distance):
    global initial_goal_distance
    initial_goal_distance = init_goal_distance