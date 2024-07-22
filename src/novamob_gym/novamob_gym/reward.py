from common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT, TIME_DELTA, TRACK_WIDTH
from common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER, REWARD_FUNCTION

# Initialize global variable
initial_goal_distance = 0.0

# TODO - Implement reward functions

# ! perfrom reward when the robot is close to the middle of the road through the measurement of the lidar distance. add 3 levels of obstacle distance that represent the desired distance to the walls.

# ! use the angle of the robot to slow down during turning and increase speed when going straight


# Step 1: Define your functions
def get_reward_1(cummulative_reward, robot_status, obstacle_distance, heading_angle):
    
    marker_1 = 0.1 * TRACK_WIDTH
    marker_2 = 0.25 * TRACK_WIDTH
    marker_3 = 0.45 * TRACK_WIDTH   # since the obstable_distance is the distance between the robot and the wall the maximum reward comes when it is driving between 0.45 and 0.5 of the track width. obstacle distance is always smaller or equal to the half the track width

    if obstacle_distance >= marker_1:
        cummulative_reward += 2.0
    elif obstacle_distance >= marker_2:
        cummulative_reward += 5.0
    elif obstacle_distance >= marker_3:
        cummulative_reward += 10.0
    
    # ! need to add the angle of the robot to the reward function

    if robot_status == GOAL_REACHED:
        cummulative_reward += 100
    elif robot_status == TIMEOUT:
        cummulative_reward -= 50.0
    elif robot_status == COLLISION or robot_status == ROLLED_OVER:
        cummulative_reward -= 20.0

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
def get_reward(cummulative_reward, robot_status, obstacle_distance, heading_angle):
    # Get the choice from the settings file
    function_to_use = REWARD_FUNCTION
    
    # Check if the choice exists in the dictionary
    if function_to_use in function_dict:
        # Call the selected function
        reward = function_dict[function_to_use](cummulative_reward, robot_status, obstacle_distance, heading_angle)
        return reward
    else:
        raise ValueError("Invalid choice in settings! Please select a valid function name.")

def reward_init(init_goal_distance):
    global initial_goal_distance
    initial_goal_distance = init_goal_distance