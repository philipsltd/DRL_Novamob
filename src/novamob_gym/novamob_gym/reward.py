from common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT, TIME_DELTA
from common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER, REWARD_FUNCTION

# Initialize global variable
initial_goal_distance = 0.0

# TODO - Implement reward functions

# Step 1: Define your functions
def get_reward_1(robot_status):
    
    if robot_status == TIMEOUT:
        reward -= 10.0
    elif robot_status == COLLISION:
        reward -= 20.0
    else:
        reward += 10.0
    return float(reward)

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
def get_reward(robot_status):
    # Get the choice from the settings file
    function_to_use = REWARD_FUNCTION
    
    # Check if the choice exists in the dictionary
    if function_to_use in function_dict:
        # Call the selected function
        reward = function_dict[function_to_use](robot_status)
        return reward
    else:
        raise ValueError("Invalid choice in settings! Please select a valid function name.")

def reward_init(init_goal_distance):
    global initial_goal_distance
    initial_goal_distance = init_goal_distance