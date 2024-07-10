from common.settings import MAX_EPISODE_TIME, GOAL_THRESHOLD, COLLISION_DISTANCE, MAX_TILT, TIME_DELTA
from common.settings import UNKNOWN, GOAL_REACHED, COLLISION, TIMEOUT, ROLLED_OVER, REWARD_FUNCTION

# reward.py

initial_goal_distance = 0.0
reward_function_selected = None

def get_reward():
    return reward_function_selected()

def get_reward_1():
    reward = -10.0
    return float(reward)

def reward_init(init_goal_distance):
    global initial_goal_distance
    initial_goal_distance = init_goal_distance

def setup_reward_function(reward_function_name):
    global reward_function_selected
    function_to_use = "get_reward_" + reward_function_name
    if function_to_use in globals():
        reward_function_selected = globals()[function_to_use]
    else:
        raise Exception(f"Reward function {reward_function_name} not found")

# Initialize the reward function
setup_reward_function(REWARD_FUNCTION)
