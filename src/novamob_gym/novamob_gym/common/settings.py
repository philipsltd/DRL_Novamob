# Define environment constants that can be manipulated
MAX_EPISODE_TIME = 60  # seconds
GOAL_THRESHOLD = 0.1  # meters
COLLISION_DISTANCE = 0.1  # meters
MAX_TILT = 1.57  # radians = 90 degrees


# Define the possible outcomes of the episode to calculate the reward
UNKNOWN = 0
GOAL_REACHED = 1
COLLISION = 2
TIMEOUT = 3
ROLLED_OVER = 4