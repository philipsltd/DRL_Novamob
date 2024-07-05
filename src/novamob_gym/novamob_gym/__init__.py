from gymnasium.envs.registration import register

register(
    id='NovamobGym-v0',
    entry_point='novamob_gym.novamob_env:NovamobGym',
)
