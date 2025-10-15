import gymnasium as gym
import numpy as np

env = gym.make(
    "CarRacing-v3",
    render_mode="human",         
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=False             
)

obs, info = env.reset()
done = False


while not done:
    action = env.action_space.sample()   
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()