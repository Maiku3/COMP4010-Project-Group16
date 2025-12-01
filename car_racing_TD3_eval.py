import numpy as np
from envs.car_racing import CarRacing
from car_racing_TD3 import TD3CarRacingAgent, TD3Config, make_env

if __name__ == "__main__":
    env = make_env(render_mode="human", seed=0, max_episode_steps=3000)
    cfg = TD3Config(total_steps=1, max_ep_len=3000, render=True)
    agent = TD3CarRacingAgent(env, cfg)
    agent.load("td3_car_racing_actor_250k.pth")

    obs, _ = env.reset(seed=0)
    state = agent._obs_to_state(obs)
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True, noise_scale=0.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        state = agent._obs_to_state(next_obs)
        done = terminated or truncated
        env.render()

    env.close()
