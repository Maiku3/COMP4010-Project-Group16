import gymnasium as gym
from envs.car_racing import CarRacing

def main():
    env = CarRacing(render_mode="human", 
                    continuous=True, 
                    lap_complete_percent=0.95, 
                    reward_shaping=True, 
                    max_episode_steps=3000)

    # reset() - get initial state
    observation, info = env.reset(seed=0)
    episode = 1
    episode_return, episode_steps = 0.0, 0
    

    while True:
        # random agent
        # TODO: action = env.controller(observation["image"])
        action = env.action_space.sample()

        # step(action) - environment evolves and returns reward
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        episode_steps += 1

        if terminated or truncated:
            if terminated:
                reason = "terminated"
            else:
                reason = "truncated"
            # render() - view current metrics
            print(f"Episode {episode} | steps={episode_steps} | reward={episode_return:.2f} | reason={reason}")
            
            # Start next episode
            episode += 1
            episode_return, episode_steps = 0.0, 0
            observation, info = env.reset()

if __name__ == "__main__":
    main()
    
