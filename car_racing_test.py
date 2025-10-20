import gymnasium as gym
from envs.car_racing import CarRacing

def main():
    env = CarRacing(render_mode="human", continuous=True, lap_complete_percent=0.95)

    try:
        observation, info = env.reset(seed=0)
        episode = 1
        episode_return, episode_steps = 0.0, 0

        while True:
            # random agent
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_steps += 1

            if terminated or truncated:
                if terminated:
                    reason = "terminated"
                else:
                    reason = "truncated"
                print(f"Episode {episode} | steps={episode_steps} | return={episode_return:.2f} | reason={reason}")
                
                # Start next episode
                episode += 1
                episode_return, episode_steps = 0.0, 0
                observation, info = env.reset()

    finally:
        env.close()

if __name__ == "__main__":
    main()
    
