import gymnasium as gym
from envs.car_racing import CarRacing
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CarRacing(render_mode="human", 
                    continuous=True, 
                    lap_complete_percent=0.95, 
                    reward_shaping=True, 
                    max_episode_steps=3000)

    observation, info = env.reset(seed=args.seed)
    episode, episode_return, episode_steps = 1, 0.0, 0
    step_print_every = 50  # print telemetry every 50 frames

    while episode <= args.episodes:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(env._compute_offset())
        episode_return += float(reward)
        episode_steps += 1

        # print telemetry
        if episode_steps % step_print_every == 0:
            s = observation["state"]
            print(f"[ep {episode:02d} step {episode_steps:04d}] v={s[1]:.1f} "
                  f"fuel={s[6]:.3f} wear={s[5]:.3f} ell={s[4]:.3f}")

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"Episode {episode} | steps={episode_steps} | return={episode_return:.2f} | {reason}")
            episode += 1
            if episode <= args.episodes:
                episode_return, episode_steps = 0.0, 0
                observation, info = env.reset(seed=args.seed)
                

if __name__ == "__main__":
    main()
    
