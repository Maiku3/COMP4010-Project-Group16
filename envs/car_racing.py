import gymnasium as gym
import numpy as np

def main():
    env = gym.make("CarRacing-v3", render_mode="human")

    try:
        obs, info = env.reset(seed=0)
        ep = 1
        ep_return, ep_steps = 0.0, 0

        while True:
            # random agent
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_steps += 1

            # Episode end conditions per Gymnasium API
            if terminated or truncated:
                reason = (
                    "terminated" if terminated and not truncated
                    else ("truncated" if not info.get("TimeLimit.truncated", False)
                          else "truncated by TimeLimit")
                )
                print(f"[DONE] Episode {ep} | steps={ep_steps} | return={ep_return:.2f} | reason={reason}")
                # Start next episode
                ep += 1
                ep_return, ep_steps = 0.0, 0
                obs, info = env.reset()

    finally:
        env.close()

if __name__ == "__main__":
    main()


