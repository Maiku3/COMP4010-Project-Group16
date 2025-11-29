# https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae
import argparse
from datetime import datetime
import itertools
import random
from collections import deque
import os

import numpy as np
import matplotlib.pyplot as plt

# pip install torch
import torch
import torch.nn as nn
import torch.optim as optim

from envs.car_racing import CarRacing

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, terminated, truncated):
        self.buffer.append((state, action_idx, reward, next_state, terminated, truncated))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, action_idx, rewards, next_states, terminated, truncated = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(action_idx, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminated, dtype=np.float32),
            np.array(truncated, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

def discretize_action_space(action_space, bins_per_dim=(7, 3, 2, 2)):
    low = action_space.low
    high = action_space.high
    action_dim = low.shape[0]

    if len(bins_per_dim) == 1:
        bins_per_dim = bins_per_dim * action_dim
    assert len(bins_per_dim) == action_dim, "bins_per_dim must match action_dim or be length 1"

    grids = []
    for d in range(action_dim):
        grids.append(np.linspace(low[d], high[d], bins_per_dim[d], dtype=np.float32))

    actions = np.array(list(itertools.product(*grids)), dtype=np.float32)
    return actions


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        hidden = 256

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        adv = self.advantage(feat)
        val = self.value(feat)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = val + (adv - adv_mean)
        return q


class DQNCarRacingAgent:
    def __init__(
        self,
        env: CarRacing,
        bins_per_dim=(7, 3, 2, 2),
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        min_replay_size: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        target_update_freq: int = 1000,
        reward_clip: float = 50.0,
        device: torch.device | None = None,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.reward_clip = reward_clip

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        state_dim = env.observation_space["state"].shape[0]

        self.actions_table = discretize_action_space(env.action_space, bins_per_dim)
        self.num_actions = self.actions_table.shape[0]

        # Q network + target network
        self.q_net = DuelingQNetwork(state_dim, self.num_actions).to(self.device)
        self.target_q_net = DuelingQNetwork(state_dim, self.num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(capacity=replay_capacity)

        # Epsilon-greedy 
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.global_step = 0
        self.gradient_steps = 0

        # Reward shaping knobs
        self.progress_coef = 60.0
        self.center_penalty = 0.5
        self.offtrack_penalty = 20.0
        self.pit_bonus = 25.0
        self.speed_target = 35.0
        self.speed_coef = 0.05
        self.prev_ell = 0.0

    # ===== saving ======

    def save_checkpoint(self, checkpoint_path: str, episode: int, episode_returns):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ckpt = {
            "episode": episode,
            "global_step": self.global_step,
            "gradient_steps": self.gradient_steps,
            "q_net_state_dict": self.q_net.state_dict(),
            "target_q_net_state_dict": self.target_q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_returns": episode_returns,
            "actions_table": self.actions_table,
        }
        torch.save(ckpt, checkpoint_path)
        print(f"[DQN] Saved checkpoint to {checkpoint_path}")
    @staticmethod
    def _obs_to_state(obs) -> np.ndarray:
        return np.asarray(obs["state"], dtype=np.float32)

    def _update_epsilon(self):
        self.global_step += 1
        frac = min(1.0, self.global_step / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray):
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.num_actions)
        else:
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, state_dim)
            with torch.no_grad():
                q_values = self.q_net(s_t)  # (1, num_actions)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return action_idx, self.actions_table[action_idx]

    def _shape_reward(self, obs, next_obs, env_reward: float, info: dict | None):
        if info is None:
            info = {}

        # Base signals from state vector
        d_t = float(next_obs["state"][0])
        v_t = float(next_obs["state"][1])
        off_track = bool(next_obs["state"][2] > 0.5)
        ell = float(next_obs["state"][4])

        delta_ell = max(0.0, ell - self.prev_ell)
        progress_r = self.progress_coef * delta_ell

        center_r = -self.center_penalty * abs(d_t)
        speed_r = self.speed_coef * min(1.0, v_t / self.speed_target)
        offtrack_r = -self.offtrack_penalty if off_track else 0.0
        pit_r = self.pit_bonus if info.get("pit_executed", False) else 0.0

        shaped = env_reward + progress_r + center_r + speed_r + offtrack_r + pit_r
        shaped = np.clip(shaped, -self.reward_clip, self.reward_clip)
        self.prev_ell = ell
        return float(shaped)

    def train_step(self):
        if len(self.replay) < self.min_replay_size:
            return

        states, action_idx, rewards, next_states, terminated, truncated = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_idx_t = torch.tensor(action_idx, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        terminated_t = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(-1)
        # Q(s,a) for current network
        q_all = self.q_net(states_t)
        q_sa = q_all.gather(1, action_idx_t.unsqueeze(-1))

        # target Q(s',a')
        with torch.no_grad():
            # Double DQN target: select action with online net, evaluate with target net
            q_next_online = self.q_net(next_states_t)
            next_action = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target = self.target_q_net(next_states_t).gather(1, next_action)

            bootstrap_mask = 1.0 - terminated_t
            target = rewards_t + self.gamma * bootstrap_mask * q_next_target

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train(self, total_timesteps: int, checkpoint_dir: str | None = None, save_every_episodes: int | None = None):
        obs, _ = self.env.reset()
        state = self._obs_to_state(obs)
        self.prev_ell = float(obs["state"][4])

        episode_reward = 0.0  # shaped return
        episode_env_reward = 0.0
        episode = 0
        episode_returns = []

        for t in range(total_timesteps):
            self._update_epsilon()

            action_idx, action = self.select_action(state)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_state = self._obs_to_state(next_obs)
            done = bool(terminated or truncated)

            shaped_reward = self._shape_reward(obs, next_obs, float(reward), info)
            self.replay.push(state, action_idx, shaped_reward, next_state, float(terminated), float(truncated))
            self.train_step()

            state = next_state
            obs = next_obs
            episode_reward += shaped_reward
            episode_env_reward += float(reward)

            episode_done = bool(terminated or truncated)
            if episode_done:
                episode += 1
                episode_returns.append(episode_reward)
                print(f"[DQN] Ep {episode} | step={t+1} | shaped_return={episode_reward:.2f} | env_return={episode_env_reward:.2f} | eps={self.epsilon:.3f} | terminated={terminated} truncated={truncated}")
                obs, _ = self.env.reset()
                state = self._obs_to_state(obs)
                self.prev_ell = float(obs["state"][4])
                episode_reward = 0.0
                episode_env_reward = 0.0

                # Periodic checkpoint saving
                if checkpoint_dir is not None and save_every_episodes is not None:
                    if episode % save_every_episodes == 0:
                        ckpt_path = os.path.join(
                            checkpoint_dir,
                            f"dqn_carracing_ep{episode}.pt",
                        )
                        self.save_checkpoint(ckpt_path, episode, episode_returns)

        return episode_returns


def make_env(render_mode=None, seed: int = 0, max_episode_steps: int = 100000) -> CarRacing:
    env = CarRacing(
        render_mode=render_mode,
        continuous=True,
        lap_complete_percent=0.95,
        reward_shaping=True,
        max_episode_steps=max_episode_steps,
    )
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bins",
        nargs=4,
        type=int,
        default=[7, 3, 2, 2],
        metavar=("STEER", "GAS", "BRAKE", "PIT"),
        help="Number of discrete bins per action dimension",
    )
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_dqn")
    parser.add_argument("--save-every-episodes", type=int, default=50)
    args = parser.parse_args()

    # Generate a unique timestamp for the training run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = "human" if args.render else None
    # render_mode = "human" # For testing purpose

    env = make_env(
        render_mode=render_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNCarRacingAgent(
        env,
        bins_per_dim=tuple(args.bins),
        device=device,
    )

    episode_returns = agent.train(
        total_timesteps=args.total_timesteps,
        checkpoint_dir=run_checkpoint_dir,
        save_every_episodes=args.save_every_episodes,
    )
    env.close()

        # Final checkpoint
    if len(episode_returns) > 0:
        final_ckpt_path = os.path.join(run_checkpoint_dir, "dqn_carracing_final.pt")
        agent.save_checkpoint(final_ckpt_path, episode=len(episode_returns), episode_returns=episode_returns)
        print(f"[DQN] Final checkpoint saved to {final_ckpt_path}")

    # Plot and persist learning output for later comparison
    if len(episode_returns) > 0:
        returns_arr = np.asarray(episode_returns, dtype=np.float32)

        plt.figure()
        plt.plot(returns_arr, label="Episode return")

        window = max(1, len(returns_arr) // 20)
        if window > 1:
            kernel = np.ones(window) / float(window)
            smooth = np.convolve(returns_arr, kernel, mode="valid")
            plt.plot(
                np.arange(window - 1, len(returns_arr)),
                smooth,
                label="Moving avg",
            )

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("DQN on Custom CarRacing")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dqn_car_racing_returns.png", dpi=150)
        plt.close()
    else:
        print("No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()
