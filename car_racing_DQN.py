# https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae
import argparse
import itertools
import random
from collections import deque

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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        hidden = 256

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 


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
        epsilon_decay_steps: int = 10000,
        target_update_freq: int = 1000,
        device: torch.device | None = None,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        state_dim = env.observation_space["state"].shape[0]

        self.actions_table = discretize_action_space(env.action_space, bins_per_dim)
        self.num_actions = self.actions_table.shape[0]

        # Q network + target network
        self.q_net = QNetwork(state_dim, self.num_actions).to(self.device)
        self.target_q_net = QNetwork(state_dim, self.num_actions).to(self.device)
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

    def train_step(self):
        if len(self.replay) < self.min_replay_size:
            return

        states, action_idx, rewards, next_states, terminated, truncated = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_idx_t = torch.tensor(action_idx, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        terminated_t = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(-1)
        truncated_t = torch.tensor(truncated, dtype=torch.float32, device=self.device).unsqueeze(-1)
        # Q(s,a) for current network
        q_all = self.q_net(states_t)
        q_sa = q_all.gather(1, action_idx_t.unsqueeze(-1))

        # target Q(s',a')
        with torch.no_grad():
            q_next_all = self.target_q_net(next_states_t)
            q_next_max, _ = torch.max(q_next_all, dim=1, keepdim=True)

            bootstrap_mask = 1.0 - terminated_t
            target = rewards_t + self.gamma * bootstrap_mask * q_next_max

        loss = nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.global_step % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train(self, total_timesteps: int):
        obs, _ = self.env.reset()
        state = self._obs_to_state(obs)

        episode_reward = 0.0
        episode = 0
        episode_returns = []

        for t in range(total_timesteps):
            self._update_epsilon()

            action_idx, action = self.select_action(state)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self._obs_to_state(next_obs)
            done = bool(terminated or truncated)

            self.replay.push(state, action_idx, float(reward), next_state, float(terminated), float(truncated))
            self.train_step()

            state = next_state
            episode_reward += float(reward)

            episode_done = bool(terminated or truncated)
            if episode_done:
                episode += 1
                episode_returns.append(episode_reward)
                print(f"[DQN] Ep {episode} | step={t+1} | return={episode_reward:.2f} | eps={self.epsilon:.3f} | terminated={terminated} truncated={truncated}")
                obs, _ = self.env.reset()
                state = self._obs_to_state(obs)
                episode_reward = 0.0

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
    parser.add_argument("--total-timesteps", type=int, default=20000)
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # render_mode = "human" if args.render else None
    render_mode = "human"  # For testing purpose

    env = make_env(
        render_mode=render_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )

    agent = DQNCarRacingAgent(
        env,
        bins_per_dim=tuple(args.bins),
    )

    episode_returns = agent.train(total_timesteps=args.total_timesteps)
    env.close()

    # # Plot for learning Output 
    # if len(episode_returns) > 0:
    #     plt.figure()
    #     plt.plot(episode_returns, label="Episode return")

    #     window = max(1, len(episode_returns) // 20)
    #     if window > 1:
    #         cumsum = np.cumsum(np.insert(episode_returns, 0, 0))
    #         smooth = (cumsum[window:] - cumsum[:-window]) / float(window)
    #         plt.plot(
    #             np.arange(window - 1, len(episode_returns)),
    #             smooth,
    #             label=f"Moving avg (window={window})",
    #         )

    #     plt.xlabel("Episode")
    #     plt.ylabel("Return")
    #     plt.title("DQN (discretized actions) on Custom CarRacing")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("dqn_car_racing_returns.png", dpi=150)
    #     plt.show()
    # else:
    #     print("No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()