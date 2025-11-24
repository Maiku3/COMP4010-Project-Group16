import argparse
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.car_racing import CarRacing

class RunningNorm:

    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size, device):
        max_index = self.capacity if self.full else self.pos
        idxs = np.random.randint(0, max_index, size=batch_size)

        # Explicit dtype keeps NumPy 1.x/2.x builds from tripping torch inference.
        states = torch.as_tensor(self.states[idxs], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=device)
        rewards = (
            torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device)
            .unsqueeze(1)
        )
        next_states = torch.as_tensor(
            self.next_states[idxs], dtype=torch.float32, device=device
        )
        dones = (
            torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device)
            .unsqueeze(1)
        )
        return states, actions, rewards, next_states, dones

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, log_std_bounds=(-5.0, 1.0)):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_bounds
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def _distribution(self, state):
        h = self.net(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

    def sample(self, state):
        dist = self._distribution(state)
        z = dist.rsample()  # reparameterization
        action = torch.tanh(z)
        # tanh correction for log_prob
        log_prob = dist.log_prob(z) - torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        dist = self._distribution(state)
        action = torch.tanh(dist.mean)
        return action


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

@dataclass
class SACConfig:
    total_steps: int = 300_000
    max_ep_len: int = 3000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 400_000
    start_steps: int = 10_000
    update_after: int = 5_000
    update_every: int = 1
    updates_per_step: int = 1
    target_entropy_scale: float = 1.0  # multiplied by -action_dim
    reward_clip: float = 50.0
    seed: int = 0
    render: bool = False


class SACCarRacingAgent:
    def __init__(self, env: CarRacing, cfg: SACConfig, device=None):
        self.env = env
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_low = np.asarray(env.observation_space["state"].low, dtype=np.float32)
        self.state_high = np.asarray(env.observation_space["state"].high, dtype=np.float32)
        self.state_scale = np.maximum(self.state_high - self.state_low, 1e-3)
        self.state_rms = RunningNorm(self.state_low.shape)

        # Explicit dtype to avoid inference issues with newer NumPy builds.
        self.action_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )

        state_dim = env.observation_space["state"].shape[0]
        action_dim = env.action_space.shape[0]

        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.cfg.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.cfg.lr)

        # Automatic entropy tuning
        self.target_entropy = -self.cfg.target_entropy_scale * float(action_dim)
        self.log_alpha = torch.tensor(np.log(0.2), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.alpha = float(self.log_alpha.exp().item())

        self.replay = ReplayBuffer(self.cfg.replay_size, state_dim, action_dim)

        # Reward shaping knobs (matches PPO/DQN for comparability)
        self.progress_coef = 60.0
        self.center_penalty = 0.5
        self.offtrack_penalty = 20.0
        self.pit_bonus = 25.0
        self.speed_target = 35.0
        self.speed_coef = 0.05
        self.prev_ell = 0.0

    def _obs_to_state(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32)
        state = np.clip(state, self.state_low, self.state_high)
        self.state_rms.update(state)
        normed = self.state_rms.normalize(state)
        return np.clip(normed, -5.0, 5.0)

    def _shape_reward(self, obs, next_obs, env_reward: float, info: dict | None):
        if info is None:
            info = {}
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
        shaped = np.clip(shaped, -self.cfg.reward_clip, self.cfg.reward_clip)
        self.prev_ell = ell
        return float(shaped)

    def _rescale_action(self, squashed_action: torch.Tensor):
        return self.action_low + (squashed_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.policy.deterministic(s_t)
            else:
                a, _ = self.policy.sample(s_t)
        a = self._rescale_action(a.squeeze(0))
        # Avoid torch -> numpy bridge (fails if torch was built against NumPy 1.x and NumPy 2.x is installed).
        a_list = a.detach().cpu().view(-1).tolist()
        return np.asarray(a_list, dtype=np.float32)

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.cfg.batch_size, self.device
        )

        # Critic targets
        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(next_states)
            next_actions_env = self._rescale_action(next_actions)
            q1_next = self.q1_target(next_states, next_actions_env)
            q2_next = self.q2_target(next_states, next_actions_env)
            q_next_min = torch.min(q1_next, q2_next)
            alpha_val = self.log_alpha.exp()
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * (q_next_min - alpha_val * next_logp)

        # Q1, Q2 losses
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = nn.functional.mse_loss(q1_pred, target_q)
        q2_loss = nn.functional.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # Policy loss
        new_actions, logp = self.policy.sample(states)
        new_actions_env = self._rescale_action(new_actions)
        q1_new = self.q1(states, new_actions_env)
        q2_new = self.q2(states, new_actions_env)
        q_new = torch.min(q1_new, q2_new)
        alpha_val = self.log_alpha.exp()
        policy_loss = (alpha_val * logp - q_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        # Temperature loss
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = float(self.log_alpha.exp().item())

        # Targets
        soft_update(self.q1_target, self.q1, self.cfg.tau)
        soft_update(self.q2_target, self.q2, self.cfg.tau)

    def train(self):
        cfg = self.cfg
        env = self.env

        episode_returns = []
        episode_env_returns = []

        obs, _ = env.reset(seed=cfg.seed)
        state = self._obs_to_state(obs)
        self.prev_ell = float(obs["state"][4])
        ep_ret = 0.0
        ep_env_ret = 0.0
        ep_len = 0

        for t in range(cfg.total_steps):
            if cfg.render and env.render_mode == "human":
                env.render()

            if t < cfg.start_steps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = self.select_action(state, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = self._obs_to_state(next_obs)
            shaped_reward = self._shape_reward(obs, next_obs, float(reward), info)

            done = bool(terminated or truncated)
            self.replay.store(state, action, shaped_reward, next_state, done)

            state = next_state
            obs = next_obs
            ep_ret += shaped_reward
            ep_env_ret += float(reward)
            ep_len += 1

            if t >= cfg.update_after and t % cfg.update_every == 0:
                for _ in range(cfg.updates_per_step):
                    self.update()

            timeout = ep_len >= cfg.max_ep_len
            if done or timeout:
                episode_returns.append(ep_ret)
                episode_env_returns.append(ep_env_ret)
                print(
                    f"[SAC] Ep {len(episode_returns):4d} | step={t+1:7d} | "
                    f"shaped_return={ep_ret:8.2f} | env_return={ep_env_ret:7.2f} | "
                    f"len={ep_len:4d} | alpha={self.alpha:.3f}"
                )
                obs, _ = env.reset()
                state = self._obs_to_state(obs)
                self.prev_ell = float(obs["state"][4])
                ep_ret = 0.0
                ep_env_ret = 0.0
                ep_len = 0

        env.close()
        return episode_returns, episode_env_returns


def make_env(render_mode=None, seed=0, max_episode_steps=3000):
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
    parser.add_argument("--total-steps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--render", action="store_true", help="Enable human render while training")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = "human" if args.render else None
    # render_mode = "human" 
    env = make_env(render_mode=render_mode, seed=args.seed, max_episode_steps=args.max_episode_steps)

    cfg = SACConfig(
        total_steps=args.total_steps,
        max_ep_len=args.max_episode_steps,
        seed=args.seed,
        render=args.render,
    )

    agent = SACCarRacingAgent(env, cfg)
    episode_returns, _ = agent.train()

    if len(episode_returns) > 0:
        returns_arr = np.asarray(episode_returns, dtype=np.float32)
        plt.figure()
        plt.plot(returns_arr, label="Episode return")
        window = max(1, len(returns_arr) // 20)
        if window > 1:
            kernel = np.ones(window) / float(window)
            smooth = np.convolve(returns_arr, kernel, mode="valid")
            plt.plot(np.arange(window - 1, len(returns_arr)), smooth, label="Moving avg")

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("SAC on Custom CarRacing")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("sac_car_racing_returns.png", dpi=150)
        plt.close()
    else:
        print("No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()
