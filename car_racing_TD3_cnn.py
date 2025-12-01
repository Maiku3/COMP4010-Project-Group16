# car_racing_TD3_cnn.py
# TD3 for custom CarRacing (continuous steer/gas/brake/pit) with CNN over image + MLP over state.
#
#   obs_vec = [ normalized_state , flattened_image(CHW in [0,1]) ]

import argparse
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.car_racing import CarRacing


# ---------------------------
# Running normalization + replay (same pattern as SAC)
# ---------------------------

class RunningNorm:
    """Track running mean/var for online state normalization."""

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
    """Simple off-policy replay buffer for TD3."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
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

        states = torch.as_tensor(self.states[idxs], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(self.next_states[idxs], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones


# ---------------------------
# TD3 networks (CNN + State)
# ---------------------------

class _SplitObsMixin:
    """Utility to split fused obs vector into (state, img[B,C,H,W])."""

    def __init__(self, state_dim: int, img_chw: tuple[int, int, int]):
        self.state_dim = int(state_dim)
        self.img_c, self.img_h, self.img_w = map(int, img_chw)
        self.flat_img_dim = int(self.img_c * self.img_h * self.img_w)

    def _split_obs(self, obs: torch.Tensor):
        """
        obs: (B, state_dim + flat_img_dim) or (state_dim + flat_img_dim,)
        returns:
          state: (B, state_dim)
          img:   (B, C, H, W)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        b = obs.shape[0]
        state = obs[:, : self.state_dim]
        img_flat = obs[:, self.state_dim : self.state_dim + self.flat_img_dim]
        img = img_flat.view(b, self.img_c, self.img_h, self.img_w)
        return state, img


def _make_cnn_encoder(img_c: int):
    # Same architecture as PPO-CNN: Conv8/4/3 with strides 4/2/1 + ReLU.
    return nn.Sequential(
        nn.Conv2d(img_c, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


class ActorCNN(nn.Module, _SplitObsMixin):
    """Deterministic actor with tanh on each action dim (later rescaled to env bounds)."""

    def __init__(self, state_dim: int, img_chw: tuple[int, int, int], action_dim: int):
        nn.Module.__init__(self)
        _SplitObsMixin.__init__(self, state_dim=state_dim, img_chw=img_chw)

        hidden = 256
        state_hidden = 64

        self.cnn = _make_cnn_encoder(self.img_c)
        with torch.no_grad():
            dummy = torch.zeros(1, self.img_c, self.img_h, self.img_w)
            cnn_out_dim = int(self.cnn(dummy).shape[1])

        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden),
            nn.Tanh(),
        )

        fused_dim = cnn_out_dim + state_hidden
        self.body = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs_vec: torch.Tensor):
        state, img = self._split_obs(obs_vec)
        img_feat = self.cnn(img)
        state_feat = self.state_mlp(state)
        fused = torch.cat([img_feat, state_feat], dim=-1)
        # Squash to [-1,1]
        return torch.tanh(self.body(fused))


class QNetworkCNN(nn.Module, _SplitObsMixin):
    """Single Q-network Q(s,a) with CNN over image + MLP over state."""

    def __init__(self, state_dim: int, img_chw: tuple[int, int, int], action_dim: int):
        nn.Module.__init__(self)
        _SplitObsMixin.__init__(self, state_dim=state_dim, img_chw=img_chw)

        hidden = 256
        state_hidden = 64

        self.cnn = _make_cnn_encoder(self.img_c)
        with torch.no_grad():
            dummy = torch.zeros(1, self.img_c, self.img_h, self.img_w)
            cnn_out_dim = int(self.cnn(dummy).shape[1])

        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden),
            nn.Tanh(),
        )

        fused_dim = cnn_out_dim + state_hidden

        self.net = nn.Sequential(
            nn.Linear(fused_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_vec: torch.Tensor, action: torch.Tensor):
        state, img = self._split_obs(obs_vec)
        img_feat = self.cnn(img)
        state_feat = self.state_mlp(state)
        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.net(torch.cat([fused, action], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging for target params."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


# ---------------------------
# Config
# ---------------------------

@dataclass
class TD3Config:
    total_steps: int = 300_000
    max_ep_len: int = 3000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 400_000
    start_steps: int = 10_000
    update_after: int = 5_000
    update_every: int = 1
    updates_per_step: int = 1

    # TD3-specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1

    reward_clip: float = 50.0
    seed: int = 0
    render: bool = False
    save_path: Optional[str] = "td3_car_racing_actor.pth"


# ---------------------------
# TD3 Agent
# ---------------------------

class TD3CarRacingAgent:
    def __init__(self, env: CarRacing, cfg: TD3Config, device=None):
        self.env = env
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State normalization (same idea as SAC/PPO, only for the state branch)
        self.state_low = np.asarray(env.observation_space["state"].low, dtype=np.float32)
        self.state_high = np.asarray(env.observation_space["state"].high, dtype=np.float32)
        self.state_scale = np.maximum(self.state_high - self.state_low, 1e-3)
        self.state_rms = RunningNorm(self.state_low.shape)

        # Image shape: env gives (H,W,C), we store flattened CHW in obs vector.
        image_shape_hwc = env.observation_space["image"].shape  # e.g. (96,96,3)
        self.img_chw = (int(image_shape_hwc[2]), int(image_shape_hwc[0]), int(image_shape_hwc[1]))  # (C,H,W)
        self.flat_img_dim = int(np.prod(self.img_chw))
        self.state_dim = int(env.observation_space["state"].shape[0])
        self.obs_dim = self.state_dim + self.flat_img_dim

        # Action bounds (Box low/high)
        self.action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)

        action_dim = env.action_space.shape[0]

        # Actor + targets
        self.actor = ActorCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.actor_target = ActorCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics + targets
        self.q1 = QNetworkCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.q2 = QNetworkCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.q1_target = QNetworkCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.q2_target = QNetworkCNN(self.state_dim, self.img_chw, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.cfg.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.cfg.lr)

        self.replay = ReplayBuffer(self.cfg.replay_size, self.obs_dim, action_dim)

        # Reward shaping knobs (same structure as SAC/PPO/DQN so results are comparable).
        self.progress_coef = 60.0
        self.center_penalty = 0.5
        self.offtrack_penalty = 20.0
        self.pit_bonus = 25.0
        self.speed_target = 35.0
        self.speed_coef = 0.05
        self.prev_ell = 0.0

        self.total_it = 0

    # ---------- helpers ----------

    def _obs_to_state(self, obs):
        """Normalize low-dim state branch from env: obs['state'] -> [-5,5]."""
        state = np.asarray(obs["state"], dtype=np.float32)
        state = np.clip(state, self.state_low, self.state_high)
        self.state_rms.update(state)
        normed = self.state_rms.normalize(state)
        return np.clip(normed, -5.0, 5.0)

    def _obs_to_vec(self, obs):
        """
        Build model input: [ normalized_state , flattened_image(CHW) ]
        """
        # ----- State branch -----
        normed_state = self._obs_to_state(obs)

        # ----- Image branch -----
        img = np.asarray(obs["image"], dtype=np.float32)  # (H,W,C)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C,H,W)
        img_flat = img.reshape(-1)  # length = flat_img_dim

        vec = np.concatenate([normed_state, img_flat], axis=0).astype(np.float32)
        return vec

    def _shape_reward(self, obs, next_obs, env_reward: float, info: dict | None):
        """Progress + centering + speed shaping, off-track penalty, optional pit bonus."""
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
        """[-1,1] -> env Box low/high."""
        return self.action_low + (squashed_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    # ---------- public API ----------

    def select_action(self, obs_vec: np.ndarray, deterministic: bool = False, noise_scale: float | None = None):
        """Return env action as np.float32 (optionally with exploration noise)."""
        noise_scale = self.cfg.exploration_noise if noise_scale is None else noise_scale
        s_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s_t).squeeze(0)
        a = self._rescale_action(a)

        if not deterministic and noise_scale > 0.0:
            a = a + noise_scale * torch.randn_like(a)
            a = torch.max(torch.min(a, self.action_high), self.action_low)

        return a.detach().cpu().numpy().astype(np.float32)

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)
        print(f"[TD3] Saved actor to {path}")

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)
        print(f"[TD3] Loaded actor from {path}")

    # ---------- TD3 update ----------

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return

        self.total_it += 1
        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size, self.device)

        with torch.no_grad():
            # Target policy smoothing
            next_actions = self.actor_target(next_states)             # [-1,1]
            next_actions = self._rescale_action(next_actions)         # env bounds
            noise = torch.randn_like(next_actions) * self.cfg.policy_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = torch.clamp(next_actions + noise, self.action_low, self.action_high)

            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * q_next

        # Critic losses
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # Delayed policy + target updates
        if self.total_it % self.cfg.policy_delay == 0:
            actor_actions = self.actor(states)                     # [-1,1]
            actor_actions_env = self._rescale_action(actor_actions) # env bounds
            actor_loss = -self.q1(states, actor_actions_env).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            soft_update(self.actor_target, self.actor, self.cfg.tau)
            soft_update(self.q1_target, self.q1, self.cfg.tau)
            soft_update(self.q2_target, self.q2, self.cfg.tau)

    # ---------- training loop ----------

    def train(self):
        cfg = self.cfg
        env = self.env

        episode_returns = []
        episode_env_returns = []

        full_obs, _ = env.reset(seed=cfg.seed)
        obs_vec = self._obs_to_vec(full_obs)
        self.prev_ell = float(full_obs["state"][4])
        ep_ret = 0.0
        ep_env_ret = 0.0
        ep_len = 0

        for t in range(cfg.total_steps):
            if cfg.render and env.render_mode == "human":
                env.render()

            if t < cfg.start_steps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = self.select_action(obs_vec, deterministic=False)

            next_full_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_vec = self._obs_to_vec(next_full_obs)
            shaped_reward = self._shape_reward(full_obs, next_full_obs, float(reward), info)

            done = bool(terminated or truncated)
            self.replay.store(obs_vec, action, shaped_reward, next_obs_vec, done)

            obs_vec = next_obs_vec
            full_obs = next_full_obs
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
                    f"[TD3] Ep {len(episode_returns):4d} | step={t+1:7d} | "
                    f"shaped_return={ep_ret:8.2f} | env_return={ep_env_ret:7.2f} | "
                    f"len={ep_len:4d}"
                )
                full_obs, _ = env.reset()
                obs_vec = self._obs_to_vec(full_obs)
                self.prev_ell = float(full_obs["state"][4])
                ep_ret = 0.0
                ep_env_ret = 0.0
                ep_len = 0

        if cfg.save_path:
            self.save(cfg.save_path)

        env.close()
        return episode_returns, episode_env_returns


# ---------------------------
# Env factory + CLI
# ---------------------------

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
    parser.add_argument("--save-path", type=str, default="td3_car_racing_actor.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = "human" if args.render else None
    env = make_env(render_mode=render_mode, seed=args.seed, max_episode_steps=args.max_episode_steps)

    cfg = TD3Config(
        total_steps=args.total_steps,
        max_ep_len=args.max_episode_steps,
        seed=args.seed,
        render=args.render,
        save_path=args.save_path,
    )

    agent = TD3CarRacingAgent(env, cfg)
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
        plt.title("TD3 on Custom CarRacing (CNN + State)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("td3_car_racing_returns_cnn.png", dpi=150)
        plt.close()
    else:
        print("No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()
