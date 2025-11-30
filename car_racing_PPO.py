# https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
# https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode
# https://github.com/openai/spinningup/blob/master/docs/algorithms/ppo.rst
import argparse
import itertools
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from envs.car_racing import CarRacing

class RunningMeanStd:
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

    def normalize(self, x: np.ndarray, clip_range: float | None = None):
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        if clip_range is not None:
            normed = np.clip(normed, -clip_range, clip_range)
        return normed


def discount_cumsum(x, discount):
    # output[k] = x[k] + discount * x[k+1] + discount^2 * x[k+2] + ...
    result = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        result[t] = running
    return result


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size, "PPOBuffer overflow"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "Buffer must be full before get()"
        self.ptr = 0
        self.path_start_idx = 0

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
            val=self.val_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_low, action_high, log_std_init: float = -0.2):
        super().__init__()

        hidden = 256
        action_dim = len(action_low)

        self.pi_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        low_t = torch.as_tensor(action_low, dtype=torch.float32)
        high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", low_t)
        self.register_buffer("action_high", high_t)
        self.register_buffer("action_scale", (high_t - low_t) / 2.0)
        self.register_buffer("action_bias", (high_t + low_t) / 2.0)

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 1.5)
        return Normal(mu, std)

    def _value(self, obs):
        return self.v_net(obs).squeeze(-1)

    def _squash_action(self, raw_action, dist: Normal):
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias

        logp = dist.log_prob(raw_action) - torch.log(torch.clamp(1 - squashed.pow(2), min=1e-6))
        logp = logp.sum(-1)
        return action, logp

    def _unsquash_action(self, action):
        normed = (action - self.action_bias) / torch.clamp(self.action_scale, min=1e-6)
        normed = torch.clamp(normed, -0.999, 0.999)
        raw_action = torch.atanh(normed)
        return raw_action, normed

    def step(self, obs):
        with torch.no_grad():
            dist = self._distribution(obs.unsqueeze(0))
            raw_action = dist.rsample()
            action, logp = self._squash_action(raw_action, dist)
            value = self._value(obs.unsqueeze(0))
        return (
            action.squeeze(0).detach().cpu().tolist(),
            float(value.item()),
            float(logp.item()),
        )

    def evaluate(self, obs, act):
        raw_action, squashed = self._unsquash_action(act)
        dist = self._distribution(obs)
        logp = dist.log_prob(raw_action) - torch.log(torch.clamp(1 - squashed.pow(2), min=1e-6))
        logp = logp.sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self._value(obs)
        return logp, entropy, value


@dataclass
class PPOConfig:
    steps_per_epoch: int = 10000
    epochs: int = 200
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 2e-4
    vf_lr: float = 6e-4
    train_pi_iters: int = 20
    train_v_iters: int = 20
    minibatch_size: int = 256
    target_kl: float = 0.02
    max_ep_len: int = 5000
    entropy_coef: float = 0.03
    entropy_coef_end: float = 0.005
    entropy_anneal: bool = True
    reward_clip: float = 100.0
    normalize_reward: bool = False
    max_grad_norm: float = 0.5
    vf_clip_param: float = 0.2
    lr_anneal: bool = True


class PPOCarRacingAgent:
    def __init__(self, env: CarRacing, config: PPOConfig, device=None):
        self.env = env
        self.cfg = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space["state"].shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

        self.state_low = np.asarray(env.observation_space["state"].low, dtype=np.float32)
        self.state_high = np.asarray(env.observation_space["state"].high, dtype=np.float32)
        self.state_scale = np.maximum(self.state_high - self.state_low, 1e-3)
        self.state_rms = RunningMeanStd(self.state_low.shape)
        self.rew_rms = RunningMeanStd((1,))

        self.ac = ActorCritic(state_dim, action_low, action_high).to(self.device)

        pi_params = list(self.ac.pi_net.parameters()) + [self.ac.log_std]
        vf_params = list(self.ac.v_net.parameters())

        self.pi_optimizer = optim.Adam(pi_params, lr=self.cfg.pi_lr)
        self.vf_optimizer = optim.Adam(vf_params, lr=self.cfg.vf_lr)

        self.buf = PPOBuffer(
            obs_dim=state_dim,
            act_dim=action_dim,
            size=self.cfg.steps_per_epoch,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

        # Reward shaping
        self.progress_coef = 120.0
        self.center_penalty = 0.1
        self.offtrack_penalty = 8.0
        self.pit_bonus = 20.0
        self.speed_target = 45.0
        self.speed_coef = 0.15
        self.heading_penalty = 2.5
        self.lateral_penalty = 0.3
        self.lap_bonus = 50.0
        self.alive_bonus = 0.02
        self.prev_ell = 0.0
        self.current_entropy_coef = self.cfg.entropy_coef

    def _obs_to_state(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32)
        state = np.clip(state, self.state_low, self.state_high)
        self.state_rms.update(state)
        normed = self.state_rms.normalize(state, clip_range=5.0)
        return normed

    def _shape_reward(self, obs, next_obs, env_reward: float, info: dict | None):
        if info is None:
            info = {}

        d_t = float(next_obs["state"][0])
        v_t = float(next_obs["state"][1])
        off_track = bool(next_obs["state"][2] > 0.5)
        ell = float(next_obs["state"][4])
        heading_err = float(next_obs["state"][8])
        lat_v = float(next_obs["state"][9])

        delta_ell = max(0.0, ell - self.prev_ell)
        progress_r = self.progress_coef * delta_ell
        center_r = -self.center_penalty * abs(d_t)
        speed_r = self.speed_coef * min(1.0, v_t / self.speed_target)
        offtrack_r = -self.offtrack_penalty if off_track else 0.0
        pit_r = self.pit_bonus if info.get("pit_executed", False) else 0.0
        heading_r = -self.heading_penalty * abs(heading_err)
        lateral_r = -self.lateral_penalty * abs(lat_v)
        lap_r = self.lap_bonus if info.get("lap_finished", False) else 0.0

        shaped = (
            env_reward
            + progress_r
            + center_r
            + speed_r
            + offtrack_r
            + pit_r
            + heading_r
            + lateral_r
            + lap_r
            + self.alive_bonus
        )
        shaped = np.clip(shaped, -self.cfg.reward_clip, self.cfg.reward_clip)
        if self.cfg.normalize_reward:
            self.rew_rms.update(np.asarray([shaped], dtype=np.float32))
            scale = float(1.0 / (np.sqrt(self.rew_rms.var) + 1e-8))
            shaped = shaped * scale
            shaped = np.clip(shaped, -self.cfg.reward_clip, self.cfg.reward_clip)
        self.prev_ell = ell
        return float(shaped)

    def _set_lr(self, lr_mult: float):
        lr_mult = max(0.0, lr_mult)
        for pg in self.pi_optimizer.param_groups:
            pg["lr"] = self.cfg.pi_lr * lr_mult
        for pg in self.vf_optimizer.param_groups:
            pg["lr"] = self.cfg.vf_lr * lr_mult

    def update(self):
        data = self.buf.get()

        obs = data["obs"].to(self.device)
        act = data["act"].to(self.device)
        ret = data["ret"].to(self.device)
        adv = data["adv"].to(self.device)
        logp_old = data["logp"].to(self.device)
        val_old = data["val"].to(self.device)

        buffer_size = obs.shape[0]
        batch_size = min(self.cfg.minibatch_size, buffer_size)

        stop_pi = False
        for _ in range(self.cfg.train_pi_iters):
            idx = torch.randperm(buffer_size, device=self.device)
            for start in range(0, buffer_size, batch_size):
                mb_idx = idx[start : start + batch_size]
                logp, entropy, _ = self.ac.evaluate(obs[mb_idx], act[mb_idx])
                ratio = torch.exp(logp - logp_old[mb_idx])

                obj1 = ratio * adv[mb_idx]
                obj2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv[mb_idx]
                entropy_bonus = self.current_entropy_coef * entropy.mean()
                pi_loss = -(torch.min(obj1, obj2).mean() + entropy_bonus)

                approx_kl = (logp_old[mb_idx] - logp).mean().item()
                if approx_kl > 1.5 * self.cfg.target_kl:
                    stop_pi = True
                    break

                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.pi_optimizer.step()
            if stop_pi:
                break

        for _ in range(self.cfg.train_v_iters):
            idx = torch.randperm(buffer_size, device=self.device)
            for start in range(0, buffer_size, batch_size):
                mb_idx = idx[start : start + batch_size]
                value = self.ac._value(obs[mb_idx])
                if self.cfg.vf_clip_param > 0.0:
                    v_pred_clipped = val_old[mb_idx] + torch.clamp(
                        value - val_old[mb_idx],
                        -self.cfg.vf_clip_param,
                        self.cfg.vf_clip_param,
                    )
                    v_loss_unclipped = (value - ret[mb_idx]) ** 2
                    v_loss_clipped = (v_pred_clipped - ret[mb_idx]) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((value - ret[mb_idx]) ** 2).mean()

                self.vf_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.vf_optimizer.step()

    def train(self):
        cfg = self.cfg
        env = self.env

        episode_returns = []
        episode_env_returns = []

        full_obs, _ = env.reset()
        obs = self._obs_to_state(full_obs)
        self.prev_ell = float(full_obs["state"][4])
        ep_ret = 0.0
        ep_env_ret = 0.0
        ep_len = 0

        for epoch in range(cfg.epochs):
            if cfg.lr_anneal:
                lr_mult = 1.0 - (epoch / float(cfg.epochs))
                self._set_lr(lr_mult)
            if cfg.entropy_anneal:
                frac = 1.0 - (epoch / float(cfg.epochs))
                self.current_entropy_coef = cfg.entropy_coef * frac + cfg.entropy_coef_end * (1.0 - frac)

            for t in range(cfg.steps_per_epoch):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                act, val, logp = self.ac.step(obs_t)
                action_np = np.asarray(act, dtype=np.float32)

                next_full_obs, reward, terminated, truncated, info = env.step(action_np)
                next_obs = self._obs_to_state(next_full_obs)

                shaped_reward = self._shape_reward(full_obs, next_full_obs, float(reward), info)
                ep_ret += shaped_reward
                ep_env_ret += float(reward)
                ep_len += 1

                self.buf.store(obs, act, shaped_reward, val, logp)

                obs = next_obs
                full_obs = next_full_obs

                timeout = ep_len >= cfg.max_ep_len
                terminal = bool(terminated)
                epoch_ended = (t == cfg.steps_per_epoch - 1)

                if terminal or timeout or epoch_ended or bool(truncated):
                    if terminal:
                        last_val = 0.0
                    else:
                        with torch.no_grad():
                            obs_t_last = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                            last_val = float(self.ac._value(obs_t_last).item())

                    self.buf.finish_path(last_val)

                    if terminal or timeout or bool(truncated):
                        episode_returns.append(ep_env_ret)
                        episode_env_returns.append(ep_env_ret)
                        print(
                            f"[PPO] Ep {len(episode_returns)} | ep_len={ep_len} | shaped_return={ep_ret:.2f} "
                            f"| env_return={ep_env_ret:.2f} | terminated={terminated} truncated={truncated}"
                        )
                        full_obs, _ = env.reset()
                        obs = self._obs_to_state(full_obs)
                        self.prev_ell = float(full_obs["state"][4])
                        ep_ret = 0.0
                        ep_env_ret = 0.0
                        ep_len = 0

                    if epoch_ended:
                        break

            self.update() 

        return episode_returns


def make_env(render_mode=None, seed=0, max_episode_steps=5000):
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    render_mode = "human" if args.render else None

    env = make_env(
        render_mode=render_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )

    cfg = PPOConfig(
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        max_ep_len=args.max_episode_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOCarRacingAgent(env, cfg, device=device)

    episode_returns = agent.train()
    env.close()

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
        plt.title("PPO on Custom CarRacing")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("ppo_car_racing_returns.png", dpi=150)
        plt.close()
    else:
        print("No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()
