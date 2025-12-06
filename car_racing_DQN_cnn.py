import argparse
from datetime import datetime
import random
from collections import deque
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from envs.car_racing import CarRacing

def curated_action_set():
    """
    Discrete set of 13 hand-crafted actions over [steer, gas, brake, pit].
    """
    actions = [
        [-0.8, 0.8, 0.0, 0.0],  # hard left, throttle
        [0.0, 0.8, 0.0, 0.0],   # straight, throttle
        [0.8, 0.8, 0.0, 0.0],   # hard right, throttle
        [-0.5, 0.4, 0.0, 0.0],  # medium left, light gas
        [0.0, 0.4, 0.0, 0.0],   # coast with a little gas
        [0.5, 0.4, 0.0, 0.0],   # medium right, light gas
        [-0.8, 0.0, 0.0, 0.0],  # hard left coast
        [0.0, 0.0, 0.0, 0.0],   # full coast
        [0.8, 0.0, 0.0, 0.0],   # hard right coast
        [0.0, 0.5, 0.7, 0.0],   # straight brake
        [-0.6, 0.2, 0.8, 0.0],  # brake left
        [0.6, 0.2, 0.8, 0.0],   # brake right
        [0.0, 0.2, 0.0, 1.0],   # pit entry (slow, pit on)
    ]
    return np.asarray(actions, dtype=np.float32)


class ReplayBufferCNN:
    """
    Simple replay buffer that stores both state vector and raw image.
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, image, action_idx, reward, next_state, next_image,
             terminated, truncated):
        self.buffer.append(
            (state, image, action_idx, reward, next_state, next_image,
             terminated, truncated)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, images, action_idx, rewards, next_states, next_images, terminated, truncated = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(images, dtype=np.uint8),
            np.array(action_idx, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_images, dtype=np.uint8),
            np.array(terminated, dtype=np.float32),
            np.array(truncated, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# CNN + Dueling Q-network

class DuelingCnnQNetwork(nn.Module):
    """
    Dueling DQN that takes:
      - low-dimensional state vector (e.g., 8 or 10 dims)
      - 96x96x3 RGB image (CarRacing)
    and outputs Q-values for each discrete action.
    """
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()

        # CNN for the image (expects input shape [B, 3, 96, 96])
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),  # [B,16,23,23]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # [B,32,10,10]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # [B,64,8,8]
            nn.ReLU(),
        )

        # Compute conv output size by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).shape[1]

        # Small MLP for the state vector
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        fused_dim = conv_out_dim + 64

        # Shared feature layer after fusion
        self.feature = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
        )

        # Dueling heads
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim]
        image: [B, 3, 96, 96], values in [0,1]
        """
        img_feat = self.conv(image)
        img_feat = img_feat.reshape(img_feat.size(0), -1)

        state_feat = self.state_mlp(state)

        fused = torch.cat([img_feat, state_feat], dim=1)
        feat = self.feature(fused)

        adv = self.advantage(feat)
        val = self.value(feat)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = val + (adv - adv_mean)
        return q


# Agent 

class DQNCarRacingCNNAgent:
    def __init__(
        self,
        env: CarRacing,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        replay_capacity: int = 100000,
        min_replay_size: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        target_update_freq: int = 1000,
        reward_clip: float = 80.0,
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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Dimensions
        self.state_dim = env.observation_space["state"].shape[0]
        self.img_shape = env.observation_space["image"].shape  # (96,96,3)

        # Discrete action table (same as base DQN)
        self.actions_table = curated_action_set()
        self.num_actions = self.actions_table.shape[0]

        # Networks
        self.q_net = DuelingCnnQNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_q_net = DuelingCnnQNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay = ReplayBufferCNN(capacity=replay_capacity)

        # Epsilon-greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.global_step = 0
        self.gradient_steps = 0

        # Reward shaping knobs 
        self.progress_coef = 60.0
        self.center_penalty = 0.5
        self.offtrack_penalty = 20.0
        self.pit_bonus = 25.0
        self.speed_target = 35.0
        self.speed_coef = 0.05
        self.heading_penalty = 1.2
        self.lateral_penalty = 0.2
        self.lap_bonus = 500.0
        self.alive_bonus = 0.01
        self.prev_ell = 0.0

    # saving/loading

    def save(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name:  # only make dirs if there *is* a directory
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        print(f"[DQN-CNN] Saved Q-network to {path}")


    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()
        print(f"[DQN-CNN] Loaded Q-network from {path}")

    # observation helpers

    @staticmethod
    def _split_obs(obs):
        state = np.asarray(obs["state"], dtype=np.float32)
        image = np.asarray(obs["image"], dtype=np.uint8)
        return state, image

    def _update_epsilon(self):
        self.global_step += 1
        frac = min(1.0, self.global_step / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # action selection

    def select_action(self, state: np.ndarray, image: np.ndarray, greedy: bool = False):
        if (not greedy) and (random.random() < self.epsilon):
            action_idx = random.randrange(self.num_actions)
        else:
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, S]
            img_t = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,H,W,C]
            img_t = img_t.permute(0, 3, 1, 2) / 255.0  # [1,3,96,96]

            with torch.no_grad():
                q_values = self.q_net(s_t, img_t)  # [1, num_actions]
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return action_idx, self.actions_table[action_idx]

    # reward shaping

    def _shape_reward(self, obs, next_obs, env_reward: float, info: dict | None):
        if info is None:
            info = {}

        s_next = np.asarray(next_obs["state"], dtype=np.float32)

        # [d_t, v_t, infield, pitroad, ell_t, w_t, f_t, kappa_t, heading_err, lat_v]
        d_t = float(s_next[0])          # lateral offset
        v_t = float(s_next[1])          # speed
        off_track = bool(s_next[2] > 0.5)
        ell = float(s_next[4])          # progress 0..1
        heading_err = float(s_next[8])  # heading error
        lat_v = float(s_next[9])        # lateral velocity

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
        shaped = np.clip(shaped, -self.reward_clip, self.reward_clip)
        self.prev_ell = ell
        return float(shaped)

    # ---------- training step ----------

    def train_step(self):
        if len(self.replay) < self.min_replay_size:
            return

        (states, images, action_idx, rewards,
         next_states, next_images, terminated, truncated) = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        imgs_t = torch.tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0

        action_idx_t = torch.tensor(action_idx, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)

        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        next_imgs_t = torch.tensor(next_images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0

        terminated_t = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(-1)
        truncated_t = torch.tensor(truncated, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Q(s,a)
        q_all = self.q_net(states_t, imgs_t)
        q_sa = q_all.gather(1, action_idx_t.unsqueeze(-1))

        # Target Q(s', a')
        with torch.no_grad():
            q_next_online = self.q_net(next_states_t, next_imgs_t)
            next_action = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target = self.target_q_net(next_states_t, next_imgs_t).gather(1, next_action)

            done_mask = 1.0 - torch.clamp(terminated_t + truncated_t, 0.0, 1.0)
            target = rewards_t + self.gamma * done_mask * q_next_target

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    # main training loop

    def train(self, total_timesteps: int, checkpoint_dir: str | None = None,
              save_every_episodes: int | None = None):

        obs, _ = self.env.reset()
        state, image = self._split_obs(obs)
        self.prev_ell = float(obs["state"][4])

        episode_reward = 0.0      # shaped return
        episode_env_reward = 0.0  # raw env return
        episode = 0
        episode_returns = []

        for t in range(total_timesteps):
            self._update_epsilon()

            action_idx, action = self.select_action(state, image)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_state, next_image = self._split_obs(next_obs)
            done = bool(terminated or truncated)

            shaped_reward = self._shape_reward(obs, next_obs, float(reward), info)

            self.replay.push(
                state, image, action_idx, shaped_reward,
                next_state, next_image,
                float(terminated), float(truncated)
            )
            self.train_step()

            state, image = next_state, next_image
            obs = next_obs
            episode_reward += shaped_reward
            episode_env_reward += float(reward)

            if done:
                episode += 1
                episode_returns.append(episode_env_reward)
                print(
                    f"[DQN-CNN] Ep {episode} | step={t+1} "
                    f"| env_return={episode_env_reward:.2f} "
                    f"| eps={self.epsilon:.3f} "
                    f"| terminated={terminated} truncated={truncated}"
                )

                obs, _ = self.env.reset()
                state, image = self._split_obs(obs)
                self.prev_ell = float(obs["state"][4])
                episode_reward = 0.0
                episode_env_reward = 0.0

                # Periodic checkpoint saving
                if checkpoint_dir is not None and save_every_episodes is not None:
                    if episode % save_every_episodes == 0:
                        ckpt_path = os.path.join(
                            checkpoint_dir,
                            f"dqn_carracing_cnn_ep{episode}.pth",
                        )
                        self.save(ckpt_path)

        return episode_returns


# Env helper & main

def make_env(render_mode=None, seed: int = 0, max_episode_steps: int = 3000) -> CarRacing:
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
    parser.add_argument("--total-timesteps", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_dqn_cnn")
    parser.add_argument("--save-every-episodes", type=int, default=50)
    parser.add_argument("--save-final", type=str, default="dqn_car_racing_cnn_final.pth")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = "human" if args.render else None

    env = make_env(
        render_mode=render_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNCarRacingCNNAgent(env, device=device)

    episode_returns = agent.train(
        total_timesteps=args.total_timesteps,
        checkpoint_dir=run_checkpoint_dir,
        save_every_episodes=args.save_every_episodes,
    )
    env.close()

    if len(episode_returns) > 0:
        final_path = args.save_final
        agent.save(final_path)

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
        plt.title("DQN-CNN on Custom CarRacing")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dqn_car_racing_cnn_returns.png", dpi=150)
        plt.close()
    else:
        print("[DQN-CNN] No completed episodes -> nothing to plot.")


if __name__ == "__main__":
    main()
