import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.car_racing import CarRacing

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
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

    def push(self, state, action, reward, next_state, done):
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

        states = torch.from_numpy(self.states[idxs]).to(device)
        actions = torch.from_numpy(self.actions[idxs]).to(device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(device)
        dones = torch.from_numpy(self.dones[idxs]).to(device)

        return states, actions, rewards, next_states, dones

# Networks
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Stochastic policy: outputs mean + log_std of a Gaussian,
    then uses tanh-squashing to keep actions in [-1, 1],
    and we rescale to env.action_space bounds outside.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        Returns:
          action: squashed + rescaled action in [-1, 1] (before env rescale)
          log_prob: log π(a|s)
          pre_tanh: the unsquashed Gaussian sample
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()               # reparameterization trick
        action = torch.tanh(z)
        # log π(a|s) with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z

    def sample_deterministic(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean)
        return action

# Soft update
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


# SAC Training Loop
def train_sac(
    num_episodes=500,
    max_steps=3000,
    gamma=0.99,
    batch_size=128,
    lr=3e-4,
    replay_size=200_000,
    start_steps=10_000,        # collect this many random steps before using policy
    updates_per_step=1,
    tau=0.005,                 # target smoothing coefficient
    alpha=0.2,                 # initial entropy temperature (if auto_alpha=False, this stays fixed)
    auto_alpha=True,
    seed=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Environment: continuous=True for SAC ---
    env = CarRacing(
        render_mode="human",
        continuous=True,
        lap_complete_percent=0.95,
        reward_shaping=True,
        max_episode_steps=max_steps,
    )

    # Use only the low-dimensional state branch
    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0]

    # Action bounds from env
    action_low = torch.tensor(env.action_space.low, device=device, dtype=torch.float32)
    action_high = torch.tensor(env.action_space.high, device=device, dtype=torch.float32)

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Networks
    policy_net = GaussianPolicy(state_dim, action_dim).to(device)
    q1_net = QNetwork(state_dim, action_dim).to(device)
    q2_net = QNetwork(state_dim, action_dim).to(device)
    q1_target = QNetwork(state_dim, action_dim).to(device)
    q2_target = QNetwork(state_dim, action_dim).to(device)

    q1_target.load_state_dict(q1_net.state_dict())
    q2_target.load_state_dict(q2_net.state_dict())

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    q1_optimizer = optim.Adam(q1_net.parameters(), lr=lr)
    q2_optimizer = optim.Adam(q2_net.parameters(), lr=lr)

    # Entropy temperature
    if auto_alpha:
        target_entropy = -float(action_dim)   # recommended default
        log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=lr)
    else:
        log_alpha = torch.tensor(np.log(alpha), device=device)
        alpha_optimizer = None
    alpha = log_alpha.exp().item()

    replay_buffer = ReplayBuffer(replay_size, state_dim, action_dim)

    total_steps = 0

    def select_action(state_np, eval_mode=False):
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        if (not eval_mode) and (total_steps < start_steps):
            # use random policy at the start
            action = env.action_space.sample()
            return action.astype(np.float32)

        with torch.no_grad():
            if eval_mode:
                a = policy_net.sample_deterministic(state_tensor)
            else:
                a, _, _ = policy_net.sample(state_tensor)

        # a is in [-1, 1], rescale to env bounds
        a = a.squeeze(0)
        scaled = action_low + (a + 1.0) * 0.5 * (action_high - action_low)
        return scaled.cpu().numpy().astype(np.float32)

    # Training
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        state = obs["state"]
        episode_return = 0.0
        episode_steps = 0

        for t in range(max_steps):
            env.render()
            total_steps += 1
            episode_steps += 1

            # --- Collect experience ---
            action = select_action(state, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs["state"]
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            # --- Update networks ---
            if len(replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(
                        batch_size, device
                    )
                    rewards = rewards.unsqueeze(1)
                    dones = dones.unsqueeze(1)

                    # 1. Compute target Q values
                    with torch.no_grad():
                        next_actions, next_log_probs, _ = policy_net.sample(next_states)
                        # Rescale actions to env bounds
                        na = action_low + (next_actions + 1.0) * 0.5 * (action_high - action_low)

                        q1_next = q1_target(next_states, na)
                        q2_next = q2_target(next_states, na)
                        q_next_min = torch.min(q1_next, q2_next)
                        alpha_val = log_alpha.exp() if auto_alpha else log_alpha.exp()
                        target_q = rewards + gamma * (1.0 - dones) * (
                            q_next_min - alpha_val * next_log_probs
                        )

                    # 2. Q1, Q2 losses
                    q1 = q1_net(states, actions)
                    q2 = q2_net(states, actions)
                    q1_loss = nn.MSELoss()(q1, target_q)
                    q2_loss = nn.MSELoss()(q2, target_q)

                    q1_optimizer.zero_grad()
                    q1_loss.backward()
                    q1_optimizer.step()

                    q2_optimizer.zero_grad()
                    q2_loss.backward()
                    q2_optimizer.step()

                    # 3. Policy loss
                    new_actions, log_probs, _ = policy_net.sample(states)
                    na2 = action_low + (new_actions + 1.0) * 0.5 * (action_high - action_low)

                    q1_pi = q1_net(states, na2)
                    q2_pi = q2_net(states, na2)
                    q_pi_min = torch.min(q1_pi, q2_pi)

                    alpha_val = log_alpha.exp() if auto_alpha else log_alpha.exp()
                    policy_loss = (alpha_val * log_probs - q_pi_min).mean()

                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    # 4. Temperature (alpha) loss
                    if auto_alpha:
                        alpha_loss = (
                            -log_alpha * (log_probs + target_entropy).detach()
                        ).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()

                    alpha = log_alpha.exp().item()

                    # 5. Soft update target networks
                    soft_update(q1_target, q1_net, tau)
                    soft_update(q2_target, q2_net, tau)

            if done:
                break

        print(
            f"Episode {episode:4d} | Return = {episode_return:7.2f} | "
            f"Steps = {episode_steps:4d} | TotalSteps = {total_steps:7d} | alpha={alpha:.3f}"
        )

    env.close()
    # Save models (policy is the main one you care about)
    torch.save(policy_net.state_dict(), "sac_car_racing_policy.pth")
    torch.save(q1_net.state_dict(), "sac_car_racing_q1.pth")
    torch.save(q2_net.state_dict(), "sac_car_racing_q2.pth")
    print("Training finished, models saved (policy + Q networks).")

if __name__ == "__main__":
    train_sac()
