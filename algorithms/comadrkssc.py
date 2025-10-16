import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from algorithms.icm import ICM 

# Actor and CentralizedCritic classes are unchanged
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))
    def get_action(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() > 1: obs = obs.flatten(start_dim=-2)
        logits = self.net(obs); dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)
    def log_prob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() > 1: obs = obs.flatten(start_dim=-2)
        logits = self.net(obs); dist = Categorical(logits=logits)
        return dist.log_prob(actions)

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, state):
        if not isinstance(state, torch.Tensor): state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1: state = state.unsqueeze(0)
        return self.net(state).squeeze(-1)


class CoMADRLKSSC:
    def __init__(self, cfg):
        self.cfg = cfg
        obs_dim, act_dim, state_dim = cfg.get('obs_dim'), cfg.get('act_dim'), cfg.get('state_dim')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = CentralizedCritic(state_dim).to(self.device)
        self.icm = ICM(obs_dim, act_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr', 3e-4))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.get('critic_lr', 1e-3))
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=cfg.get('icm_lr', 3e-4))
        self.log_lambda_c = torch.nn.Parameter(torch.tensor(np.log(cfg.get('lambda_c', 1.0)), dtype=torch.float32, device=self.device))
        self.lambda_c_opt = torch.optim.Adam([self.log_lambda_c], lr=cfg.get('lambda_c_lr', 1e-3))
        self.gamma, self.lam, self.clip = cfg.get('gamma', 0.99), cfg.get('lam', 0.95), cfg.get('clip', 0.2)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.max_grad_norm = cfg.get('max_grad_norm', 0.5)
        self.cost_threshold = cfg.get('cost_threshold', 0.05)
        self.icm_beta = cfg.get('icm_beta', 0.2)
        self.intrinsic_reward_weight = cfg.get('intrinsic_reward_weight', 0.01)

    def update(self, batch):
        # --- FIX IS HERE ---
        obs, actions, rewards, costs, dones, old_logp, states, next_states, next_obs = (
            batch[k].to(self.device) for k in ['obs', 'actions', 'rewards', 'costs', 'dones', 'logp', 'states', 'next_states', 'next_obs']
        )
        
        flat_obs = obs.flatten(start_dim=-2)
        next_flat_obs = next_obs.flatten(start_dim=-2) # Use next_obs, not next_states

        intrinsic_reward, forward_loss, inverse_loss = self.icm(flat_obs, next_flat_obs, actions)
        icm_loss = ((1 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss).mean()
        self.icm_opt.zero_grad(); icm_loss.backward(); self.icm_opt.step()
        
        total_rewards = rewards + self.intrinsic_reward_weight * intrinsic_reward.detach()

        with torch.no_grad():
            values_old = self.critic(states)
            next_values = self.critic(next_states)
            deltas = total_rewards + self.gamma * next_values * (1 - dones) - values_old
            advs = torch.zeros_like(total_rewards)
            gae = 0
            for t in reversed(range(len(total_rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                advs[t] = gae
            returns = advs + values_old

        critic_loss = F.mse_loss(self.critic(states), returns)
        self.critic_opt.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.critic_opt.step()

        lambda_c = torch.exp(self.log_lambda_c).detach()
        safe_advantage = advs.detach() - lambda_c * costs
        safe_advantage = (safe_advantage - safe_advantage.mean()) / (safe_advantage.std() + 1e-8)

        new_logp = self.actor.log_prob(obs, actions)
        ratio = torch.exp(new_logp - old_logp)
        surr1, surr2 = ratio * safe_advantage, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * safe_advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        _, logp_for_entropy = self.actor.get_action(obs)
        entropy = -logp_for_entropy.mean()
        actor_loss = policy_loss - self.entropy_coef * entropy
        self.actor_opt.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.actor_opt.step()

        avg_cost_batch = costs.mean()
        lagrangian_loss = -(self.log_lambda_c * (avg_cost_batch - self.cost_threshold).detach())
        self.lambda_c_opt.zero_grad(); lagrangian_loss.backward(); self.lambda_c_opt.step()
            
        return {'critic_loss': critic_loss.item(), 'cost_loss': 0, 'policy_loss': policy_loss.item(), 'value_loss': critic_loss.item(), 'entropy': entropy.item(), 'lambda_c': lambda_c.item(), 'avg_cost': avg_cost_batch.item()}