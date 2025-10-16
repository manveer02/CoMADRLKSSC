import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Actor Network (Unchanged) ---
# The individual agent's brain. It decides what to do based on its local view.
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.log_std = nn.Linear(hidden, act_dim), nn.Parameter(torch.zeros(act_dim))
    def forward(self, x):
        h = self.net(x); mu = self.mu(h); std = torch.exp(self.log_std); return mu, std
    def get_action(self, obs):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        mu, std = self.forward(obs); dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample(); logp_raw = dist.log_prob(raw).sum(dim=-1); action = torch.tanh(raw)
        return action, logp_raw
    def log_prob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor): actions = torch.tensor(actions, dtype=torch.float32)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        if actions.dim() == 1: actions = actions.unsqueeze(0)
        a = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6); raw = 0.5 * torch.log((1 + a) / (1 - a))
        mu, std = self.forward(obs); dist = torch.distributions.Normal(mu, std)
        logp_raw = dist.log_prob(raw).sum(dim=-1); log_det = torch.sum(torch.log(1 - a**2 + 1e-8), dim=-1)
        return logp_raw - log_det

# --- STATE-OF-THE-ART CHANGE: Centralized Critic ---
# This network sees the full state of the environment (all agents, all landmarks)
# to make a stable and accurate value prediction.
class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state):
        if not isinstance(state, torch.Tensor): 
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1: 
            state = state.unsqueeze(0)
        return self.net(state).squeeze(-1)

class CoMADRLKSSC:
    def __init__(self, cfg):
        self.cfg = cfg
        obs_dim = cfg.get('obs_dim')
        act_dim = cfg.get('act_dim')
        state_dim = cfg.get('state_dim') # Global state dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor is decentralized (one per agent, but we use one network for all)
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        
        # Value function (Critic) is CENTRALIZED
        self.critic = CentralizedCritic(state_dim).to(self.device)
        
        # Optimizers with robust learning rates
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr', 1e-4))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.get('critic_lr', 3e-4))

        # Lagrangian multiplier for cost constraints
        self.log_lambda_c = torch.nn.Parameter(torch.tensor(np.log(cfg.get('lambda_c', 1.0)), dtype=torch.float32, device=self.device))
        self.lambda_c_opt = torch.optim.Adam([self.log_lambda_c], lr=cfg.get('lambda_c_lr', 1e-3))
        
        # Hyperparameters
        self.gamma = cfg.get('gamma', 0.99)
        self.lam = cfg.get('lam', 0.95) # GAE lambda
        self.clip = cfg.get('clip', 0.2) # PPO clip
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.max_grad_norm = cfg.get('max_grad_norm', 1.0)
        self.cost_threshold = cfg.get('cost_threshold', 0.1)

    def update(self, batch):
        # Unpack batch and move to device
        obs, actions, rewards, costs, dones, old_logp, states, next_states = (
            batch[k].to(self.device) for k in ['obs', 'actions', 'rewards', 'costs', 'dones', 'logp', 'states', 'next_states']
        )

        # --- CENTRALIZED VALUE FUNCTION UPDATE ---
        with torch.no_grad():
            # Get value predictions from the centralized critic using the GLOBAL state
            values_old = self.critic(states)
            next_values = self.critic(next_states)
            
            # Use Generalized Advantage Estimation (GAE) for a stable advantage calculation
            deltas = rewards + self.gamma * next_values * (1 - dones) - values_old
            advs = torch.zeros_like(rewards)
            gae = 0
            # This loop calculates how much better than average an action was
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                advs[t] = gae
            returns = advs + values_old

        # Update the critic by minimizing the difference between its predictions and the actual returns
        critic_loss = F.mse_loss(self.critic(states), returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # --- ACTOR (POLICY) UPDATE ---
        lambda_c = torch.exp(self.log_lambda_c).detach()
        
        # Incorporate the cost into the advantage
        # This is the "safe" part of your algorithm's name
        safe_advantage = advs - lambda_c * costs
        
        # Normalize advantage for stability
        safe_advantage = (safe_advantage - safe_advantage.mean()) / (safe_advantage.std() + 1e-8)

        # PPO Clipping objective
        new_logp = self.actor.log_prob(obs, actions)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * safe_advantage
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * safe_advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus for exploration
        mu, std = self.actor.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        actor_loss = policy_loss - self.entropy_coef * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # --- LAGRANGIAN MULTIPLIER UPDATE ---
        avg_cost_batch = costs.mean()
        lagrangian_loss = - (self.log_lambda_c * (avg_cost_batch - self.cost_threshold).detach())
        self.lambda_c_opt.zero_grad()
        lagrangian_loss.backward()
        self.lambda_c_opt.step()
            
        return {
            'critic_loss': critic_loss.item(),
            'cost_loss': 0, # Cost is now implicitly handled in the policy loss
            'policy_loss': policy_loss.item(),
            'value_loss': critic_loss.item(), # In this formulation, critic_loss is the value_loss
            'entropy': entropy.item(),
            'lambda_c': lambda_c.item(),
            'avg_cost': avg_cost_batch.item()
        }