import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM).
    This module creates an intrinsic reward signal to encourage exploration.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

        # --- FIX IS HERE ---
        # The forward model's input is the encoded state (hidden_dim) plus the
        # one-hot encoded action (act_dim), not hidden_dim + 1.
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs, next_obs, action):
        """
        Calculates the intrinsic reward and the loss for training the module.
        """
        phi_obs = self.encoder(obs)
        with torch.no_grad():
            phi_next_obs = self.encoder(next_obs)

        # --- Inverse Model ---
        predicted_action_logits = self.inverse_model(torch.cat((phi_obs, phi_next_obs), dim=1))
        inverse_loss = F.cross_entropy(predicted_action_logits, action.long(), reduction='none')

        # --- Forward Model ---
        action_one_hot = F.one_hot(action.long(), num_classes=self.inverse_model[-1].out_features).float()
        predicted_phi_next_obs = self.forward_model(torch.cat((phi_obs, action_one_hot), dim=1))
        
        intrinsic_reward = 0.5 * (predicted_phi_next_obs.detach() - phi_next_obs.detach()).pow(2).mean(dim=-1)
        forward_loss = 0.5 * (predicted_phi_next_obs - phi_next_obs).pow(2).mean(dim=-1)

        return intrinsic_reward, forward_loss, inverse_loss