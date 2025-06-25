import torch
import torch.nn as nn
from torch.distributions import Categorical
class ActorCritic(nn.Module):
    """
    An Actor-Critic module that contains both the policy network (Actor)
    and the value network (Critic).
    """
    def __init__(self, state_dim: int, cfg):
        """
        Initializes the ActorCritic model.

        Args:
            state_dim (int): The dimensionality of the input state representation `z_t`.
            cfg (ActionConfig): The configuration object for the action agent.
        """
        super().__init__()
        
        num_actions = cfg.num_actions
        hidden_dim = cfg.hidden_dim

        # --- Actor Network ---
        # The policy network that outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # --- Critic Network ---
        # The value network that estimates the value of a state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Outputs a single scalar value for the state
            nn.Linear(hidden_dim, 1)
        )
        
        self.num_actions = num_actions

    def forward(self, state_representation: torch.Tensor):
        """
        A 'forward' pass that returns both actor logits and critic value.
        This is useful for the update step.

        Args:
            state_representation (torch.Tensor): A batch of state vectors `z_t`.
                                                 Shape: (batch_size, state_dim).

        Returns:
            A tuple (action_logits, state_value):
                - action_logits (torch.Tensor): (batch_size, num_actions)
                - state_value (torch.Tensor): (batch_size, 1)
        """
        action_logits = self.actor(state_representation)
        state_value = self.critic(state_representation)
        return action_logits, state_value

    def get_action(self, state_representation: torch.Tensor, deterministic: bool = False):
        """
        Selects an action from the policy (actor) network.
        
        Args:
            state_representation (torch.Tensor): A single state vector `z_t`.
                                                 Shape: (1, state_dim).
            deterministic (bool): If True, take the most likely action.
        
        Returns:
            A tuple (action, log_prob):
                - action (int): The chosen action.
                - log_prob (torch.Tensor): The log probability of the chosen action.
        """
        # We only need the actor part for action selection
        logits = self.actor(state_representation)
        dist = Categorical(logits=logits)
        
        action = dist.probs.argmax() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        if state_representation.shape[0] == 1:
            return action.item(), log_prob.squeeze()
        else:
            return action, log_prob
    
    def get_value(self, state_representation: torch.Tensor) -> torch.Tensor:
        """
        Estimates the value of a state using the critic network.

        Args:
            state_representation (torch.Tensor): A batch of state vectors `z_t`.

        Returns:
            torch.Tensor: The estimated value of each state. Shape: (batch_size, 1).
        """
        return self.critic(state_representation)