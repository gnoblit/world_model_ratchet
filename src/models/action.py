import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActionAgent(nn.Module):
    """
    The "Student" or policy network.
    It takes a state representation and outputs a policy (a distribution over actions).
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        """
        Initializes the ActionAgent.

        Args:
            state_dim (int): The dimensionality of the input state representation, `z_t`.
                                          Must match the output dim of the PerceptionAgent.
            num_actions (int): The number of discrete actions available in the environment.
            hidden_dim (int): The size of the hidden layers in the MLP. 
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # CrossEntropyLoss will take in raw logits, so no SoftMax
            nn.Linear(hidden_dim, num_actions)
        )
        self.num_actions = num_actions

    def forward(self, state_representation: torch.Tensor):
        """
        Processes a batch of state representations to produce action logits.
        
        Args:
            state_representation (torch.Tensor): A batch of state vectors, `z_t`.
                                                                                   Shape: (batch_size, hidden_dim).
                                                                                   
        Returns:
            torch.Tensor: The logits for each action.
                                      Shape: (batch_size, num_actions).
        """

        action_logits = self.network(state_representation)
        return action_logits
    
    def get_action(self, state_representation: torch.Tensor, deterministic: bool = False):
        """
        A helper method to select an action from the policy.
        
        Args:
            state_representation (torch.Tensor): A single state vector `z_t`
                                                                                    Shape: (1, state_dim).
            deterministic (bool): If True, take the action with the highest probability.
                                                    If False, sample from the distribution.
                                                    
        Returns:
            A tuple (action, log_prob):
                - action (int): The chosen action.
                - log_prob (torch.Tensor): The log probability of the chosen action.
        """
        # Get logits from the network
        logits = self.forward(state_representation)

        # Create categorical distribution over logits
        dist = Categorical(logits=logits)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            # Sample from distribution
            action = dist.sample()

        # Get the log probability of the sampled action
        log_prob = dist.log_prob(action)

        # Return the action as a Python number, log_prob as a tensor
        return action.item(), log_prob