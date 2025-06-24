import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """
    The JEPA style predictive world model.
    It takes the current state representation, `z-t`, and an action, `a_t`,
    and predicts the next state's representation, `z_hat_t+1'.
    """
    def __init__(self, state_dim: int, num_actions: int, cfg):
        """
        Initializes the WorldModel.

        Args:
            state_dim (int): The dimensionality of the state representation `z`.
            num_actions (int): The number of possible discrete actions.
            cfg (WorldModelConfig): The configuration object for the world model.
        """
        super().__init__()

        # Need to represent discrete action as a vector
        # Embedding layer is a good way to accomplish this
        action_embedding_dim = cfg.action_embedding_dim
        hidden_dim = cfg.hidden_dim

        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)

        # Input to network will be the concatenated state and action embedding
        input_dim = state_dim + action_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # The output layer must produce a vector of the same size as the state representation
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state_representation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next state representation.

        Args:
            state_representation (torch.Tensor): A batch of state vectors `z_t`.
                                                 Shape: (batch_size, state_dim).
            action (torch.Tensor): A batch of action indices.
                                   Shape: (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: The predicted next state representation `z_hat_t+1`.
                          Shape: (batch_size, state_dim).
        """
        # Ensure action is long type for embedding lookup
        action = action.long()
        
        # Get the learnable embedding for the action
        # Shape: (batch_size, action_embedding_dim)
        action_emb = self.action_embedding(action.squeeze(-1) if action.dim() > 1 else action)
        
        # Concatenate the state and action embedding to form the input
        # Shape: (batch_size, state_dim + action_embedding_dim)
        combined_input = torch.cat([state_representation, action_emb], dim=-1)
        
        # Pass through the network to get the prediction
        predicted_next_state = self.network(combined_input)
        
        return predicted_next_state

