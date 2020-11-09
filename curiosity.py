import torch 
import torch.nn as nn

## Intrinsic Curiosity Module (ICM) from 
## Curiosity-driven Exploration by Self-supervised Prediction
## (Pathak et. al., https://arxiv.org/abs/1705.05363)
## Comments refer to the above-mentioned paper

class ICM(nn.Module):
    def __init__(self, state_size, action_size, feature_size, hidden_sizes):
        super(ICM, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size

        # The feature encoding, or \phi from 2.2
        # Encodes the state as a vector of features (state features)
        self.feature = nn.Sequential(
                nn.Linear(self.state_size, hidden_sizes['feature']),
                nn.ReLU(),
                nn.Linear(hidden_sizes['feature'], self.feature_size),
        )

        # The inverse model, or g from equation 2
        # Predicts action from current state features and next state features
        self.inverse_net = nn.Sequential(
                nn.Linear(self.feature_size, hidden_sizes['inverse']),
                nn.ReLU(),
                nn.Linear(hidden_sizes['inverse'], self.action_size)
        ) 

        # The forward model, or f from from equation 4 
        # Predicts next state features from actions and current state features
        self.forward_net = nn.Sequential(
                nn.Linear(
                    self.feature_size + self.action_size,
                    hidden_sizes['forward']),
                nn.ReLU(),
                nn.Linear(hidden_sizes['forward'], self.feature_size)
        )



