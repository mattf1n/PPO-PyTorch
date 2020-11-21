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

        # # The feature encoding, or \phi from 2.2
        # # Encodes the state as a vector of features (state features)
        # self.feature = nn.Sequential(
        #         nn.Linear(self.state_size, hidden_sizes['feature']),
        #         nn.ReLU(),
        #         nn.Linear(hidden_sizes['feature'], self.feature_size),
        # )

        # # The inverse model, or g from equation 2
        # # Predicts action from current state features and next state features
        # self.inverse_net = nn.Sequential(
        #         nn.Linear(self.feature_size, hidden_sizes['inverse']),
        #         nn.ReLU(),
        #         nn.Linear(hidden_sizes['inverse'], self.action_size)
        # ) 

        # # The forward model, or f from from equation 4 
        # # Predicts next state features from actions and current state features
        # self.forward_net = nn.Sequential(
        #         nn.Linear(
        #             self.feature_size + self.action_size,
        #             hidden_sizes['forward']),
        #         nn.ReLU(),
        #         nn.Linear(hidden_sizes['forward'], self.feature_size)
        # )


        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_output * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(output_size + feature_output, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_output)
        )

        def forward(self, inputs):
            state, next_state, action = inputs

            encode_state = self.feature(state)
            # get pred action
            pred_action = torch.cat((encode_state, self.feature(next_state)), 1)
            pred_action = self.inverse_net(pred_action)
            # ---------------------

            # get pred next state
            pred_next_state_feature = torch.cat((encode_state, action), 1)
            pred_next_state_feature = self.forward_net(pred_next_state_feature)

            real_next_state_feature = self.feature(next_state)
            return real_next_state_feature, pred_next_state_feature, pred_action

