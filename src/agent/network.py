import torch
from torch import nn


class QNetwork(nn.Module):
    """
    Neural network used to approximate Q(s, a).

    Input:
        environment observation

    Output:
        one Q-value for every possible action
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_size,
                hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                action_size,
            ),
        )


    def forward(self, state):
        return self.network(state)

