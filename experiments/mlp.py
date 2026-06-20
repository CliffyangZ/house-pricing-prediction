import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        act_map = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_fn = act_map[activation]

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
