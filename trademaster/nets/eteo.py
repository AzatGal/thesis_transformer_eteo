import math

import torch
import torch.nn as nn

from .builder import NETS
from .custom import Net


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        """
        print(x.shape)
        print(self.pe.shape)
        print(self.pe[:x.size(0)].shape)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def build_transformer(_num_layers, _d_model, _nhead, _dim_feedforward=2048, _dropout=0.1, _batch_first=False):
    _encoder = nn.TransformerEncoderLayer(
        d_model=_d_model,
        nhead=_nhead,
        dim_feedforward=_dim_feedforward,
        dropout=_dropout,
        batch_first=_batch_first
    )
    return nn.TransformerEncoder(_encoder, _num_layers)


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


@NETS.register_module()
class ETEOStacked(Net):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 batch_first=False,
                 time_steps=10
                 ):
        super(ETEOStacked, self).__init__()

        self.net = build_transformer(num_layers,
                                     d_model,
                                     nhead,
                                     dim_feedforward,
                                     dropout,
                                     batch_first
                                     )

        self.pos_encoding = PositionalEncoding(d_model, dropout, time_steps)

        self.act_linear_volume = nn.Linear(d_model * time_steps, 2)
        self.act_linear_price = nn.Linear(d_model * time_steps, 2)
        self.v_linear = nn.Linear(d_model * time_steps, 1)

        # init weights
        # self.net.apply(self.init_weights)
        self.act_linear_volume.apply(self.init_weights)
        self.act_linear_price.apply(self.init_weights)
        self.v_linear.apply(self.init_weights)

    def forward(self, x):
        B = x.shape
        x = x.view(B[-2], B[0], B[-1])

        x = self.pos_encoding(x)

        x = self.net(x)
        x = x.view(B[0], -1)

        action_volume = self.act_linear_volume(x)
        action_price = self.act_linear_price(x)
        v = self.v_linear(x)

        return action_volume, action_price, v
    def init_weights(self, m):
        # init linear
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.zero_()


"""
    def __init__(self, dims: [int], state_dim: int, action_dim: int, time_steps=10, explore_rate=0.25):
        # nodes is a list where the element reprensents the nodes on each layer
        super(ETEOStacked, self).__init__()

        self.net = build_mlp(dims=[state_dim * time_steps, *dims])

        self.act_linear_volume = nn.Linear(dims[-1], 2)
        self.act_linear_price = nn.Linear(dims[-1], 2)
        self.v_linear = nn.Linear(dims[-1], 1)

        # init weights
        self.net.apply(self.init_weights)
        self.act_linear_volume.apply(self.init_weights)
        self.act_linear_price.apply(self.init_weights)
        self.v_linear.apply(self.init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)

        x = self.net(x)

        action_volume = self.act_linear_volume(x)
        action_price = self.act_linear_price(x)
        v = self.v_linear(x)

        return action_volume, action_price, v
"""