import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor


@NETS.register_module()
class EIIETrans(Net):
    def __init__(self,
                 d_model,
                 nhead,
                 batch_first=True,
                 num_layers=3,
                 time_steps=10,
                 n_tics=29
                 ):
        super(EIIETrans, self).__init__()
        self.d_model = d_model
        self.n_tics = n_tics
        self.time_steps = time_steps

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=batch_first, norm_first=True  # , dropout=0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(4 * d_model, n_tics)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_tics * time_steps, d_model)
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):  # (batch_size, num_seqs, action_dim, time_steps, state_dim)
        if len(x.shape) > 4:
            x = x.squeeze(1)

        x = x.transpose(1, 2)
        x_size = x.shape

        pos_embedding = self.pos_embedding.expand(x_size[0], self.n_tics * self.time_steps, self.d_model)
        cls_token = self.cls_token.expand(x_size[0], 1, self.d_model)

        x = x.reshape(x_size[0], x_size[1] * x_size[2], x_size[3])

        x = x + pos_embedding
        x = torch.cat((cls_token, x), 1)

        x = self.encoder(x)
        x = x[:, 0]
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        para = self.para.repeat(x.shape[0], 1)
        x = torch.cat((x, para), dim=1)

        x = torch.softmax(x, dim=1)
        return x


@NETS.register_module()
class EIIEConv(Net):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 time_steps=10,
                 kernel_size=3,
                 dims=(32,)):
        super(EIIEConv, self).__init__()

        self.kernel_size = kernel_size
        self.time_steps = time_steps

        self.net = build_conv2d(
            dims=[input_dim, *dims, output_dim],
            kernel_size=[(1, self.kernel_size), (1, self.time_steps - self.kernel_size + 1)]
        )
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):  # (batch_size, num_seqs, action_dim, time_steps, state_dim)
        if len(x.shape) > 4:
            x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.view(x.shape[0], -1)

        para = self.para.repeat(x.shape[0], 1)
        x = torch.cat((x, para), dim=1)
        x = torch.softmax(x, dim=1)
        return x


@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 input_dim,
                 action_dim,
                 output_dim=1,
                 time_steps=10,
                 num_layers=1,
                 hidden_size=32,
                 ):
        super(EIIECritic, self).__init__()

        self.time_steps = time_steps

        self.lstm = nn.LSTM(input_size=input_dim * time_steps,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        # print("lstm", x.shape, a.shape)
        if len(x.shape) >= 4:
            x = x.view(x.shape[0], x.shape[1], -1)

        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out)
        x = self.act(x)

        x = x.view(x.shape[0], -1)
        para = self.para.repeat(x.shape[0], 1)

        x = torch.cat((x, para, a), dim=1)
        x = self.linear2(x)
        x = x.mean(dim=1, keepdim=True)

        return x


@NETS.register_module()
class EIIETransCritic(Net):
    def __init__(self,
                 d_model,
                 nhead,
                 batch_first=True,
                 num_layers=3,
                 time_steps=10,
                 n_tics=29
                 ):
        super(EIIETransCritic, self).__init__()
        self.d_model = d_model
        self.n_tics = n_tics
        self.time_steps = time_steps

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=batch_first, norm_first=True  # , dropout=0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(4 * d_model, n_tics)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_tics * time_steps, d_model)
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())
        self.linear3 = nn.Linear(2 * (n_tics + 1), d_model)  # d_model
        self.act2 = nn.Tanh()
        self.linear4 = nn.Linear(d_model, 1)  # d_model

    def forward(self, x, a):  # (batch_size, num_seqs, action_dim, time_steps, state_dim)
        if len(x.shape) > 4:
            x = x.squeeze(1)

        x = x.transpose(1, 2)
        x_size = x.shape

        pos_embedding = self.pos_embedding.expand(x_size[0], self.n_tics * self.time_steps, self.d_model)
        cls_token = self.cls_token.expand(x_size[0], 1, self.d_model)

        x = x.reshape(x_size[0], x_size[1] * x_size[2], x_size[3])

        x = x + pos_embedding
        x = torch.cat((cls_token, x), 1)

        x = self.encoder(x)
        x = x[:, 0]
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        para = self.para.repeat(x.shape[0], 1)
        x = torch.cat((x, para, a), dim=1)

        x = self.linear3(x)
        x = self.act2(x)

        x = self.linear4(x)
        return x

@NETS.register_module()
class EIIEConvCritic(Net):
    def __init__(self,
                 input_dim,
                 action_dim,
                 output_dim=1,
                 time_steps=10,
                 kernel_size=3,
                 dims=(32,)):
        super(EIIEConvCritic, self).__init__()

        self.kernel_size = kernel_size
        self.time_steps = time_steps

        self.net = build_conv2d(
            dims=[input_dim, *dims, output_dim],
            kernel_size=[(1, self.kernel_size), (1, self.time_steps - self.kernel_size + 1)]
        )
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())
        self.linear = nn.Linear(2 * (action_dim + 1), 1)

    def forward(self, x, a):  # (batch_size, num_seqs, action_dim, time_steps, state_dim)
        if len(x.shape) > 4:
            x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.view(x.shape[0], -1)

        para = self.para.repeat(x.shape[0], 1)
        x = torch.cat((x, para, a), dim=1)

        x = self.linear(x)

        return x
