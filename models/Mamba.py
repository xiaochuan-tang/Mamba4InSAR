import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding, DataEmbedding_inverted_Mamba

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"

        self.embedding = DataEmbedding_inverted_Mamba(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out_period, bias=False)
        # self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

        # Decoder
        if self.task_name == 'long_term_forecast':
            # self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            self.projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model * 2),
                nn.GELU(),
                nn.Linear(configs.d_model * 2, configs.pred_len)
            )

    def forecast(self, x_enc, x_mark_enc):

        # print(x_enc.shape,x_mark_enc.shape) # torch.Size([32, 60, 1]) torch.Size([32, 60, 4])
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        _, _, N = x_enc.shape

        x = self.embedding(x_enc, x_mark_enc)
        # print(x.shape) #torch.Size([32, 60, 256])
        x = self.mamba(x)
        # print(x.shape) #torch.Size([32, 60, 256])
        # x_out = self.out_layer(x)

        x_out = self.projection(x).permute(0, 2, 1)[:, :, :N]

        x_out = x_out * std_enc + mean_enc
        # print(x_out.shape) # torch.Size([32, 60, 1])

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
