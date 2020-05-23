import torch
import torch.nn as nn
import math

@torch.no_grad()
def init_weights(m):
    if type(m) in {
        nn.Linear,
        nn.LSTM,
        nn.LSTMCell
    }:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.xavier_normal_(m.bias)
    # elif type(m) in {
    #     nn.Conv1d
    # }:
    #     nn.init.orthogonal_(self.conv_layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
    else:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='leaky_relu')
        if m.state_dict().get('bias', None) is not None:
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)