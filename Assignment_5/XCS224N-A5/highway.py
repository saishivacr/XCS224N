#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Part 1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Highway(nn.Module):
    def __init__(self, word_embed_size: int, dropout_rate: float = 0.3):
        """
        Class that implements a highway network
        
        @param word_embed_size: number of features of the input
            tensor (the output of the convolutional network)
        @param dropout_rate: percentage of dropout to use
            on the output of the highway

        self.projection_layer: nn.Linear skip-connection
            layer with bias.
        self.gate_layer: nn.Linear layer with bias.
        self.dropout_layer: nn.Dropout layer applied to the
            word embeddings.
        """
        super(Highway, self).__init__()
        self.dropout_rate = dropout_rate
        self.projection_layer = nn.Linear(in_features=word_embed_size,
                                          out_features=word_embed_size,
                                          bias=True)
        self.gate_layer = nn.Linear(in_features=word_embed_size,
                                    out_features=word_embed_size,
                                    bias=True)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        Takes in batches of words and maps from x_{conv_out}
        to x_{highway}, using a projection and a skip-connection
        controlled by a dynamic gate, combining these outputs,
        and finally applying dropout to the resulting vector.

        @param x_conv: output of the convolutional network,
            a vector with shape (batch_size, word_embed_size)
        
        @returns x_highway: embeddings that represent
            words in x, replacing lookup-based word
            embeddings

            x_highway has shape(batch_size, word_embed_size)      

        """

        x_proj = F.relu(self.projection_layer(x_conv))
        x_gate = torch.sigmoid(self.gate_layer(x_conv))

        # Hadamard product with overloaded `*`
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv
        
        return self.dropout_layer(x_highway)
