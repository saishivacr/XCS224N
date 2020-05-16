#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import max_pool1d
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CNN(nn.Module):
    """
    A 1-dimensional convolutional layer. Convolves over the last
      dimension of the tensor (for embeddings, reshape the input
      so that the max_word_length is the last dimension).

    Hyperparameters:
            in_channels (int): Equal to the size of character embeddings  
            out_channels (int): Equal to the size of the final word embedding  
            kernel_size (int or tuple): Default 5  
            padding (int or tuple, optional): None
            stride (int): 1
            dilation (int): 1
            groups (int): 1
            bias (bool, optional): `True` 

    word_embed_size = char_embed_size x max_word_length

    """
    def __init__(self,
                 char_embed_size: int,
                 embed_size: int,
                 max_word_length: int,
                 kernel_size: int = 5):
        
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=char_embed_size,
                                    out_channels=embed_size,
                                    kernel_size=kernel_size)
        
        self.max_pool = nn.MaxPool1d(
            kernel_size=max_word_length - kernel_size + 1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Perform 1-D convolutions over characters to
        compute output features via max-pooling the
        outputs of the ReLU function applied to the
        sums of element-wise matrix multiplications
        between input windows and their corresponding
        weights.

        Hyperparameters:
            W: Weight matrix (word_embed_size, char_embed_size, kernel_size)
            b: bias vector (word_embed_size)
            
        @param x_reshaped: padded torch.Tensor
            with shape (sent_len*batch_size, char_embed_size, max_word_length)
        
        @return x_conv_out: torch.Tensor with shape (sent_len, batch_size, word_embed_size)
        """
        x_conv = self.conv_layer(x_reshaped)
        
        # Max pool
        # Method 1:
        # x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
        # return x_conv_out

        # Method 2:
        x_conv_out = self.max_pool(F.relu(x_conv))
        return torch.squeeze(x_conv_out, dim=2)
