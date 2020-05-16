#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from vocab import VocabEntry

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size: int, vocab: VocabEntry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hyperparameters:
          @param char_embed_size: dimensionality of character embeddings
          @param max_word_length: maximum length to which a word will be
            padded or truncated
          @param dropout_rate: probability to use in the dropout layer
        """
        super(ModelEmbeddings, self).__init__()

        # A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        # End A4 code

        self.embed_size = embed_size
        self.char_embed_size = 50
        self.max_word_length = 21
        self.dropout_rate = 0.3
        self.vocab = vocab

        self.char_embedding_layer = nn.Embedding(
            num_embeddings=len(self.vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=self.vocab.char2id['<pad>']
        )

        self.cnn_layer = CNN(
            char_embed_size=self.char_embed_size,
            embed_size=self.embed_size,
            max_word_length=self.max_word_length,
            kernel_size=5)

        self.highway_layer = Highway(
            word_embed_size=self.embed_size,
            dropout_rate=self.dropout_rate
        )

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # A4 code
        # output = self.embeddings(input)
        # return output
        # End A4 code

        char_embs = self.char_embedding_layer(input_tensor)
        sent_len, batch_size, max_word, _ = char_embs.shape
        cnn_shape = (sent_len*batch_size, max_word, self.char_embed_size)
        char_embs = char_embs.view(cnn_shape).permute(0, 2, 1)

        x_conv = self.cnn_layer(char_embs)
        dropout_output = self.highway_layer(x_conv)

        output = dropout_output.view(sent_len, batch_size, self.embed_size)
        return output
