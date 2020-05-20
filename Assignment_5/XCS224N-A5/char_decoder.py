#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 44 : xavier initialization

import torch
import torch.nn as nn
from vocab import VocabEntry


class CharDecoder(nn.Module):
    def __init__(self, hidden_size: int,
                 char_embedding_size: int = 50,
                 target_vocab: VocabEntry = None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character
            embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language.
            See vocab.py for documentation.
        """
        # YOUR CODE HERE for part 2a
        # TODO - Initialize as an nn.Module.
        #      - Initialize the following variables:
        #        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        #        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        #        self.decoderCharEmb: Embedding matrix of character embeddings
        #        self.target_vocab: vocabulary for the target language
        #
        # Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        #       - Set the padding_idx argument of the embedding matrix.
        #       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()

        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                                   hidden_size=hidden_size)
        
        # logit calculations (pass output through softmax to get next char)
        self.char_output_projection = nn.Linear(in_features=hidden_size,
                                                out_features=len(target_vocab.char2id),  # Project onto all possible chars
                                                bias=True)
        nn.init.xavier_uniform_(self.char_output_projection.weight, gain=1) #TODO: calculate softmax gain

        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id),
                                           embedding_dim=char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>'])
        self.padding_idx = target_vocab.char2id['<pad>']
        self.len_vocab = len(target_vocab.char2id)
        self.hidden_size = hidden_size
        # END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Single forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading
            the input characters.
            A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF.
            shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading
            the input characters.
            A tuple of two tensors of shape (1, batch, hidden_size)
        """
        # YOUR CODE HERE for part 2b
        # TODO - Implement the forward pass of the character decoder.

        # l, b = input.shape[0], input.shape[1]
        # shape is (len, batch, char_embedding_size)
        char_embs = self.decoderCharEmb(input)
        
        # `out` accesses all the hidden states of the sequence,
        # `dec_hidden` accesses the most recent hidden state and allows
        # you to backprop and feed the most recent timestep back
        # into the model as input
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

        # hidden shape is (len, batch, hidden_size)
        out, dec_hidden = self.charDecoder(char_embs, dec_hidden)
        # print(len(out))
        # print(len(dec_hidden))
        # print(f'hidden shape is {dec_hidden[0].shape}')
        # print(f'hidden shape should be (1, {b}, {self.hidden_size})')
        # print(f'output shape is {out.size()}')
        
        # score shape is (len, batch, len_vocab)
        scores = self.char_output_projection(out)
        # print(f'score shape is {scores.size()}')
        # print(f'score shape should be ({l}, {b}, {self.len_vocab})')
        return scores, dec_hidden

        # END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch).
            Note that "length" here and in forward() need not be the same.

            char_sequence corresponds to the sequence x_1 ... x_{n+1} 
            (e.g., <START>,m,u,s,i,c,<END>)
        
        @param dec_hidden: initial internal state of the LSTM, obtained
            from the output of the word-level decoder.
            A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of
            cross-entropy losses of all the words in the batch,
            for every character in the sequence.
            
            Padding characters do not contribute to the
            cross-entropy loss.
        """
        # TODO fix truncation

        # Implement training forward pass.
        
        # Truncate <END> character
        input_seq = char_sequence[:-1]
        # Get the candidates for next characters
        # shape is (len, batch, vocab_len)
        scores, dec_hidden = self.forward(input_seq, dec_hidden)
        scores = scores.view(-1, scores.shape[-1])

        # Get the target sequences against which to train
        # Note: the decoder is trained to predict all words,
        # not just <UNK> tokens.

        # Truncate <START> chars
        target = char_sequence[1:].contiguous().view(-1)

        # Note: When computing cross-entropy, take the SUM,
        # not the AVERAGE, and ignore `<pad>` characters.
        # https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html

        loss = nn.CrossEntropyLoss(reduction='sum',
                                   ignore_index=self.padding_idx)

        return loss(scores, target)

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of
            two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or
            GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of
            which has length <= max_length.
            The decoded strings should NOT contain the start-of-word and
            end-of-word characters.
        """

        # YOUR CODE HERE for part 2d
        # TODO - Implement greedy decoding.
        # Hints:
        #      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        #      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        #      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        #        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        start_index = self.target_vocab.start_of_word
        end_index = self.target_vocab.end_of_word
        dec_hidden = initialStates

        batch_size = initialStates[0].shape[1]
        
        # start_char = [[start_index]] * batch_size
        input_seq = torch.tensor([start_index for _ in range(batch_size)],
                                  device=device).unsqueeze(0)
        # input = torch.tensor(start_char, device=device).transpose(1, 0)

        decoded = [["", False] for _ in range(batch_size)]

        for _ in range(max_length):
            score, dec_hidden = self.forward(input_seq, dec_hidden)  # score shape: (len, batch_size, vocab_len)
            input_seq = score.argmax(dim=2)
        
            for seq_idx, char_idx in enumerate(input_seq.detach().squeeze(0)):
                if not decoded[seq_idx][1]:
                    if char_idx != end_index:
                        decoded[seq_idx][0] += self.target_vocab.id2char[char_idx.item()]
                    else:
                        decoded[seq_idx][1] = True
        return [w[0] for w in decoded]
        

