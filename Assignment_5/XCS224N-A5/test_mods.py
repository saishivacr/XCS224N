"""
CS224N 2018-19: Homework 5
test_mods.py: sanity checks for assignment 5 modules
Usage:
    test_mods.py 1d
    test_mods.py 1e
    test_mods.py 1f
"""

import json
import pickle
from sanity_check import BATCH_SIZE, DROPOUT_RATE, EMBED_SIZE, HIDDEN_SIZE
import sys
from collections import namedtuple

import numpy as np
import torch
import torch.nn.utils
import torch.nn as nn 
import torch.nn.functional as F

from docopt import docopt

from char_decoder import CharDecoder
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry
from highway import Highway

torch.manual_seed(0)
np.random.seed(0)

# Test parameters

BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0


def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)

def generate_data(batch_size, embed_size):
    '''
    Generate fake CNN output and gold outputs.
    
    x_conv: input values
    W_proj, W_gate: weight matrices (embed_size, embed_size)
    b_proj, b_gate: bias vectors (embed_size)
    '''
    Params = namedtuple("Params", "x_conv W_proj b_proj W_gate b_gate")
    Outputs = namedtuple("Outputs", "x_highway_gold output_size_gold")

    dropout = nn.Dropout(p=DROPOUT_RATE)

    x_conv = torch.randn(batch_size, embed_size)

    # highway projection layer
    W_proj = torch.randn(embed_size, embed_size)
    b_proj = torch.randn(embed_size)

    # highway gate layer
    W_gate = torch.randn(embed_size, embed_size)
    b_gate = torch.randn(embed_size)

    params_gold = Params(x_conv, W_proj, b_proj, W_gate, b_gate)

    # Gold outputs
    x_proj_gold = F.relu(x_conv @ W_proj.T + b_proj)
    x_gate_gold = torch.sigmoid(x_conv @ W_gate.T + b_gate)
    x_highway_gold = dropout(x_gate_gold * x_proj_gold +
                             (1 - x_gate_gold) * x_conv)
    output_size_gold = (BATCH_SIZE, EMBED_SIZE)

    outputs_gold = Outputs(x_highway_gold, output_size_gold)

    return params_gold, outputs_gold


def test_1d():
    '''
    Check the output of the highway layer.
    '''

    print("-"*80)
    print("Running sanity checks for Question 1d: Highway")
    print("-"*80)

    params_gold, outputs_gold = generate_data(BATCH_SIZE, EMBED_SIZE)

    # Initialize model with generated weights
    hw_model = Highway(EMBED_SIZE, DROPOUT_RATE)
    
    hw_model.projection_layer.weight.data = params_gold.W_proj
    hw_model.projection_layer.bias.data = params_gold.b_proj

    hw_model.gate_layer.weight.data = params_gold.W_gate
    hw_model.gate_layer.bias.data = params_gold.b_gate

    # Use the model
    x_test = hw_model(params_gold.x_conv)

    # Check output shape
    assert(x_test.size() == outputs_gold.output_size_gold), \
        f"Shape of highway output is incorrect.\nExpected:\n \
            {outputs_gold.output_size_gold}\nActual:\n{x_test.size()}"

    # Check output validity
    assert((x_test == outputs_gold.x_highway_gold).all()), \
        f"Highway output is incorrect.\nExpected:\n \
            {outputs_gold.x_highway_gold}\nActual:\n{x_test}"

    print("Sanity checks passed for Question 1d: Highway!")
    print("-"*80)

def test_1e():

    import torch
    import json
    from nmt_model import NMT
    from vocab import Vocab

    sentence_length = 10
    max_word_length = 21
    batch_size = 5
    embed_size = 3

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    print('-'*80)
    print('Running Sanity Checks for Question 1e:  Conv1d')
    def test_model(sentence_length, batch_size, max_word_length, embed_size):
        model = NMT(embed_size=embed_size, hidden_size=3, dropout_rate=0.2, vocab=vocab)
        x = torch.zeros(sentence_length, batch_size, max_word_length,   dtype=torch.long)
        ME_source = model.model_embeddings_source
        output = ME_source.forward(x)
        output_expected_size = [sentence_length, batch_size, embed_size]

        assert list(output.size()) == output_expected_size, f'Output shape should be:\n{output_expected_size}, but is:\n{list(output.size())}'

    params = (10, 5, 21, 3)
    test_model(*params)

    #print('Testing batch size 1')
    params = (10, 1, 21, 3)
    test_model(*params)

    #print('Testing max word length = sentence length')
    params = (21, 5, 21, 3)
    test_model(*params)

    #print('Testing max word length = 20')
    #params = (10, 5, 20, 3)
    #test_model(*params)
    # print('Testing embed_size=1')
    
    params = (10, 5, 21, 1)
    test_model(*params)
    print('-'*80)
    print('Sanity Checks Passed for Question 1e: Conv1d!')


def main():
    args = docopt(__doc__)
    if args['1d']:

        test_1d()
    elif args['1e']:
        test_1e()
    else:
        raise RuntimeError("Invalid argument for testing")

if __name__ == "__main__":

    main()