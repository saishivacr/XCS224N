#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda --batch-size=2 --max-num-trial=30 \
        --max-epoch=60	# batch-size 2 -> 64

elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es outputs/test_outputs.txt --cuda

elif [ "$1" = "train_local" ]; then
  	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --batch-size=2

elif [ "$1" = "test_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    python run.py decode model.bin ./en_es_data/grader.es outputs/test_outputs.txt

elif [ "$1" = "train_local_q1" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder

elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q1.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt \
        --no-char-decoder

elif [ "$1" = "train_local_q2" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100
elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2.txt 

elif [ "$1" = "vocab" ]; then
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --size=200 --freq-cutoff=1 vocab_tiny_q1.json
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        vocab_tiny_q2.json
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json

###
elif [ "$1" = "my_train" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda --batch-size=64 --beam-size=10 \
        --max-epoch=60 --patience=10 --max-num-trial=30 --lr-decay=0.01 --uniform-init=0 --valid-niter=63

elif [ "$1" = "my_test" ]; then
    touch experimental_outputs/test_outputs_mytest.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es outputs/test_outputs_mytest.txt --cuda \
    --beam-size=10 --max-decoding-time-step=70

elif [ "$1" = "my_train_local" ]; then 
    python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --batch-size=64 --beam-size=10 \
        --max-epoch=60 --patience=10 --max-num-trial=30 --lr-decay=0.01 --uniform-init=0 --valid-niter=63

elif [ "$1" = "my_test_local" ]; then 
    touch experimental_outputs/test_outputs_mytest.txt
    python run.py decode model.bin ./en_es_data/test.es experimental_outputs/test_outputs_mytest.txt \
    --beam-size=10 --max-decoding-time-step=70
###
else
	echo "Invalid Option Selected"
fi
