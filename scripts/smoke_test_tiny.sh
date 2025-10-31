#!/bin/bash

TINY_HIDDEN=240
TINY_HEADS=12
TINY_HEAD_SIZE=20
TINY_BATCH=2048
TINY_LR=0.1

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 distributed_rge.py --tokenizer hf --model_type LSTM --hidden_size $TINY_HIDDEN --num_heads $TINY_HEADS --head_size $TINY_HEAD_SIZE --batch_size $TINY_BATCH --learning_rate $TINY_LR
