#!/bin/bash

# For evaluating the KV cache
python sample.py --init_from=gpt2-large --start="humaneval.jsonl" --batch_size=32 --max_new_tokens=128 --prompt_length=128 --num_samples=3 --num_warmup=1

# For evaluating speculative decoding
# python sample.py --init_from=gpt2-large --start="humaneval.jsonl" --batch_size=1 --max_new_tokens=256 --prompt_length=192 --num_samples=3 --num_warmup=1

