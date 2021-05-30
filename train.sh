#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --embed_dim 512 \
                      --ff_embed_dim 1024 \
                      --num_heads 8 \
                      --layers 6 \
                      --dropout 0.1 \
                      --train_data ./data/train_with_kw.txt \
                      --val_data ./data/val.txt \
                      --vocab ./data/vocab.txt \
                      --min_occur_cnt 0 \
                      --batch_size 32 \
                      --warmup_steps 10000 \
                      --lr 1e-4 \
                      --max_len 64 \
                      --world_size 4 \
                      --gpus 4 \
                      --start_rank 0 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 28512 \
                      --print_every 10000 \
                      --valid_every 30000 \
                      --save_every 30000 \
                      --save_dir ckpt \
                      --backend nccl \
                      --use_src_attn 1 \
                      --use_resp_kw 1