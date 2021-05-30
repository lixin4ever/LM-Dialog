#!/usr/bin/env bash
python prepare_data.py \
    --src_file /home/tedxli/dataset/dialog-700w-filtered/train.txt \
    --tgt_file ./data/train.txt \
    --nprocessors 8
