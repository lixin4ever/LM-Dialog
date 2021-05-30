#!/usr/bin/env bash
# perform inference using the saved checkpoint
MODEL='model3_layer6_epoch4_batch239999'
DECODE_TYPE='tk'
K='20'
python eval.py --model ${MODEL} \
               --test_data ./data/test.txt \
               --vocab ./data/vocab.txt \
               --decode_type ${DECODE_TYPE} \
               --gpu 3 \
               --k ${K}
python metrics.py ${MODEL} ${DECODE_TYPE} ${K} 0

