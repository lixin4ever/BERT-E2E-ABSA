#!/usr/bin/env bash
TASK_NAME="laptop14"
OUTPUT_DIR="./bert-tfm-rest_total-finetune"
CUDA_VISIBLE_DEVICES=0 python work.py --output_dir ${OUTPUT_DIR} \
                      --ckpt ${OUTPUT_DIR}/checkpoint-1400 \
                      --model_type bert \
                      --data_dir ./data/${TASK_NAME} \
                      --task_name ${TASK_NAME} \
                      --model_name_or_path bert-base-uncased \
                      --cache_dir ./cache \
                      --max_seq_length 128 \
                      --tagging_schema BIEOS