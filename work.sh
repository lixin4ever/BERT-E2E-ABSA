#!/usr/bin/env bash
TASK_NAME="test20"
ABSA_HOME="./bert-linear-laptop14-finetune"
CUDA_VISIBLE_DEVICES=0 python work.py --absa_home ${ABSA_HOME} \
                      --ckpt ${ABSA_HOME}/checkpoint-1500 \
                      --model_type bert \
                      --data_dir ./data/${TASK_NAME} \
                      --task_name ${TASK_NAME} \
                      --model_name_or_path bert-base-uncased \
                      --cache_dir ./cache \
                      --max_seq_length 128 \
                      --tagging_schema BIEOS