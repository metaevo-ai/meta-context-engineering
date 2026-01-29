#!/bin/bash

uv run python -m mce.main \
    --workspace "workspace/symptom_diagnosis" \
    --env "symptom_diagnosis" \
    --train-data "env/symptom_diagnosis/data/train.jsonl" \
    --val-data "env/symptom_diagnosis/data/val.jsonl" \
    --model "deepseek/deepseek-chat-v3.1" \
    --iterations 3 \
    --train-limit 50 \
    --val-limit 20 \
    --train-batch-size 25 \
    --log-dir "logs/symptom_diagnosis"
