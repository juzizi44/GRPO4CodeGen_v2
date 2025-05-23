#!/bin/bash

# 指定可见的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 设置模型路径
MODEL_PATH="/data/GRPO4CodeGen_v2/cachemodels/models--Qwen--Qwen2.5-Coder-7B-Instruct/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"

# 输入和输出路径
INPUT_FILE="/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-train.jsonl"
OUTPUT_FILE="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/Qwen2.5-Coder-7B-Instruct_train_output.jsonl"

# 运行推理脚本
python base_code_generator.py \
  --model_name_or_path "$MODEL_PATH" \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_length 3000 \
  --temperature 0.8 \
  --padding_side "left" \
  --torch_dtype "bfloat16"
