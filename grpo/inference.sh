# python inference.py \
#     --model_name_or_path "/data/GRPO4CodeGen_v2/cachemodels/models--Qwen--Qwen2.5-Coder-7B-Instruct/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242" \
#     --checkpoint_path "/data/GRPO4CodeGen_v2/grpo/output_model/20250519/" \
#     --checkpoint_step 1008 \
#     --device_ids "0,1" \
#     --torch_dtype "bfloat16" \
#     --lora_enable True \
#     --use_special_tokens False \
#     --max_length 3000 \
#     --input_file "/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl" \
#     --output_file "/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/grpo_onlycorrectness_Qwen2.5-Coder-7B-Instruct_checkpoint1008.jsonl"


# 设置模型路径
MODEL_PATH="/data/GRPO4CodeGen_v2/cachemodels/models--Qwen--Qwen2.5-Coder-7B-Instruct/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
# 输入和输出路径
INPUT_FILE="/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-train.jsonl"
OUTPUT_FILE="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/Qwen2.5-Coder-7B-Instruct_train_output.jsonl"
CUDA_VISIBLE_DEVICES=0,1 python inference.py \
  --model_name_or_path "$MODEL_PATH" \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_length 3000 \
  --torch_dtype "bfloat16"



