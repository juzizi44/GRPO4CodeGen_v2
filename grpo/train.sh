export TOKENIZERS_PARALLELISM=false 

deepspeed --master_port=28508 --include localhost:4,5,6,7 train.py \
    --lora_enable True \
    --freeze_llm False \
    --lora_r 16 \
    --lora_alpha 128 \
    --lora_namespan_exclude "['score', 'rm_head', 'embed_tokens']" \
    --bf16 True \
    --torch_dtype "bfloat16" \
    --num_lora_modules -1 \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --meta_data "/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-train.jsonl" \
    --max_samples 1000 \
    --output_dir output_model/20250519 \
    --output_dim 1 \
    --use_special_tokens False \
    --reward_token "special" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 12 \
    --num_generations 4 \
    --num_iterations 2 \
    --learning_rate 1e-5 \
    --special_token_lr 1e-5  \
    --report_to tensorboard \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "epoch" \
    --logging_steps 10 \
    --eval_epochs 20 \
    --save_epochs 2 \
    --max_length 4800 \
    --gradient_checkpointing True \
    --deepspeed ds_config/zero0.json \
    --save_only_model True \
    --save_full_model False \
    --dataloader_num_workers 8 \
    --max_prompt_length 1800 \
    --max_completion_length 3000 \
    --reward_dimensions correctness \
    --use_weight_net False \
    --fixed_weights "{\"correctness\": 1}" \

    # --beta 0.005 \
    # --optim "adamw_8bit" \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.99 \
    # --weight_decay 0.1 \
    # --max_grad_norm 0.1 \
    # --log_on_each_node False \
    # --use_vllm False \
    # --load_from_pretrained "./output_model/20250414/" \
    # --load_from_pretrained_step 60 \
    # --reward_dimensions "false" \
    # --fixed_weights "{\"maintainability\": 0.7, \"comment\": 0.3}" \
    # --reward_dimensions correctness maintainability \
# ValueError: The global train batch size (3 x 2) must be evenly divisible by the number of generations per prompt (4). Given the current train batch size, the valid values for the number of generations are: [2, 3, 6].