import os
import json
import torch
import argparse
from dataclasses import asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import ModelConfig, PEFTLoraConfig, load_model_from_checkpoint
from trl import get_kbit_device_map, get_quantization_config

def create_model_and_tokenizer(model_config, peft_lora_config, cache_dir=None):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="right",
        cache_dir=cache_dir,
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ['<|reward|>']
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
        local_files_only=False,
        trust_remote_code=True,
        **model_kwargs
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def generate_response(model, tokenizer, prompt, device, max_length=3000):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

def inference(args):
    device = f"cuda:{args.device_ids.split(',')[0]}" if args.device_ids else ("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        use_special_tokens=args.use_special_tokens,
    )
    peft_lora_config = PEFTLoraConfig(
        lora_enable=args.lora_enable,
    )

    model, tokenizer = create_model_and_tokenizer(
        model_config=model_config,
        peft_lora_config=peft_lora_config,
    )

    if args.checkpoint_path:
        print(f"从 checkpoint 加载模型: {args.checkpoint_path}")
        model, checkpoint_step = load_model_from_checkpoint(
            model,
            args.checkpoint_path,
            args.checkpoint_step,
        )
    else:
        print(f"未提供 checkpoint，直接使用模型路径 {args.model_name_or_path}")

    model.to(device)
    model.eval()
    print(f"模型已加载到设备: {device}")

    input_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            input_data.append(json.loads(line))
    print(f"加载了 {len(input_data)} 条数据")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(input_data):
            print(f"处理第 {i+1}/{len(input_data)} 条数据...")
            prompt = item.get('prompt', '')
            if not prompt:
                print(f"警告：第 {i+1} 条数据缺少prompt字段")
                continue
            response = generate_response(model, tokenizer, prompt, device, args.max_length)
            result = item.copy()
            result['response'] = response
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(f"第 {i+1}/{len(input_data)} 条数据已处理并保存")

    print(f"推理完成，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理脚本")

    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="模型名称或路径")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="检查点路径（可选）")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="检查点步数（可选）")
    parser.add_argument("--device_ids", type=str, default=None,
                        help="使用的GPU ID，用逗号分隔")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        help="模型数据类型")
    parser.add_argument("--lora_enable", type=bool, default=True,
                        help="是否启用LoRA")
    parser.add_argument("--use_special_tokens", type=bool, default=False,
                        help="是否使用特殊token")
    parser.add_argument("--max_length", type=int, default=3000,
                        help="生成的最大长度")
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出文件路径")

    args = parser.parse_args()
    inference(args)
