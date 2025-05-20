# File: grpo_code_generator.py
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import safetensors
from typing import Optional, List, Dict

def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility to insert adapter name into state_dict keys."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
        peft_model_state_dict[key] = val
    return peft_model_state_dict


def _adjust_state_dict_keys(state_dict, current_model_keys):
    """Adjust state_dict keys to match current model."""
    adjusted = {}
    old_prefix = "base_model.model.base_model.model.model."
    new_prefix = "base_model.model.model."
    special = {"base_model.model.base_model.model.lm_head.": "base_model.model.lm_head."}
    has_adapter = any(".default." in k for k in state_dict)
    for k, v in state_dict.items():
        mapped = False
        for old, new in special.items():
            if k.startswith(old):
                adjusted[new + k[len(old):]] = v
                mapped = True
                break
        if mapped:
            continue
        if k.startswith(old_prefix):
            adjusted[new_prefix + k[len(old_prefix):]] = v
        else:
            if "base_model.model.base_model.model" in k:
                candidate = k.replace("base_model.model.base_model.model", "base_model.model")
                if candidate in current_model_keys:
                    adjusted[candidate] = v
                    continue
            adjusted[k] = v
    return adjusted, has_adapter


class GrpoCodeGenerator:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda:0",
        use_adapter: bool = True,
        temperature: float = 0.8,
        padding_side: str = "left"
    ):
        """Initialize GRPO code generator."""
        self.device = torch.device(device)
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = padding_side
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16
        ).to(self.device)
        if use_adapter:
            self._load_adapter(adapter_path)
        self.model.eval()

    def _load_adapter(self, adapter_path: str):
        """Load adapter weights into the model."""
        sd = self.model.state_dict()
        lora_ckpt = os.path.join(adapter_path, "adapter_model.safetensors")
        non_lora_ckpt = os.path.join(adapter_path, "non_lora_state_dict.pth")
        if os.path.exists(lora_ckpt):
            lora_sd = safetensors.torch.load_file(lora_ckpt)
            lora_sd, has_name = _adjust_state_dict_keys(lora_sd, sd.keys())
            if not has_name:
                lora_sd = _insert_adapter_name_into_state_dict(
                    lora_sd, adapter_name="default", parameter_prefix="lora_"
                )
            sd.update(lora_sd)
        if os.path.exists(non_lora_ckpt):
            non_lora_sd = torch.load(non_lora_ckpt, map_location="cpu")
            non_lora_sd, _ = _adjust_state_dict_keys(non_lora_sd, sd.keys())
            sd.update(non_lora_sd)
        self.model.load_state_dict(sd, strict=False)

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        batch_size: int = 4
    ) -> List[str]:
        """Batch generate code, returning only generated part (no prompt)."""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            messages = [
                [ {'role':'system','content':''}, {'role':'user','content':p} ]
                for p in batch
            ]
            inputs = self.tokenizer(
                [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages],
                return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            input_ids = inputs.input_ids
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.8,
                    top_k=40,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )
            # slice off prompt tokens
            for seq, prompt in zip(outputs, batch):
                gen_tokens = seq[input_ids.shape[1]:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                results.append(text)
        return results

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 1024,
        batch_size: int = 4,
        num_samples: Optional[int] = None
    ):
        """Load prompts from JSONL, generate and save only model output."""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # 计算总样本数
        if num_samples is not None:
            total = num_samples
        else:
            with open(input_file, 'r', encoding='utf-8') as f_tmp:
                total = sum(1 for _ in f_tmp)
                
        # 初始化进度条
        progress = tqdm(total=total, desc="生成进度")
        processed = 0
        
        # 打开输出文件，准备边生成边写入
        with open(output_file, 'w', encoding='utf-8') as fout:
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if num_samples is not None and i >= num_samples:
                        break
                    item = json.loads(line)
                    data.append(item)
                    
                    # 当收集到一个批次的数据时，进行生成
                    if len(data) == batch_size or (i == total - 1 and data):
                        prompts = [item['prompt'] for item in data]
                        responses = self.generate_batch(prompts, max_new_tokens, batch_size)
                        
                        # 保存这个批次的结果
                        for item, resp in zip(data, responses):
                            out = {"index": item["index"], "prompt": item["prompt"], "response": resp}
                            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                            fout.flush()  # 确保立即写入磁盘
                        
                        # 更新进度
                        batch_processed = len(data)
                        processed += batch_processed
                        progress.update(batch_processed)
                        
                        # 清空数据列表，准备下一个批次
                        data = []
        
        progress.close()
        print(f"结果已保存至: {output_file}")

if __name__ == '__main__':
    gen = GrpoCodeGenerator(
        base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path="/data/GRPO4CodeGen_v2/grpo/output_model/20250519/checkpoint-112",
        device="cuda:1",
        use_adapter=True
    )
    gen.generate_from_file(
        input_file="/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl",
        output_file="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/output_grpo.jsonl",
        max_new_tokens=2000,
        batch_size=1,
        num_samples=10
    )
