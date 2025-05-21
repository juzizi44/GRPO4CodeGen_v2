import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Optional


class BaseCodeGenerator:
    def __init__(
        self,
        base_model_path: str,
        temperature: float = 0.8,
        padding_side: str = "left"
    ):
        """
        初始化基础模型生成器
        Args:
            base_model_path: 基础模型路径（Hugging Face 格式）
            temperature: 生成温度，用于控制输出的随机性
            padding_side: 填充方向，可选 "left" 或 "right"
        """
        self.base_model_path = base_model_path
        self.temperature = temperature

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side=padding_side
        )

        # 加载模型并分配到 GPU（自动分配多卡）
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 2048,
        num_samples: Optional[int] = None
    ):
        """
        从输入文件批量生成代码并保存到输出文件
        """
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        if num_samples is not None:
            total = num_samples
        else:
            with open(input_file, 'r', encoding='utf-8') as f_tmp:
                total = sum(1 for _ in f_tmp)

        progress = tqdm(total=total, desc="生成进度")
        processed = 0

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line in fin:
                if num_samples is not None and processed >= num_samples:
                    break

                data = json.loads(line)
                prompt = data.get("prompt", "")

                # 构造对话 prompt（适配 Qwen Chat 模型）
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 编码并移动到 GPU
                device = next(self.model.parameters()).device
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # 推理
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                # 解码输出
                output_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # 截取回答部分
                response = output_text[len(input_text):].strip()

                # 写入结果
                result = dict(data)
                result["response"] = response
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                processed += 1
                progress.update(1)

        progress.close()


# 示例用法
if __name__ == "__main__":
    generator = BaseCodeGenerator(
        base_model_path="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.8,
        padding_side="left"
    )
    generator.generate_from_file(
        input_file="/data/ytan089/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl",
        output_file="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/Qwen2.5-Coder-32B-Instruct_testoutput.jsonl",
        max_new_tokens=2000,
        num_samples=None
    )