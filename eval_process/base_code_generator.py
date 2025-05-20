# File: base_code_generator.py
import os
import json
import torch
from transformers import pipeline
from tqdm import tqdm
from typing import Optional

class BaseCodeGenerator:
    def __init__(
        self,
        base_model_path: str,
        device: str = "cuda:0",
        temperature: float = 0.8,
        padding_side: str = "left"
    ):
        """
        初始化基础模型生成器
        Args:
            base_model_path: 基础模型路径（Hugging Face 格式）
            device: 运行设备，例如 "cuda:0" 或 "cpu"
            temperature: 生成温度，用于控制输出的随机性
            padding_side: 填充方向，可选 "left" 或 "right"
        """
        self.base_model_path = base_model_path
        self.device = device
        self.temperature = temperature
        
        # 使用pipeline API加载模型
        self.pipe = pipeline(
            "text-generation",
            model=base_model_path,
            device=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 2048,
        num_samples: Optional[int] = None
    ):
        """
        从输入文件批量生成代码并保存到输出文件
        Args:
            input_file: 输入 JSONL 文件路径，每行包含至少 "index" 和 "prompt" 字段
            output_file: 输出 JSONL 文件路径，每行包含生成结果以及原始索引；若路径目录不存在，会自动创建
            max_new_tokens: 每次生成的最大 token 数
            num_samples: 最多处理的样本数量，None 表示处理所有样本
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        # 确定进度条总数
        if num_samples is not None:
            total = num_samples
        else:
            # 统计总行数以便完整进度展示
            with open(input_file, 'r', encoding='utf-8') as f_tmp:
                total = sum(1 for _ in f_tmp)

        progress = tqdm(total=total, desc="生成进度")
        processed = 0

        # 逐行读取并生成，同时保存结果
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                if num_samples is not None and processed >= num_samples:
                    break
                data = json.loads(line)
                idx = data.get("index")
                prompt = data.get("prompt", "")
        
         
                # 构造消息格式输入
                messages = [
                    {"role": "user", "content": prompt}
                ]

                # 使用pipeline生成响应
                response_obj = self.pipe(
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
             
                )
                
                # 从生成的结果中提取响应文本
                response = response_obj[0]['generated_text']
                
                # 如果响应包含模型回复部分，可能需要提取出实际的回复内容
                # 这里假设response中包含了完整对话，需要提取助手的回复部分
                # 具体提取逻辑可能需要根据模型输出格式调整
                if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                    # 如果返回格式是消息列表，提取助手回复
                    for msg in response:
                        if msg.get("role") == "assistant":
                            response = msg.get("content", "")
                            break
                
            

                # 写入结果并立即刷新
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
        base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        device="cuda:0",
        temperature=0.8,
        padding_side="left"
    )
    # num_samples=50 表示最多生成 50 条；若设为 None，则处理全部
    generator.generate_from_file(
        input_file="/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl",
        output_file="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/Qwen2.5-Coder-7B-Instruct_testoutput.jsonl",
        max_new_tokens=3000,
        num_samples=None
    )
