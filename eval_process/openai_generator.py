import os
import json
from tqdm import tqdm
from typing import Optional
from openai import OpenAI
import time
from api_key import APIKEY2
class OpenAICodeGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.8,
        system_prompt: str = ""
    ):
        """
        初始化OpenAI代码生成器
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
            temperature: 生成温度，用于控制输出的随机性
            system_prompt: 系统提示词
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_retries: int = 2,
        num_samples: Optional[int] = None
    ):
        """
        从输入文件批量生成代码并保存到输出文件
        Args:
            input_file: 输入JSONL文件路径，每行包含至少"index"和"prompt"字段
            output_file: 输出JSONL文件路径，每行包含生成结果以及原始索引
            max_retries: 每个样本的最大重试次数
            num_samples: 最多处理的样本数量，None表示处理所有样本
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        # 确定进度条总数
        if num_samples is not None:
            total = num_samples
        else:
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

                # 尝试生成，带重试机制
                attempt = 0
                response = None
                while attempt < max_retries:
                    try:
                        attempt += 1
                        print("=" * 50)
                        print(f"🔹 Attempt: {attempt}")
                        print(f"🔹 Model: {self.model}")
                        print(f"🔹 Base URL: {self.base_url}")
                        print(f"🔹 apikey: {self.api_key}")
                        print(f"🔹 Client: {self.client}")
            
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=self.temperature
                        )
                        response = completion.choices[0].message.content
                    
                        break
                    except Exception as e:
                        print(f"Attempt {attempt} failed for index {idx}: {e}")
                        if attempt < max_retries:
                            print("Retrying...")

                # 如果所有重试都失败，记录失败结果
                if response is None:
                    response = "failed"

                # 写入结果并立即刷新，保留原始所有字段并加上response
                data["response"] = response
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()

                processed += 1
                progress.update(1)

        progress.close()

# 示例用法
if __name__ == "__main__":
    generator = OpenAICodeGenerator(

        api_key=APIKEY2,
        
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini-2025-04-14",
        temperature=0.8,
        system_prompt=""
    )
    
    generator.generate_from_file(
        input_file="/data/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl",
        output_file="/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/gpt-4.1-mini-2025-04-14_testoutput.jsonl",
        num_samples=None,
        max_retries=2
    )

