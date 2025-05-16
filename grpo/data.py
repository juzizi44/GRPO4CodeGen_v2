from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path


from datasets import load_dataset

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


from prompt_template.code_quality import (
   PROMPT
)

@dataclass
class DataConfig:
    """
    数据配置类，用于存储数据处理相关的参数
    """
    meta_data: str = "dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-train.jsonl"  # 元数据CSV文件路径
    meta_data_test: str = None  # 测试集元数据路径
    question_id: str = None  # 当前训练的问题ID
    max_samples: int = None  # 控制使用的最大数据量，None表示使用全部数据
    # eval_dim: str = "comment"  # 评估维度



def build_prompt(code_problem):

    return PROMPT.format(prompt=code_problem)


def convert_anno_csv_to_grpo_data(
        example, 
):
    # 返回处理后的prompt和question_id
    return {
        'prompt': build_prompt(example['prompt']),
        'question_id': str(example['question_id']),  # 确保question_id是字符串类型
        'preCodeSegment': example['preCodeSegment'],
        'postCodeSegment': example['postCodeSegment'],
        'unittests': example['unittests'],
    }


def create_dataset(data_config: DataConfig, meta_file=None):
    # 从json中加载数据，然后进行数据转换
    if meta_file is None:
        meta_file = data_config.meta_data
    dataset = load_dataset('json', data_files=meta_file)

    convert_func = lambda example : convert_anno_csv_to_grpo_data(
        example
    )

    # 只更新prompt列，保留其他列
    # dataset = dataset.map(convert_func, load_from_cache_file=False)
    print("="*50)
    dataset = dataset['train']
    
    # 限制数据量
    if data_config.max_samples is not None and data_config.max_samples > 0:
        dataset = dataset.select(range(min(data_config.max_samples, len(dataset))))
        
    print(dataset)
    print("="*50)
    return dataset





if __name__ == '__main__':
    """
    主函数：用于测试数据加载和处理流程
    """

    data_args = DataConfig()

    dataset = create_dataset(data_args, meta_file=data_args.meta_data)
    dataset = iter(dataset)
    for iter in dataset:
        print(iter)
        break