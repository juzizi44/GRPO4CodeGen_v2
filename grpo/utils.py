import os
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union, Dict, Any
import json
from pathlib import Path

import safetensors
import torch
from transformers import TrainingArguments
from trl import GRPOConfig
# from trl import GRPOConfig as BaseGRPOConfig


@dataclass
class TrainingConfig(GRPOConfig):
    max_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    disable_flash_attn2: bool = field(default=False)
    reward_device: Optional[str] = None

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    special_token_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False
    reward_model_path: Optional[str] = None

    save_full_model: Optional[bool] = False
    num_generations: Optional[int] = 4
    
    # 新增参数 - 改为使用Optional[List[str]]，用字符串"False"表示禁用奖励维度
    reward_dimensions: Optional[List[str]] = field(default_factory=lambda: ["correctness", "efficiency", "comment", "maintainability"])
    use_weight_net: bool = field(default=True)
    fixed_weights: Optional[str] = None
    execute_code_url: Optional[str] = None

@dataclass
class PEFTLoraConfig:
    lora_enable: bool = False
    # vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"

    output_dim: int = 1

    use_special_tokens: bool = False

    # freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    # tune_merger: bool = field(default=False)

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    # loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
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
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _adjust_state_dict_keys(state_dict, current_model_keys):
    """
    调整状态字典的键名以匹配当前模型结构
    
    Args:
        state_dict: 加载的状态字典
        current_model_keys: 当前模型的键名列表
    
    Returns:
        调整后的状态字典
    """
    adjusted_state_dict = {}
    
    # 确定正确的前缀映射
    old_prefix = "base_model.model.base_model.model.model."
    new_prefix = "base_model.model.model."
    
    # 特殊处理非标准层级
    special_patterns = {
        "base_model.model.base_model.model.lm_head.": "base_model.model.lm_head."
    }
    
    print(f"===> Attempting to map keys from '{old_prefix}' to '{new_prefix}' and handle special patterns")
    
    # 计数器用于统计
    mapped_keys = 0
    special_mapped_keys = 0
    ignored_keys = 0
    
    # 检查键是否已包含适配器名
    has_adapter_name = any(".default." in k for k in state_dict.keys())
    
    for k, v in state_dict.items():
        # 首先检查特殊模式
        mapped = False
        for old_pattern, new_pattern in special_patterns.items():
            if k.startswith(old_pattern):
                new_key = new_pattern + k[len(old_pattern):]
                adjusted_state_dict[new_key] = v
                special_mapped_keys += 1
                mapped = True
                break
                
        if mapped:
            continue
                
        # 标准前缀映射
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
            adjusted_state_dict[new_key] = v
            mapped_keys += 1
        else:
            # 检查是否可以通过其他规则映射
            if "base_model.model.base_model.model" in k:
                possible_new_key = k.replace("base_model.model.base_model.model", "base_model.model")
                if possible_new_key in current_model_keys:
                    adjusted_state_dict[possible_new_key] = v
                    mapped_keys += 1
                    continue
            
            # 如果没有匹配规则，保留原始键
            adjusted_state_dict[k] = v
            ignored_keys += 1
    
    # 打印调整的统计信息
    print(f"===> Mapped {mapped_keys} keys from standard prefix")
    print(f"===> Mapped {special_mapped_keys} keys from special patterns")
    print(f"===> Retained {ignored_keys} keys without mapping")
    print(f"===> Total keys in adjusted state_dict: {len(adjusted_state_dict)}")
    print(f"===> State dict already contains adapter name: {has_adapter_name}")
    
    return adjusted_state_dict, has_adapter_name


def load_model_from_checkpoint(
    model, checkpoint_dir, checkpoint_step
):
    print(f"===> Loading checkpoint from {checkpoint_dir}, step {checkpoint_step}")
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    print(f"===> Found checkpoints: {checkpoint_paths}")
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        # get the latest checkpoint
        checkpoint_path = checkpoint_paths[0]
        print(f"===> Checkpoint step is not provided, using the latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        if checkpoint_path not in checkpoint_paths:
            checkpoint_path = checkpoint_paths[0]
            print(f"===> Checkpoint step {checkpoint_step} not found, using the latest checkpoint: {checkpoint_path}")
        else:
            print(f"===> Checkpoint step {checkpoint_step} found, using the specified checkpoint: {checkpoint_path}")
    
    checkpoint_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

    
    print(f"===> Checking for checkpoint files:")
    print(f"  - Full checkpoint: {os.path.exists(full_ckpt)}")
    print(f"  - LoRA checkpoint: {os.path.exists(lora_ckpt)}")
    print(f"  - Non-LoRA checkpoint: {os.path.exists(non_lora_ckpt)}")

    
    # 打印当前模型的state_dict键
    current_model_keys = list(model.state_dict().keys())
    print(f"===> Current model state_dict keys (first 10):")
    for i, key in enumerate(current_model_keys[:10]):
        print(f"  {i}: {key}")
    print(f"Total keys in current model: {len(current_model_keys)}")
    
    if os.path.exists(full_ckpt):
        print("===> Loading full checkpoint")
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        
        # 打印加载的状态字典的键
        loaded_keys = list(model_state_dict.keys())
        print(f"===> Loaded state_dict keys (first 10):")
        for i, key in enumerate(loaded_keys[:10]):
            print(f"  {i}: {key}")
        print(f"Total keys in loaded state_dict: {len(loaded_keys)}")
        
        # 分析键名差异
        diff_keys = set(loaded_keys) - set(current_model_keys)
        print(f"===> Keys in loaded state_dict but not in current model (first 10):")
        for i, key in enumerate(list(diff_keys)[:10]):
            print(f"  {i}: {key}")
        print(f"Total different keys: {len(diff_keys)}")
        
        # 如果存在键名差异，尝试调整状态字典键名
        if len(diff_keys) > 0:
            print("===> Attempting to adjust state_dict keys...")
            model_state_dict, _ = _adjust_state_dict_keys(model_state_dict, current_model_keys)
            
        # 添加strict=False参数允许部分键不匹配
        try:
            model.load_state_dict(model_state_dict)
            print("===> Successfully loaded model with exact key matching")
        except RuntimeError as e:
            print(f"===> Error loading model with exact matching: {e}")
            print("===> Trying to load with strict=False...")
            model.load_state_dict(model_state_dict, strict=False)
            print("===> Successfully loaded model with strict=False")
    else:
        print("===> Loading split checkpoints (LoRA + non-LoRA)")
        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = {}
        if os.path.exists(non_lora_ckpt):
            non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")

        print(f"===> LoRA state dict keys (first 5):")
        lora_keys = list(lora_state_dict.keys())
        for i, key in enumerate(lora_keys[:5]):
            print(f"  {i}: {key}")
        print(f"Total keys in LoRA state_dict: {len(lora_keys)}")
        
        print(f"===> Non-LoRA state dict keys (first 5):")
        non_lora_keys = list(non_lora_state_dict.keys())
        for i, key in enumerate(non_lora_keys[:5]):
            print(f"  {i}: {key}")
        print(f"Total keys in non-LoRA state_dict: {len(non_lora_keys)}")

        # 调整LoRA和非LoRA状态字典的键名
        print("===> Adjusting LoRA state dict keys...")
        lora_state_dict, has_adapter_name = _adjust_state_dict_keys(lora_state_dict, current_model_keys)
        
        if non_lora_state_dict:
            print("===> Adjusting non-LoRA state dict keys...")
            non_lora_state_dict, _ = _adjust_state_dict_keys(non_lora_state_dict, current_model_keys)
        
        # 只有在需要时才插入适配器名称
        if not has_adapter_name:
            print("===> Inserting adapter name 'default'...")
            lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")
        else:
            print("===> Adapters already named, skipping name insertion")
        
        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        
        # 更新后再次打印键，检查合并后的状态
        combined_keys = list(model_state_dict.keys())
        print(f"===> Combined state_dict keys (first 10):")
        for i, key in enumerate(combined_keys[:10]):
            print(f"  {i}: {key}")
        
        # 尝试加载模型，添加错误处理
        try:
            model.load_state_dict(model_state_dict)
            print("===> Successfully loaded model with exact key matching")
        except RuntimeError as e:
            print(f"===> Error loading model with exact matching: {e}")
            print("===> Trying to load with strict=False...")
            model.load_state_dict(model_state_dict, strict=False)
            print("===> Successfully loaded model with strict=False")

    return model, checkpoint_step


def load_problem_unittests_by_id(jsonl_path: str, question_id: str) -> List[Dict[str, Any]]:
    """
    从数据集中读取指定question_id的unittests
    Args:
        jsonl_path: 数据集路径
        question_id: 题目ID
    Returns:
        unittests列表
    """
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if str(data.get('question_id', '')) == str(question_id):
                    return data.get('unittests', [])
    except Exception as e:
        print(f"读取unittests失败: {str(e)}")
        return []
    return []

