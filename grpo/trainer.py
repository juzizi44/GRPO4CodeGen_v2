import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Tuple, List, Dict
import aiohttp
import asyncio
from typing import List, Tuple
import torch
import torch.utils.data
from peft import PeftModel
import safetensors
from transformers import (
    PreTrainedModel,
)
from transformers.trainer import (
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from radon.metrics import mi_visit

from transformers.utils import is_peft_available

from utils import get_peft_state_non_lora_maybe_zero_3,load_problem_unittests_by_id


from trl import GRPOTrainer


import aiohttp
import asyncio
import re
from reward.reward_function import CodeCommentScorer
from data import create_dataset, DataConfig
from reward.reward_function import RealTimeRewardRunner, LLMCommentScorer
from reward.reward_weight_net import RewardWeightNet

def extract_code_from_completion(completion: str) -> str:
    """从 markdown 风格的字符串中提取代码块；若没有则返回原始内容"""
    try:
        if not isinstance(completion, str):
            return ""

        # 提取所有 markdown 代码块
        code_blocks = re.findall(r"```(?:[a-zA-Z]*\n)?([\s\S]*?)```", completion)
        if code_blocks:
            return "\n".join(code_blocks).strip()

        # 若没有匹配的代码块，则返回原始内容
        return completion.strip()
    except Exception:
        return ""

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        meta_data_path: str,  # 添加数据集路径参数
        **kwargs
    ):
        # 初始化权重网络
        self.reward_weight_net = RewardWeightNet(num_dimensions=4)
        
        # 预留可学习系数，实际训练时需要实现参数化和优化
        self.reward_weights = {
            'correctness': 0.5,  # 初始权重，后续会被网络更新
            'efficiency': 0.15,
            'comment': 0.2,
            'maintainability': 0.15
        }
        kwargs['reward_funcs'] = [self.reward_func]
        super().__init__(model=model, **kwargs)
        
        # 将权重网络移动到与模型相同的设备上
        self.reward_weight_net = self.reward_weight_net.to(self.model.device)
        # 如果使用了分布式训练，需要包装权重网络
        if self.is_deepspeed_enabled:
            self.reward_weight_net = self.accelerator.prepare_model(self.reward_weight_net)
        elif self.is_fsdp_enabled:
            self.reward_weight_net = self.accelerator.prepare_model(self.reward_weight_net)
        else:
            self.reward_weight_net = self.accelerator.prepare_model(self.reward_weight_net)

        self.reward_weight_optimizer = torch.optim.Adam(self.reward_weight_net.parameters(), lr=1e-4)

    async def compute_correctness_efficiency_reward(self, datalist: List[Dict[str, Any]]) -> List[float]:
        scorer = RealTimeRewardRunner(datalist)
        rewards = scorer.run()
        print("****rewards*****")
        print(rewards)
        correctness_scores = [reward["pass_rate"] for reward in rewards]
        tracemalloc_peak = [reward["avg_tracemalloc_peak"] for reward in rewards]
        cpu_instruction_count = [reward["avg_cpu_instruction_count"] for reward in rewards]
        time_consumed = [reward["avg_time_consumed"] for reward in rewards]
        
        mem_max = max(tracemalloc_peak)
        mem_min = min(tracemalloc_peak)
        mem_scores = []
        for mem in tracemalloc_peak:
            if mem == 0:
                mem_score = 0.0
            elif mem_max - mem_min == 0:
                mem_score = 0.5
            else:
                mem_score = (mem_max - mem) / (mem_max - mem_min)
                mem_score = max(0.0, min(1.0, mem_score))
            mem_scores.append(mem_score)

        time_max = max(time_consumed)
        time_min = min(time_consumed)
        time_scores = []
        for t in time_consumed:
            if t == 0:
                time_score = 1.0
            elif time_max - time_min == 0:
                time_score = 0.5
            else:
                time_score = (time_max - t) / (time_max - time_min)
                time_score = max(0.0, min(1.0, time_score))
            time_scores.append(time_score)

        cpu_max = max(cpu_instruction_count)
        cpu_min = min(cpu_instruction_count)
        cpu_instruction_scores = []
        for cpu in cpu_instruction_count:
            if cpu == 0:
                cpu_score = 1.0
            elif cpu_max - cpu_min == 0:
                cpu_score = 0.5
            else:
                cpu_score = (cpu_max - cpu) / (cpu_max - cpu_min)
                cpu_score = max(0.0, min(1.0, cpu_score))
            cpu_instruction_scores.append(cpu_score)

        return correctness_scores, mem_scores, time_scores, cpu_instruction_scores

    async def compute_comment_reward(self, completions: List[str]) -> List[float]:
        """
        本地计算comment得分，使用CodeCommentScorer计算代码注释质量
        """
        scorer = CodeCommentScorer()
        comment_scores = scorer.score_multiple_completions(completions)
        scorer2 = LLMCommentScorer()
        comment_scores2 = scorer2.score_multiple_completions(completions)
        comment_score = [(comment_scores[i] + comment_scores2[i]/9) / 2 for i in range(len(comment_scores))]
  
        return comment_score

    async def compute_maintainability_reward(self, completions: List[str]) -> List[float]:
        """
        本地计算maintainability得分，使用radon计算MI指数，score=MI/100
        """
        maintainability_scores = []
        for code in completions:
            try:
                mi_score = mi_visit(code, True)
                score = mi_score / 100.0
                score = max(0.0, min(1.0, score))  # 保证分数在[0,1]区间
            except Exception as e:
                score = 0.0
            maintainability_scores.append(score)
        return maintainability_scores

    def reward_func(self, prompts=None, completions=None, completion_ids=None, **kwargs):
        batch_size = len(completions)

        datalist = [{
            "question_id": kwargs['question_id'][i],
            "preCodeSegment": kwargs['preCodeSegment'][i],
            "postCodeSegment": kwargs['postCodeSegment'][i],
            "unittests": kwargs['unittests'][i],
            "response": completions[i]
        } for i in range(batch_size)]

        completions_clean = [extract_code_from_completion(code) for code in completions]

        # 创建异步任务
        async def run_parallel_computations():
            tasks = [
                self.compute_correctness_efficiency_reward(datalist),
                self.compute_comment_reward(completions_clean),
                self.compute_maintainability_reward(completions_clean)
            ]
            results = await asyncio.gather(*tasks)
            return results

        # 运行异步任务
        loop = asyncio.get_event_loop()
        correctness_efficiency_results, comment_scores, maintain_scores = loop.run_until_complete(run_parallel_computations())
        
        correctness_scores, mem_scores, time_scores, cpu_instruction_scores = correctness_efficiency_results
        efficiency_scores = [(mem_scores[j] + time_scores[j]) / 2 for j in range(len(completions))]

        print("****correctness_scores*****")
        print(correctness_scores)
        print("****mem_scores*****")
        print(mem_scores)
        print("****time_scores*****")
        print(time_scores)
        print("****cpu_instruction_scores*****")
        print(cpu_instruction_scores)
        print("*********")
        print("****efficiency_scores*****")
        print(efficiency_scores)
        print("*********")
        print("****comment_scores*****")
        print(comment_scores)
        print("*********")
        print("****maintain_scores*****")
        print(maintain_scores)
        print("*********")
        
        # 将所有维度的分数组合成tensor
        reward_features = torch.tensor([
            [correctness_scores[j], efficiency_scores[j], comment_scores[j], maintain_scores[j]]
            for j in range(len(correctness_scores))
        ], device=self.model.device)
        
        # 使用权重网络计算动态权重
        with torch.set_grad_enabled(self.model.training):
            dynamic_weights = self.reward_weight_net(reward_features)
            
            # 计算加权奖励
            sample_rewards = (reward_features * dynamic_weights).sum(dim=1).tolist()
            
            if self.model.training:
                # 添加权重正则化损失
                weight_variance = dynamic_weights.var(dim=0).mean()
                weight_loss = 0.1 * weight_variance  # 0.1是正则化系数
                
                # 更新权重网络
                self.reward_weight_optimizer.zero_grad()
                weight_loss.backward()
                self.reward_weight_optimizer.step()

        all_rewards = sample_rewards
        
        print("*****dynamic_weights****")
        print(dynamic_weights)
        print("*****all_rewards****")
        print(all_rewards)
        
        return all_rewards

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        保存检查点
        主要功能：
        1. 保存模型权重
        2. 保存优化器和调度器状态
        3. 保存随机数生成器状态
        4. 支持PEFT模型的特殊保存策略
        Args:
            model: 模型实例
            trial: 超参数搜索试验
            metrics: 评估指标
        """
        if isinstance(self.model, PeftModel):
            # 创建检查点文件夹
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            # 获取输出目录
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存模型
            self.save_model(output_dir, _internal_call=True)

            # 保存非LoRA权重
            if not self.args.save_full_model:
                non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
                torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.pth"))

            # 保存优化器和调度器状态
            self._save_optimizer_and_scheduler(output_dir)
           
            # 保存trainer状态
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # 保存权重网络
            weight_net_path = os.path.join(output_dir, "reward_weight_net.pt")
            torch.save(self.reward_weight_net.state_dict(), weight_net_path)

            # 保存权重网络优化器
            optimizer_path = os.path.join(output_dir, "reward_weight_optimizer.pt")
            torch.save(self.reward_weight_optimizer.state_dict(), optimizer_path)

        else:
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        保存模型
        主要功能：
        1. 保存模型权重和配置
        2. 保存tokenizer
        3. 保存训练参数
        4. 支持不同的保存格式（safetensors/pytorch）
        Args:
            output_dir: 输出目录
            state_dict: 模型状态字典
        """
        # 设置输出目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 确定支持的模型类型
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        
        # 保存模型
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 处理特殊保存策略
            if not self.args.save_full_model:
                state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, 'model.pth'))

        # 保存tokenizer
        if self.processing_class is not None:
            os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)
            self.processing_class.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 额外保存权重网络
        if output_dir is None:
            output_dir = self.args.output_dir
        
        weight_net_path = os.path.join(output_dir, "reward_weight_net.pt")
        torch.save(self.reward_weight_net.state_dict(), weight_net_path)
        
        # 保存优化器状态
        optimizer_path = os.path.join(output_dir, "reward_weight_optimizer.pt")
        torch.save(self.reward_weight_optimizer.state_dict(), optimizer_path)
