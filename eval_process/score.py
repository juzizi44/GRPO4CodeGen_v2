import requests
import json
from pathlib import Path
from collections import deque
import re
from tqdm import tqdm
import logging
import datetime
import os
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import math
from typing import List, Set
# Re-run the class definition after reset
import json
import math
import re
import traceback
from typing import List, Dict
from collections import OrderedDict
# from reward.openai_client import *
from openai_client import *
import concurrent.futures
import json
import re
import os
# from reward.api_key import APIKEY
from api_key import APIKEY



def setup_logging(question_id: int, date_str: str, log_root: str = "eval_log/RealTimeRewardRunner_log") -> logging.Logger:
    """设置日志记录器"""
    log_dir = Path(log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日期+时间文件夹
    date_dir = log_dir / date_str
    date_dir.mkdir(exist_ok=True)
    
    # 设置日志文件名
    log_file = date_dir / f"question_{question_id}.log"
    
    # 配置日志记录器
    logger = logging.getLogger(f"question_{question_id}")
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_test_results(logger: logging.Logger, data: Dict[str, Any], question_id: int) -> None:
    """记录测试结果到日志"""
    logger.info(f"{'='*50}")
    logger.info(f"Question ID: {question_id}")
    logger.info(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录完整的测试代码
    logger.info(f"\n完整测试代码:\n{data['source_code']}\n")
    
    # 记录测试用例结果
    unittest_results = data.get("result", {}).get("data", [])
    total_tests = len(unittest_results)
    passed_tests = sum(1 for ut in unittest_results if ut.get("exec_outcome") == "PASSED")
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    logger.info(f"测试用例总数: {total_tests}")
    logger.info(f"通过用例数: {passed_tests}")
    logger.info(f"通过率: {pass_rate:.2%}")
    
    # 记录每个测试用例的详细信息
    logger.info("\n各测试用例详细结果:")
    for idx, ut in enumerate(unittest_results, 1):
        outcome_str = "通过" if ut.get("exec_outcome") == "PASSED" else "不通过"
        logger.info(f"\n测试用例 #{idx}（{outcome_str}）: {json.dumps(ut, ensure_ascii=False, indent=2)}")

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

def log_and_print(msg, file):
    print(msg)
    file.write(msg + '\n')
    file.flush()

class VerboseRetry(Retry):
    def increment(self, *args, **kwargs):
        new_retry = super().increment(*args, **kwargs)
        print("请求超时或失败，正在重试……")
        return new_retry

class RealTimeRewardRunner:
    def __init__(self, 
                 data_list: list,
                 log_root: str = "eval_log/RealTimeRewardRunner_log",
                 execute_code_url: str = "http://172.18.0.1:8899/api/execute_code"):
        self.data_list = data_list
        self.date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
        self.log_dir = self.log_root / self.date_str
        self.log_dir.mkdir(exist_ok=True)
        self.overall_log_path = self.log_dir / "overall.log"
        self.session = requests.Session()
        retries = VerboseRetry(
            total=10,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.rewards = []
        self.execute_code_url = execute_code_url
        # 统计question_id出现次数和索引
        self.qid_count = {}
        self.qid_indices = {}
        for idx, obj in enumerate(self.data_list):
            qid = obj.get("question_id", idx)
            self.qid_count[qid] = self.qid_count.get(qid, 0) + 1
            if qid not in self.qid_indices:
                self.qid_indices[qid] = []
            self.qid_indices[qid].append(idx)

    def process_question(self, question_id: int, overall_log, idx=None) -> dict:
        # 处理重复question_id，生成唯一id
        if self.qid_count.get(question_id, 0) > 1 and idx is not None:
            # 获取当前 idx 在该 question_id 所有索引中的位置
            qid_indices = self.qid_indices.get(question_id, [])
            if idx in qid_indices:
                sub_idx = qid_indices.index(idx) + 1  # 从1开始
            else:
                sub_idx = 1
            unique_qid = f"{question_id}_{sub_idx}"
        else:
            unique_qid = str(question_id)
        logger = setup_logging(unique_qid, self.date_str, str(self.log_root))
        raw_response = ""
        pre_code = ""
        post_code = ""
        unittests = []
        found = False
        for i, obj in enumerate(self.data_list):
            obj_qid = obj.get("question_id", i)
            if str(obj_qid) != str(question_id) or (idx is not None and i != idx):
                continue
            raw_response = obj.get("response", "")
            pre_code = obj.get("preCodeSegment", "")
            post_code = obj.get("postCodeSegment", "")
            for ut in obj.get("unittests", []):
                try:
                    evaluated_input = json.loads(ut["input"])
                except Exception as e:
                    evaluated_input = f"<Invalid input: {e}>"
                unittests.append({
                    "input": evaluated_input,
                    "output": ut["output"]
                })
            found = True
            break
        if not found:
            log_and_print(f"未找到 question_id={unique_qid} 的数据，跳过", overall_log)
            return None
  
        response_code = extract_code_from_completion(raw_response)
    
        full_test_code = "\n\n".join([pre_code, response_code, post_code])
        test_data = {
            "language": "Python 3",
            "source_code": full_test_code,
            "unittests": unittests,
            "block_network": True,
            "stop_on_first_fail": False,
            "use_sanitizer": False,
        }
        try:
            response = self.session.post(
                self.execute_code_url,
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                result = response.json()
                unittest_results = result.get("data", [])
                pass_ut = [ut for ut in unittest_results if ut.get("exec_outcome") == "PASSED"]
                def avg(lst):
                    return sum(lst) / len(lst) if lst else 0
                avg_time_consumed = avg([ut.get("time_consumed", 0) for ut in pass_ut])
                avg_tracemalloc_current = avg([ut.get("tracemalloc_current", 0) for ut in pass_ut])
                avg_tracemalloc_peak = avg([ut.get("tracemalloc_peak", 0) for ut in pass_ut])
                avg_cpu_instruction_count = avg([ut.get("cpu_instruction_count", 0) for ut in pass_ut])
                pass_count = sum(ut.get("exec_outcome") == "PASSED" for ut in unittest_results)
                pass_rate = pass_count / len(unittest_results) if unittest_results else 0
                reward = {
                    "question_id": unique_qid,
                    "pass_rate": pass_rate,
                    "avg_time_consumed": avg_time_consumed,
                    "avg_tracemalloc_peak": avg_tracemalloc_peak,
                    "avg_cpu_instruction_count": avg_cpu_instruction_count
                }
                self.rewards.append(reward)
                log_and_print(
                    f"question_id={unique_qid} reward: {json.dumps(reward, ensure_ascii=False)}",
                    overall_log
                )
                logger.info(f"reward: {json.dumps(reward, ensure_ascii=False)}")
                log_test_results(logger, {
                    "source_code": full_test_code,
                    "unittests": unittests,
                    "result": result
                }, unique_qid)
                return reward
            else:
                error_msg = f"question_id={unique_qid} 错误: HTTP {response.status_code}\n{response.text}"
                log_and_print(error_msg, overall_log)
                return None
        except requests.exceptions.ConnectionError as e:
            error_msg = f"question_id={unique_qid} 连接错误: 无法连接到服务器，请确保服务正在运行"
            log_and_print(error_msg, overall_log)
            return None
        except Exception as e:
            error_msg = f"question_id={unique_qid} 发生错误: {str(e)}"
            log_and_print(error_msg, overall_log)
            return None

    def run(self):
        with open(self.overall_log_path, "a", encoding="utf-8") as overall_log:
            # for idx, obj in enumerate(tqdm(self.data_list, desc="处理进度")):
            for idx, obj in enumerate(self.data_list):
                question_id = obj.get("question_id", idx)
                self.process_question(question_id, overall_log, idx=idx)
        return self.rewards

class CodeCommentScorer:
    def __init__(self):
        self.python_keywords = {
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
            "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
        }


    def _extract_python_comments(self, text: str):
        if text is None:
            return [], 0
        lines = text.splitlines()
        comments = []
        comment_line_count = 0

        # Multi-line comments
        multiline_pattern = r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")"
        for block in re.findall(multiline_pattern, text):
            comments.append(block.strip("'''").strip("\"\"\""))
            comment_line_count += block.count('\n') + 1

        # Single-line and inline comments
        for i in range(len(lines)):
            line = lines[i]
            prev_line = lines[i - 1] if i > 0 else ""
            next_line = lines[i + 1] if i < len(lines) - 1 else ""
            if line.strip().startswith("#"):
                comment_content = line.split('#', 1)[1].strip()
                if not prev_line.strip().startswith("#") and not next_line.strip().startswith("#"):
                    comments.append(comment_content)
                    comment_line_count += 1
                elif not prev_line.strip().startswith("#") and next_line.strip().startswith("#"):
                    combined = comment_content
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith("#"):
                        combined += ' ' + lines[j].split('#', 1)[1].strip()
                        j += 1
                    comments.append(combined)
                    comment_line_count += j - i
                else:
                    comments.append(comment_content)
                    comment_line_count += 1
            elif "#" in line:
                comments.append(line.split('#', 1)[1].strip())
                comment_line_count += 1

        return comments, comment_line_count

    def _extract_identifiers(self, code: str) -> Set[str]:
        if not code:
            return set()

        # 去除注释
        code = re.sub(r'#.*', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)

        # 提取标识符
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        parts = re.findall(r'[a-zA-Z]+', code.replace('_', ' '))
        all_tokens = set(parts + identifiers)

        return {t for t in all_tokens if t not in self.python_keywords}


    def _extract_comment_terms(self, comment_list: List[str]) -> Set[str]:
        terms = set()
        for comment in comment_list:
            for word in comment.split():
                terms.add(word.strip().lower())
        return terms

    def _compute_cic(self, comment_terms: Set[str], identifiers: Set[str]) -> float:
        if not comment_terms and not identifiers:
            return 1.0
        return len(comment_terms & identifiers) / len(comment_terms | identifiers)

    def _compute_cls(self, comment_length: int) -> float:
        if comment_length <= 2:
            return 0
        elif comment_length <= 30:
            return (comment_length - 2) / 28
        return 1.0

    def _compute_average_cls(self, comments: List[str]) -> float:
        if not comments:
            return 0
        return sum(self._compute_cls(len(c)) for c in comments) / len(comments)

    def _compute_ccrs(self, comment_lines: int, total_code: str) -> float:
        total_lines = len(total_code.splitlines())
        return min(comment_lines / total_lines, 1.0) if total_lines else 0.0

    def score_single_completion(self, code: str) -> float:
        try:
            if not code:
                return 0.0
            comments, comment_lines = self._extract_python_comments(code)
            identifiers = self._extract_identifiers(code)
            comment_terms = self._extract_comment_terms(comments)

            cic = self._compute_cic(comment_terms, identifiers)
            cls = self._compute_average_cls(comments)
            ccrs = self._compute_ccrs(comment_lines, code)

            return round((cic + cls + ccrs) / 3, 3)
        except Exception:
            return 0.0

    def score_multiple_completions(self, solutions: List[str]) -> List[float]:
        """
        对多个代码解决方案进行并行评分
        Args:
            solutions: 代码解决方案列表
        Returns:
            List[float]: 评分结果列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(len(solutions), 8)) as executor:
            # 提交所有任务并收集结果
            future_to_solution = {
                executor.submit(self.score_single_completion, solution): i 
                for i, solution in enumerate(solutions)
            }
            
            # 按原始顺序收集结果
            results = [None] * len(solutions)
            for future in concurrent.futures.as_completed(future_to_solution):
                idx = future_to_solution[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Solution {idx} failed with error: {str(e)}")
                    results[idx] = 0.0
                    
            return results


class LLMCommentScorer:
    def __init__(self, api_key = APIKEY, base_url="https://api.openai.com/v1", model="gpt-4.1-mini-2025-04-14", temperature=0.8):
        """
        初始化LLM评分器
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
            temperature: 温度参数
        """
        self.openai_client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
    


    def _extract_final_score(self, response: str) -> tuple[float, bool]:
        """
        从LLM响应中提取final_score
        Args:
            response: LLM的响应文本
        Returns:
            tuple[float, bool]: (分数, 是否成功提取)的元组
        """
        try:
            # 尝试找到 JSON 部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                # 提取 final_score
                final_score = result.get('solution', {}).get('final_score', 0.0)
                return float(final_score), True
        except Exception:
            pass  # 不打印任何信息，转而尝试正则方式

        # 使用正则表达式尝试提取 final_score 字段后的数字（例如 final_score: 0.85）
        match = re.search(r'final_score[^0-9]*(\d(?:\.\d{1,2})?)', response)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 9.0:
                    return score, True
            except Exception:
                pass  # 正则匹配失败则进入最终异常

        # 最终失败才打印
        print("提取分数失败：无法从响应中提取 final_score")
        return 0.0, False

    
    def score_single_completion(self, solution: str) -> float:
        """
        对单个代码解决方案进行评分
        Args:
            solution: 代码解决方案
        Returns:
            float: 评分结果
        """
        for attempt in range(2):  # 最多尝试两次
            try:
                # 构建提示
                prompt = COMMENT_USER_PROMPT.format(
                    solution=solution
                )
                
                # 获取LLM响应
                response = self.openai_client.get_answer(prompt)
                if response == "failed":
                    if attempt == 0:  # 第一次失败，继续尝试
                        continue
                    return 0.0
                    
                # 提取分数
                score, success = self._extract_final_score(response)
                if success:  # 如果成功提取到分数，直接返回
                    return score
                elif attempt == 0:  # 第一次提取失败，继续尝试
                    continue
                # 第二次提取也失败，返回0.0
                return 0.0
                
            except Exception as e:
                print(f"评分过程出错: {str(e)}")
                if attempt == 0:  # 第一次出错，继续尝试
                    continue
                return 0.0
        
        return 0.0  # 如果所有尝试都失败，返回0.0

    def score_multiple_completions(self, solutions: List[str]) -> List[float]:
        """
        对多个代码解决方案进行并行评分
        Args:
            solutions: 代码解决方案列表
        Returns:
            List[float]: 评分结果列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(len(solutions), 8)) as executor:
            # 提交所有任务并收集结果
            future_to_solution = {
                executor.submit(self.score_single_completion, solution): i 
                for i, solution in enumerate(solutions)
            }
            
            # 按原始顺序收集结果
            results = [None] * len(solutions)
            for future in concurrent.futures.as_completed(future_to_solution):
                idx = future_to_solution[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Solution {idx} failed with error: {str(e)}")
                    results[idx] = 0.0
                    
            return results


if __name__ == "__main__":
    # 先在外部读取并筛选数据
    data_list = []
    jsonl_path = "/data/GRPO4CodeGen_v2/dataset/grpo_inference_data/Qwen2.5-Coder-7B-Instruct_testoutput.jsonl"
    # start_question_id = 1
    # end_question_id = 1000
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("question_id")
            # if start_question_id <= int(qid) <= end_question_id:
            data_list.append(obj)
    # 支持外部传参
    import sys
    if len(sys.argv) > 1:
        execute_code_url = sys.argv[1]
    else:
        execute_code_url = "http://172.18.0.1:8899/api/execute_code"
    tester = RealTimeRewardRunner(
        data_list=data_list,
        execute_code_url=execute_code_url
    )
    print("开始评分")
    print(len(data_list))
    rewards = tester.run()
    print(rewards)
    
    # 保存评分结果到指定路径
    save_dir = "/data/GRPO4CodeGen_v2/dataset/score"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Qwen2.5-Coder-7B-Instruct_score.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(rewards, f, ensure_ascii=False, indent=4)
    print(f"评分结果已保存到: {save_path}")


# if __name__ == "__main__":
#     completions = [
#         "# This function adds two numbers\ndef add(a, b):\n    return a + b",
#         "def subtract(x, y):\n    '''Subtract y from x'''\n    return x - y"
#     ]

#     scorer = CodeCommentScorer()
#     print(scorer.score_single_completion(completions[0]))     # 评分单个
#     print(scorer.score_multiple_completions(completions))     # 评分多个



