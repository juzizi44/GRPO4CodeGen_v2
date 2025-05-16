import sys
import json
import tracemalloc
import gc
import os
from pathlib import Path
import cProfile
import time
from collections import deque
import functools

sys.path.append(str(Path(__file__).parent))

from unittest import ExtendedUnittest
from exec_outcome import ExecOutcome
from helper import convert_crlf_to_lf
from resource_limit import ResourceLimits

import subprocess
import shlex
import signal
from threading import Timer

def retry_on_error(max_retries=3, delay=0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_error(max_retries=3, delay=0.1)
def get_collector():
    from cirron import Collector
    return Collector()

# 添加TreeNode、ListNode数据结构类
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_to_listnode(arr):
    """将列表转换为链表"""
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def list_to_treenode(arr):
    """将列表转换为树（层序）"""
    if not arr:
        return None
    root = TreeNode(arr[0])
    queue = deque([root])
    i = 1
    while queue and i < len(arr):
        node = queue.popleft()
        # 处理左子节点
        if i < len(arr):
            if arr[i] is not None:  # 修改这里，检查是否为None
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
        # 处理右子节点
        if i < len(arr):
            if arr[i] is not None:  # 修改这里，检查是否为None
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
    return root

def is_same_tree(p, q):
    """比较两棵树是否相同"""
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

def is_same_list(p, q):
    """比较两个链表是否相同"""
    while p and q:
        if p.val != q.val:
            return False
        p = p.next
        q = q.next
    return p is None and q is None

def preprocess_output(output_list):
    """预处理输出，检查是否包含树或链表数据结构"""
    processed_output = []
    for out in output_list:
        # 检查是否包含类型标记的对象
        if isinstance(out, dict) and "type" in out and "value" in out:
            data_type = out["type"]
            value = out["value"]
            
            if data_type == "tree_node" and isinstance(value, list):
                # 转换为树结构
                processed_output.append(list_to_treenode(value))
            elif data_type == "list_node" and isinstance(value, list):
                # 转换为链表结构
                processed_output.append(list_to_listnode(value))
            elif data_type == "general ":
                # 普通类型，直接使用value
                processed_output.append(value)
            else:
                # 不支持的类型或格式，保持原样
                processed_output.append(value)
        # 兼容没有类型标记的老数据 - 但不自动转换为树
        elif isinstance(out, list) and len(out) > 0:
            # 不再自动将所有列表都视为树，保持列表类型
            processed_output.append(out)
        else:
            processed_output.append(out)
    return processed_output

def run_test_case(tc_dict, executor, run_uid, run_gid, exec_env, code_store_dir, limits_dict, output_validator):
    # 增加总超时限制
    class TotalTimeout(Exception):
        pass
    def timeout_handler(signum, frame):
        raise TotalTimeout()
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 总超时5秒
    try:
        WARMUP = 1  # 热启动次数
        N = 1      # 统计次数
        total_current = 0
        total_peak = 0
        total_time = 0.0
        total_instr = 0
        last_result = None
        last_exec_outcome = None
        
        # 处理可能是树或链表的输出
        if "output" in tc_dict:
            tc_dict["output"] = preprocess_output(tc_dict["output"])
        
        tc = ExtendedUnittest(**tc_dict)
        result, exec_outcome = None, None
        outs, errs = None, None
        limits = ResourceLimits(**limits_dict)

        # 确保输入是字符串
        input_str = tc.input
        if isinstance(input_str, dict):
            input_str = json.dumps(input_str) + "\n"
        elif not isinstance(input_str, str):
            input_str = str(input_str)
        if not input_str.endswith("\n"):
            input_str += "\n"

        # 热启动阶段
        for _ in range(WARMUP):
            tracemalloc.start()
            tracemalloc.clear_traces()
            tracemalloc.reset_peak()
            gc.collect()
            with get_collector() as col:
                with subprocess.Popen(
                    shlex.split(executor),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    user=run_uid,
                    group=run_gid,
                    cwd=code_store_dir,
                    env=exec_env,
                    start_new_session=True,
                ) as child_process:
                    def handler():
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                    timer = Timer(limits.cpu, handler)
                    timer.start()
                    try:
                        outs, errs = child_process.communicate(
                            input_str.encode("ascii"), timeout=limits.cpu
                        )
                        timer.cancel()
                    except subprocess.TimeoutExpired:
                        exec_outcome = ExecOutcome.TIME_LIMIT_EXCEEDED
                        result = "Time Limit Exceeded"
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                        outs, errs = None, None
                    except Exception as e:
                        timer.cancel()
                        exec_outcome = ExecOutcome.RUNTIME_ERROR
                        result = str(e)
                    finally:
                        timer.cancel()
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                        try:
                            child_process.communicate(timeout=1)
                        except:
                            pass
                        child_process.wait()
            tracemalloc.stop()
            gc.collect()
        # 正式统计阶段
        for _ in range(N):
            tracemalloc.start()
            tracemalloc.clear_traces()
            tracemalloc.reset_peak()
            gc.collect()
            start_time = time.perf_counter()
            with get_collector() as col:
                with subprocess.Popen(
                    shlex.split(executor),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    user=run_uid,
                    group=run_gid,
                    cwd=code_store_dir,
                    env=exec_env,
                    start_new_session=True,
                ) as child_process:
                    def handler():
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                    timer = Timer(limits.cpu, handler)
                    timer.start()
                    try:
                        outs, errs = child_process.communicate(
                            input_str.encode("ascii"), timeout=limits.cpu
                        )
                        timer.cancel()
                    except subprocess.TimeoutExpired:
                        exec_outcome = ExecOutcome.TIME_LIMIT_EXCEEDED
                        result = "Time Limit Exceeded"
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                        outs, errs = None, None
                    except Exception as e:
                        timer.cancel()
                        exec_outcome = ExecOutcome.RUNTIME_ERROR
                        result = str(e)
                    finally:
                        timer.cancel()
                        if child_process.poll() is None:
                            os.killpg(os.getpgid(child_process.pid), signal.SIGKILL)
                            child_process.kill()
                        try:
                            child_process.communicate(timeout=1)
                        except:
                            pass
                        child_process.wait()
                    if exec_outcome is None:
                        if child_process.returncode == 0 and outs is not None:
                            result = outs.decode(errors="ignore").strip()
                            exec_outcome = (
                                ExecOutcome.PASSED
                                if any(
                                    output_validator(output, result)
                                    for output in tc.output
                                )
                                else ExecOutcome.WRONG_ANSWER
                            )
                        elif errs is not None and len(errs) != 0:
                            exec_outcome = ExecOutcome.RUNTIME_ERROR
                            errs = errs.decode(errors="ignore")
                            if (
                                "out of memory" in errs.lower()
                                or "bad_alloc" in errs.lower()
                                or "bad alloc" in errs.lower()
                                or "memoryerror" in errs.lower()
                            ):
                                exec_outcome = ExecOutcome.MEMORY_LIMIT_EXCEEDED
                            if child_process.returncode > 0:
                                result = errs
                            else:
                                result = f"Process exited with code {-child_process.returncode}, {signal.strsignal(-child_process.returncode)} stderr: {errs}"
                        else:
                            exec_outcome = ExecOutcome.MEMORY_LIMIT_EXCEEDED
                            if outs is not None:
                                result = outs.decode(errors="ignore").strip()
                            elif errs is not None:
                                result = errs.decode(errors="ignore").strip()
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # 统计指令数
            instr_count = col.counters.instruction_count
            total_instr += instr_count
            total_current += current
            total_peak += peak
            total_time += elapsed
            last_result = result
            last_exec_outcome = exec_outcome
        avg_current = int(total_current / N)
        avg_peak = int(total_peak / N)
        avg_time = total_time / N
        avg_instr = int(total_instr / N)
        
        # 创建结果字典
        result_dict = {
            "input": tc.input,
            "output": tc.output,
            "result": last_result,
            "exec_outcome": last_exec_outcome.value if last_exec_outcome else None,
            "time_consumed": avg_time,
            "peak_memory_consumed": None,
            "tracemalloc_current": avg_current,
            "tracemalloc_peak": avg_peak,
            "cpu_instruction_count": avg_instr
        }
        
        # 自定义JSON序列化
        class CompactListEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, TreeNode):
                    return {"type": "tree_node", "value": self._tree_to_list(obj)}
                elif isinstance(obj, ListNode):
                    return {"type": "list_node", "value": self._list_to_array(obj)}
                return super().default(obj)
            
            def _tree_to_list(self, root):
                if not root:
                    return []
                result = []
                queue = deque([root])
                while queue:
                    node = queue.popleft()
                    if node:
                        result.append(node.val)
                        queue.append(node.left)
                        queue.append(node.right)
                    else:
                        result.append(None)
                # 移除尾部的None
                while result and result[-1] is None:
                    result.pop()
                return result
            
            def _list_to_array(self, head):
                result = []
                current = head
                while current:
                    result.append(current.val)
                    current = current.next
                return result
        
        # 输出JSON结果 - 设置ensure_ascii=False保证Unicode正常显示，separators设置使列表显示紧凑
        print(json.dumps(result_dict, cls=CompactListEncoder, ensure_ascii=False, separators=(',', ':')))
    except TotalTimeout:
        # 超时处理
        result_dict = {
            "input": tc_dict.get("input", ""),
            "output": tc_dict.get("output", []),
            "result": "Timeout (total > 5s)",
            "exec_outcome": ExecOutcome.TIME_LIMIT_EXCEEDED.value,
            "time_consumed": 5.0,
            "peak_memory_consumed": None,
            "tracemalloc_current": None,
            "tracemalloc_peak": None,
            "cpu_instruction_count": None
        }
        print(json.dumps(result_dict, ensure_ascii=False, separators=(',', ':')))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

if __name__ == '__main__':
    # 参数1: 测试用例json字符串
    # 参数2: executor命令
    # 参数3: run_uid
    # 参数4: run_gid
    # 参数5: code_store_dir
    # 参数6: limits_dict_json
    # 参数7: exec_env_json
    # 参数8: output_validator_json（可选，暂不支持复杂序列化）
    tc_dict = json.loads(sys.argv[1])
    executor = sys.argv[2]
    run_uid = int(sys.argv[3])
    run_gid = int(sys.argv[4])
    code_store_dir = sys.argv[5]
    limits_dict = json.loads(sys.argv[6])
    exec_env = json.loads(sys.argv[7])
    # output_validator 需要在主进程内定义并传递，这里自定义验证器支持树和链表结构
    def output_validator(expected, actual):
        # 如果实际结果是字符串，尝试解析为JSON
        parsed_actual = actual
        if isinstance(actual, str):
            try:
                # 尝试解析字符串形式的列表
                if actual.startswith('[') and actual.endswith(']'):
                    # 将字符串中的None替换为null以便JSON解析
                    actual = actual.replace('None', 'null')
                    parsed_actual = json.loads(actual)
                else:
                    parsed_actual = json.loads(actual)
            except:
                # 如果解析失败，保持原样
                parsed_actual = actual
        
        # 1. 如果预期是TreeNode，使用is_same_tree比较
        if isinstance(expected, TreeNode):
            # 如果实际结果是列表，尝试将其视为树
            if isinstance(parsed_actual, list):
                actual_tree = list_to_treenode(parsed_actual)
                return is_same_tree(expected, actual_tree)
            # 如果实际结果是带类型标记的字典
            elif isinstance(parsed_actual, dict) and parsed_actual.get("type") == "tree_node":
                actual_tree = list_to_treenode(parsed_actual.get("value", []))
                return is_same_tree(expected, actual_tree)
            # 如果实际结果已经是TreeNode
            elif isinstance(parsed_actual, TreeNode):
                return is_same_tree(expected, parsed_actual)
            return False  # 无法比较
        
        # 2. 如果预期是ListNode，使用is_same_list比较
        if isinstance(expected, ListNode):
            # 如果实际结果是列表，尝试将其视为链表
            if isinstance(parsed_actual, list):
                actual_list = list_to_listnode(parsed_actual)
                return is_same_list(expected, actual_list)
            # 如果实际结果是带类型标记的字典
            elif isinstance(parsed_actual, dict) and parsed_actual.get("type") == "list_node":
                actual_list = list_to_listnode(parsed_actual.get("value", []))
                return is_same_list(expected, actual_list)
            # 如果实际结果已经是ListNode
            elif isinstance(parsed_actual, ListNode):
                return is_same_list(expected, parsed_actual)
            return False  # 无法比较
        
        # 3. 如果expected是带类型标记的字典
        if isinstance(expected, dict) and "type" in expected and "value" in expected:
            expected_type = expected["type"]
            expected_value = expected["value"]
            
            # 3.1 如果预期是树节点，转换后比较
            if expected_type == "tree_node":
                expected_tree = list_to_treenode(expected_value)
                # 递归调用，将转换后的TreeNode作为预期
                return output_validator(expected_tree, actual)
            
            # 3.2 如果预期是链表，转换后比较
            elif expected_type == "list_node":
                expected_list = list_to_listnode(expected_value)
                # 递归调用，将转换后的ListNode作为预期
                return output_validator(expected_list, actual)
            
            # 3.3 如果预期是普通类型，直接比较值
            elif expected_type == "general ":
                return output_validator(expected_value, actual)
        
        # 4. 直接的值比较逻辑
        
        # 4.1 列表比较 - 递归比较每个元素
        if isinstance(expected, list) and isinstance(parsed_actual, list):
            if len(expected) != len(parsed_actual):
                return False
            return all(output_validator(e, a) for e, a in zip(expected, parsed_actual))
        
        # 4.2 数值比较
        if isinstance(expected, (int, float)) and isinstance(parsed_actual, (int, float, str)):
            try:
                return float(expected) == float(parsed_actual)
            except (ValueError, TypeError):
                pass
        
        # 4.3 布尔值比较
        if isinstance(expected, bool) or isinstance(parsed_actual, bool):
            try:
                return bool(expected) == bool(parsed_actual)
            except:
                pass
        
        # 4.4 None值比较
        if expected is None or parsed_actual is None:
            return expected is None and parsed_actual is None
        
        # 4.5 最后尝试字符串比较
        return str(expected).strip() == str(parsed_actual).strip()
    
    run_test_case(tc_dict, executor, run_uid, run_gid, exec_env, code_store_dir, limits_dict, output_validator) 