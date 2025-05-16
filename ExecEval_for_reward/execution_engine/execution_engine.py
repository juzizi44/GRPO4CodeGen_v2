import os
import shlex
import signal
import subprocess
import tracemalloc
import gc
from pathlib import Path
from threading import Timer, Thread
from unittest import ExtendedUnittest
import json
from collections import deque
import multiprocessing
from functools import partial
import time
import concurrent.futures

import gmpy2
from code_store import CodeStore
from config import Config
from exec_outcome import ExecOutcome
from helper import convert_crlf_to_lf
from job import JobData, LanguageError
from prlimit import get_prlimit_str
from resource_limit import ResourceLimits
from runtime import Runtime
from seccomp_filter import make_filter
from settings import JavaClassNotFoundError


class TreeNodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TreeNode):
            return {"type": "tree_node", "value": self.tree_to_list(obj)}
        elif isinstance(obj, ListNode):
            return {"type": "list_node", "value": self.list_to_array(obj)}
        elif isinstance(obj, (int, float, bool, str, list, dict)) or obj is None:
            # 可以选择性地为普通类型添加标记，但一般不需要
            # 只有在需要特别区分时才添加
            # return {"type": "general ", "value": obj}
            return obj  # 大多数情况下，普通类型直接返回
        return super().default(obj)
    
    def tree_to_list(self, root):
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
    
    def list_to_array(self, head):
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result


class CompilationError(Exception):
    """Shows the compilation error message

    Args:
        Exception command list[str]: command to compile
        message str: compilation error message
    """

    def __init__(self, command, message: subprocess.CalledProcessError):
        self.command = command
        self.message = message
        super().__init__(f"command: {self.command} produced: {self.message.stderr}")


# def init_validate_outputs():
#     _token_set = {"yes", "no", "true", "false"}
#     PRECISION = gmpy2.mpfr(1e-12, 129)

#     def validate_outputs(output1: str, output2: str) -> bool:
#         # for space sensitive problems stripped string should match
#         def validate_lines(lines1, lines2):
#             validate_line = lambda lines: lines[0].strip() == lines[1].strip()
#             if len(lines1) != len(lines2):
#                 return False
#             return all(map(validate_line, zip(lines1, lines2)))

#         if validate_lines(output1.strip().split("\n"), output2.strip().split("\n")):
#             return True

#         # lines didn't work so token matching
#         tokens1, tokens2 = output1.strip().split(), output2.strip().split()
#         if len(tokens1) != len(tokens2):
#             return False

#         for tok1, tok2 in zip(tokens1, tokens2):
#             try:
#                 num1, num2 = gmpy2.mpfr(tok1, 129), gmpy2.mpfr(tok2, 129)
#                 if abs(num1 - num2) > PRECISION:
#                     return False
#             except ValueError:
#                 if tok1.lower() in _token_set:
#                     tok1 = tok1.lower()
#                 if tok2.lower() in _token_set:
#                     tok2 = tok2.lower()
#                 if tok1 != tok2:
#                     return False

#         return True

#     return validate_outputs


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
    
def init_validate_outputs():
    _token_set = {"yes", "no", "true", "false"}
    PRECISION = gmpy2.mpfr(1e-12, 129)

    def validate_outputs(actual, expected) -> bool:
        # === TreeNode 结构比较 ===
        if isinstance(expected, TreeNode):
            return is_same_tree(actual, expected)

        # === ListNode 结构比较 ===
        if isinstance(expected, ListNode):
            return is_same_list(actual, expected)

        # === 字符串默认输出比较 ===
        if isinstance(expected, str) and isinstance(actual, str):
            def validate_lines(lines1, lines2):
                validate_line = lambda lines: lines[0].strip() == lines[1].strip()
                if len(lines1) != len(lines2):
                    return False
                return all(map(validate_line, zip(lines1, lines2)))

            if validate_lines(actual.strip().split("\n"), expected.strip().split("\n")):
                return True

            # fallback: float or token matching
            tokens1, tokens2 = actual.strip().split(), expected.strip().split()
            if len(tokens1) != len(tokens2):
                return False

            for tok1, tok2 in zip(tokens1, tokens2):
                try:
                    num1, num2 = gmpy2.mpfr(tok1, 129), gmpy2.mpfr(tok2, 129)
                    if abs(num1 - num2) > PRECISION:
                        return False
                except ValueError:
                    if tok1.lower() in _token_set:
                        tok1 = tok1.lower()
                    if tok2.lower() in _token_set:
                        tok2 = tok2.lower()
                    if tok1 != tok2:
                        return False
            return True

        # Fallback strict equality
        return actual == expected

    return validate_outputs


class MonitorThread(Thread):
    def __init__(self, proc):
        Thread.__init__(self)
        self.total_time = None
        self.peak_memory = None
        self.proc = proc
        self.clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def run(self):
        while self.proc.poll() is None:
            # print(self.total_time, self.peak_memory)
            try:
                # print(f"/proc/{self.proc.pid}/stat", os.path.exists(f"/proc/{self.proc.pid}/stat"))
                # print(f"/proc/{self.proc.pid}/status", os.path.exists(f"/proc/{self.proc.pid}/status"))
                # print(self.total_time, self.peak_memory)
                with open(f"/proc/{self.proc.pid}/stat") as pid_stat:
                    vals = pid_stat.read().split()
                    self.total_time = (
                        float(vals[13])
                        + float(vals[14])
                        + float(vals[15])
                        + float(vals[16])
                    ) / self.clk_tck  # adding user time and sys time, also childs utime, stime
                with open(f"/proc/{self.proc.pid}/status") as pid_status:
                    vm_peak_line = [l for l in pid_status if l.startswith("VmPeak:")]
                    if len(vm_peak_line) == 0:
                        continue
                    vm_peak_line = vm_peak_line[0]
                    self.peak_memory = vm_peak_line.split(":")[-1].strip()
            except (FileNotFoundError, ProcessLookupError):
                pass


def run_single_testcase(args_dict):
    """
    独立的测试用例执行函数
    args_dict: 包含执行所需的所有参数的字典
    """
    tc = args_dict['tc']
    executor = args_dict['executor']
    limits_dict = args_dict['limits_dict']
    exec_env = args_dict['exec_env']
    run_uid = args_dict['run_uid']
    run_gid = args_dict['run_gid']
    source_dir = args_dict['source_dir']
    logger = args_dict['logger']

    # 确保输入是字符串
    if isinstance(tc.input, dict):
        tc_dict = tc.__dict__.copy()
        tc_dict["input"] = json.dumps(tc.input) + "\n"
    else:
        tc_dict = tc.__dict__.copy()
        if not isinstance(tc_dict["input"], str):
            tc_dict["input"] = str(tc_dict["input"])
        if not tc_dict["input"].endswith("\n"):
            tc_dict["input"] += "\n"

    # 组装参数
    args = [
        'python3',
        str(Path(__file__).parent / 'testcase_runner.py'),
        json.dumps(tc_dict, cls=TreeNodeEncoder),
        executor,
        str(run_uid),
        str(run_gid),
        str(source_dir),
        json.dumps(limits_dict),
        json.dumps(exec_env)
    ]
    
    # 调用子进程
    try:
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"Testcase runner failed: {proc.stderr}")
            return None
            
        tc_result = json.loads(proc.stdout.strip())
        # 兼容 ExtendedUnittest
        result = ExtendedUnittest(
            input=tc_result.get("input", ""),
            output=tc_result.get("output", []),
            result=tc_result.get("result", None),
            exec_outcome=ExecOutcome(tc_result.get("exec_outcome")) if tc_result.get("exec_outcome") else None,
            time_consumed=tc_result.get("time_consumed", None),
            peak_memory_consumed=tc_result.get("peak_memory_consumed", None)
        )
        # 额外字段可按需添加
        result.tracemalloc_current = tc_result.get("tracemalloc_current", None)
        result.tracemalloc_peak = tc_result.get("tracemalloc_peak", None)
        result.cpu_instruction_count = tc_result.get("cpu_instruction_count", None)
        return result
    except Exception as e:
        logger.error(f"Test case execution failed: {e}")
        return None

class ExecutionEngine:
    def __init__(
        self,
        cfg: Config,
        limits_by_lang: dict[str, ResourceLimits],
        run_ids: tuple[int, int],
        logger,
    ) -> None:
        self.code_store = CodeStore(cfg.code_store, run_ids)
        self.supported_languages: dict[str, Runtime] = dict()
        self.output_validator = init_validate_outputs()
        for lang, sup_cfg in cfg.supported_languages.items():
            self.supported_languages[lang] = Runtime(sup_cfg)

        self.run_uid = run_ids[1]
        self.run_gid = run_ids[0]
        self.socket_filter = make_filter(["socket"])
        self.logger = logger
        self.limits_by_lang = limits_by_lang
        self.memory_stats = []

        self.exec_env = os.environ.copy()
        self.exec_env["GOCACHE"] = str(self.code_store._source_dir.resolve())

    def start(self):
        self.code_store.create()

    def stop(self):
        self.code_store.destroy()

    def _compile(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            shlex.split(command),
            user=self.run_uid,
            group=self.run_gid,
            capture_output=True,
            cwd=self.code_store._source_dir,
            env=self.exec_env,
            timeout=60,
        )

    def _get_executable_after_compile(
        self,
        lang: str,
        source_file: Path,
        cmd: str | None = None,
        flags: str | None = None,
    ) -> tuple[str | Path, bool]:
        if not self.supported_languages[lang].is_compiled_language:
            return source_file, False

        compile_str, executable = self.supported_languages[lang].compile(
            source_file, cmd, flags
        )
        try:
            cp = self._compile(compile_str)
        except subprocess.TimeoutExpired as e:
            return f"{e}", True

        if cp.returncode == 0:
            return executable, False

        return cp.stderr.decode(errors="ignore"), True

    def get_executor(
        self, job: JobData, limits: ResourceLimits
    ) -> tuple[str | Path | LanguageError, int]:
        language = job.language
        if language is None:
            return LanguageError("Language must be selected to execute a code."), -1

        if language not in self.supported_languages:
            return LanguageError(f"Support for {language} is not implemented."), -1

        source_code = convert_crlf_to_lf(job.source_code)

        if self.supported_languages[language].has_sanitizer and job.use_sanitizer:
            source_code = self.supported_languages[language].sanitize(source_code)

        source_path = self.supported_languages[language].get_file_path(source_code)
        if isinstance(source_path, JavaClassNotFoundError):
            return source_path, -1
        source_path = self.code_store.write_source_code(source_code, source_path)

        executable, err = self._get_executable_after_compile(
            language, source_path, cmd=job.compile_cmd, flags=job.compile_flags
        )

        if err:
            return executable, -1

        execute_flags = job.execute_flags

        if self.supported_languages[language].extend_mem_for_vm:
            if limits._as != -1:
                if execute_flags is None:
                    execute_flags = f" -{self.supported_languages[language].extend_mem_flag_name}{limits._as} "
                else:
                    execute_flags += f" -{self.supported_languages[language].extend_mem_flag_name}{limits._as} "

        return (
            self.supported_languages[language].execute(
                executable, cmd=job.execute_cmd, flags=execute_flags
            ),
            self.supported_languages[language].timelimit_factor,
        )

    def check_output_match(self, job: JobData) -> list[ExtendedUnittest]:
        limits = job.limits
        if limits is None:
            limits = ResourceLimits()
            limits.update(self.limits_by_lang[job.language])

        executor, timelimit_factor = self.get_executor(job, limits)
        if timelimit_factor == -1:
            result = executor
            if isinstance(executor, (LanguageError, JavaClassNotFoundError)):
                result = executor.msg
            elif not isinstance(result, str):
                result = "Some bug in ExecEval, please do report."
            return [
                ExtendedUnittest(
                    input="",
                    output=[],
                    result=result,
                    exec_outcome=ExecOutcome.COMPILATION_ERROR,
                )
            ]

        if (
            self.supported_languages[job.language].extend_mem_for_vm
            and limits._as != -1
        ):
            limits._as += 2**30
            
        executor = f"{get_prlimit_str(limits)} {executor}"
        self.logger.debug(
            f"Execute with gid={self.run_gid}, uid={self.run_uid}: {executor}"
        )

        # 准备并行执行所需的参数
        limits_dict = limits.__dict__.copy()
        exec_env = self.exec_env.copy()
        
        # 为每个测试用例准备参数字典
        test_args = []
        for tc in job.unittests:
            args_dict = {
                'tc': tc,
                'executor': executor,
                'limits_dict': limits_dict,
                'exec_env': exec_env,
                'run_uid': self.run_uid,
                'run_gid': self.run_gid,
                'source_dir': self.code_store._source_dir.resolve(),
                'logger': self.logger
            }
            test_args.append(args_dict)
        
        # 进程池异常保护和整体超时机制
        max_retries = 3
        timeout_seconds = 10  # 你可以根据需要调整超时时间
        for attempt in range(max_retries):
            try:
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor_f:
                        future = executor_f.submit(pool.map, run_single_testcase, test_args)
                        results = future.result(timeout=timeout_seconds)
                new_test_cases = [r for r in results if r is not None]
                return new_test_cases
            except concurrent.futures.TimeoutError:
                self.logger.error(f"进程池执行超时（第{attempt+1}次，{timeout_seconds}秒），重试……")
            except Exception as e:
                self.logger.error(f"进程池执行异常（第{attempt+1}次）：{e}")
            time.sleep(1)
        self.logger.error("进程池多次重试后仍然失败，返回空列表。")
        return []


if __name__ == "__main__":

    class Test:
        file: str
        lang: str

        def __init__(self, file, lang):
            self.file = file
            self.lang = lang

    tests = [
        Test("execution_engine/test_codes/test.c", "GNU C"),
        Test("execution_engine/test_codes/test.cpp", "GNU C++17"),
        Test("execution_engine/test_codes/test.go", "Go"),
        Test("execution_engine/test_codes/test.js", "Node js"),
        Test("execution_engine/test_codes/test.php", "PHP"),
        Test("execution_engine/test_codes/test.py", "PyPy 3"),
        Test("execution_engine/test_codes/test.py", "Python 3"),
        Test("execution_engine/test_codes/test.rb", "Ruby"),
        Test("execution_engine/test_codes/test.rs", "Rust"),
        Test("execution_engine/test_codes/test.java", "Java 7"),
        Test("execution_engine/test_codes/test.kt", "Kotlin"),
    ]

    unittests = [
        ExtendedUnittest("1 1", ["2"]),
        ExtendedUnittest("1 3", ["4"]),
        ExtendedUnittest("-1 2", ["1"]),
        ExtendedUnittest("122 2", ["124"]),
    ]

    from config import load_config
    from job import JobData
    from resource_limit import ResourceLimits

    cfg = load_config(Path("execution_engine/config.yaml"))

    ce = ExecutionEngine(cfg)

    for t in tests:
        with open(t.file) as f:
            s = f.read()
        updated_unittests = ce.check_output_match(
            JobData(
                language=t.lang,
                source_code=s,
                unittests=unittests,
                limits=ResourceLimits(),
            )
        )

        print(f"{t.lang} got: \n", updated_unittests)
