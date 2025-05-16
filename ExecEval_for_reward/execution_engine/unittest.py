from dataclasses import dataclass, field
import json

from exec_outcome import ExecOutcome
from helper import convert_crlf_to_lf


@dataclass
class Unittest:
    input: str
    output: str
    result: str | None = None
    exec_outcome: ExecOutcome | None = None

    def __post_init__(self):
        self.input = convert_crlf_to_lf(self.input)
        self.output = convert_crlf_to_lf(self.output)

    def update_result(self, result):
        self.result = result

    def update_exec_outcome(self, exec_outcome):
        self.exec_outcome = exec_outcome

    def match_output(self):
        return self.result == self.output


@dataclass
class ExtendedUnittest:
    input: str
    output: list
    result: str | None = None
    exec_outcome: ExecOutcome | None = None
    time_consumed: float | None = None
    peak_memory_consumed: str | None = None
    tracemalloc_current: int | None = None
    tracemalloc_peak: int | None = None
    cpu_instruction_count: int | None = None

    def __post_init__(self):
        # 确保input是字符串
        if isinstance(self.input, dict):
            self.input = json.dumps(self.input) + "\n"
        elif not isinstance(self.input, str):
            self.input = str(self.input)
        if not self.input.endswith("\n"):
            self.input += "\n"
            
        # 处理输出
        if self.output is not None:
            self.output = [convert_crlf_to_lf(o) for o in self.output.copy()]

    def update_time_mem(self, tc, mc):
        self.time_consumed = tc
        self.peak_memory_consumed = mc

    def update_result(self, result):
        self.result = result

    def update_exec_outcome(self, exec_outcome):
        self.exec_outcome = exec_outcome

    def match_output(self, result=None):
        if result is None:
            result = self.result
        return result in self.output

    def json(self):
        _json = self.__dict__.copy()
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.value
        return _json
