"""噪声模型组合器：按规则在执行过程中插入噪声通道。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set


@dataclass
class NoiseRule:
    channel: object
    after_gates: Optional[Set[str]] = None


@dataclass
class NoiseModel:
    """噪声模型：由一组 NoiseRule 组成。"""

    rules: List[NoiseRule] = field(default_factory=list)

    def add_channel(self, channel, after_gates: Optional[Sequence[str]] = None) -> "NoiseModel":
        """
        添加噪声通道规则。

        参数:
            channel: NoiseChannel 实例
            after_gates: 仅在这些 gate type 之后施加；None 表示每个门后都施加
        """
        gate_set = None if after_gates is None else {str(g) for g in after_gates}
        self.rules.append(NoiseRule(channel=channel, after_gates=gate_set))
        return self

    def _match_rule(self, rule: NoiseRule, gate_type: Optional[str]) -> bool:
        if rule.after_gates is None:
            return True
        if gate_type is None:
            return False
        return gate_type in rule.after_gates

    def apply(self, rho, n_qubits: int, backend, gate_type: Optional[str] = None):
        """
        对密度矩阵应用所有匹配规则。

        参数:
            rho: 当前密度矩阵（后端张量）
            n_qubits: 总量子比特数
            backend: 后端实例
            gate_type: 当前门类型（用于规则过滤）
        """
        out = rho
        for rule in self.rules:
            if not self._match_rule(rule, gate_type):
                continue
            kraus = rule.channel.kraus_operators(n_qubits, backend)
            acc = backend.zeros(out.shape)
            for k in kraus:
                acc = acc + backend.matmul(backend.matmul(k, out), backend.dagger(k))
            out = acc
        return out

    def __len__(self) -> int:
        return len(self.rules)
