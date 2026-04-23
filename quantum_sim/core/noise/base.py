"""噪声模型抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class NoiseChannel(ABC):
    """Kraus 通道抽象：E(rho) = sum_k K_k rho K_k^dagger。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称。"""

    @abstractmethod
    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        """返回作用在 n_qubits 全系统上的 Kraus 算符列表。"""
