"""nexq.algorithms.qas

Quantum architecture search and state-synthesis utilities.
"""

from .expressibility import KL_Haar_relative, MMD_relative
from .expressibility_noise import (
    KL_Haar_noisy,
    MMD_noisy,
    noise_sensitivity,
    NoiseSensitivityResult,
    comparative_expressibility,
    expressibility_score,
)
from .multi_objective_reward import (
    RewardWeights,
    MultiObjectiveReward,
    ExpressibilityScore,
    TrainabilityScore,
    NoiseRobustnessScore,
    HardwareEfficiencyScore,
    QASRewardWrapper,
)
from .qas_evaluation import (
    QASEvaluator,
    EvaluationResult,
    quick_evaluate,
    noise_aware_reward,
)
from .state_qas import StateQASConfig, state_to_circuit

__all__ = [
    # 原始表达能力
    "KL_Haar_relative",
    "MMD_relative",
    # 含噪表达能力
    "KL_Haar_noisy",
    "MMD_noisy",
    "noise_sensitivity",
    "NoiseSensitivityResult",
    "comparative_expressibility",
    "expressibility_score",
    # 多目标奖励
    "RewardWeights",
    "MultiObjectiveReward",
    "ExpressibilityScore",
    "TrainabilityScore",
    "NoiseRobustnessScore",
    "HardwareEfficiencyScore",
    "QASRewardWrapper",
    # 评估器
    "QASEvaluator",
    "EvaluationResult",
    "quick_evaluate",
    "noise_aware_reward",
    # QAS
    "state_to_circuit",
    "StateQASConfig",
]

if StateQASConfig is not None and state_to_circuit is not None:
    __all__.extend(["state_to_circuit", "StateQASConfig"])
