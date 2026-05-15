# QAS File Inventory

本文档记录当前 `nexq.algorithms.qas` 的扁平正交目录规划与兼容文件定位。

## 主线文件

| 文件 | 定位 | 备注 |
|---|---|---|
| `__init__.py` | Layer 4，对外 API | 只 re-export 稳定入口 |
| `search_env.py` | Layer 3，`NoisyQASEnv` | 面向逐步加门搜索的环境封装 |
| `noise_adaptive_search.py` | Layer 3，`NoiseAdaptiveQAS` | 两阶段：候选生成，再统一评分排序 |
| `evaluator.py` | Layer 2，架构评分器 | 调用四组 metrics，输出 `ArchitectureScore` |
| `reward.py` | Layer 2，奖励聚合 | 只组合已计算分数，不直接 import 具体指标函数 |
| `metrics_expressibility.py` | Layer 1，无噪表达能力 | `kl_haar`、`mmd_relative` 等 |
| `metrics_noise.py` | Layer 1，噪声鲁棒性 | `ion_trap_error_budget_proxy`、含噪 KL/MMD、noise sensitivity |
| `metrics_trainability.py` | Layer 1，可训练性 | `structure_proxy` |
| `metrics_hardware.py` | Layer 1，硬件效率 | `native_depth_twoq_efficiency` |
| `_types.py` | Layer 0，共享类型 | `ArchitectureSpec`、`ArchitectureScore`、`SearchConfig` 等 |
| `_utils.py` | Layer 0，共享工具 | gate counting、parameter counting、rank helper 等 |
| `candidates.py` | Layer 0，候选库新入口 | 兼容转发旧 `architecture_candidates.py` |
| `ion_trap_noise_config.py` | Layer 0，离子阱噪声配置 | 从 markdown 参数构造 `NoiseModel` |

## 保留兼容文件

| 文件 | 当前定位 | 后续建议 |
|---|---|---|
| `expressibility.py` | 无噪表达能力旧实现来源 | 长期可收敛到 `metrics_expressibility.py` |
| `noise_robustness.py` | 噪声鲁棒性旧实现来源 | 长期可收敛到 `metrics_noise.py` |
| `multi_objective_reward.py` | 旧多目标 reward 和 RL wrapper | 保留给 PPR-DQL / legacy demos |
| `qas_evaluation.py` | 旧 circuit-level 综合评估器 | `evaluator.py` 是新架构主入口 |
| `architecture_candidates.py` | 旧候选库实现 | `candidates.py` 是新导入入口 |
| `CRLQAS.py` | legacy RL generator / baseline | 保留 |
| `PPO_RB.py` | legacy RL generator / baseline | 可生成候选线路后交给 evaluator 评分 |
| `PPR_DQL.py` | legacy RL generator / baseline | 已有 noise-adaptive reward 接口，可优先整合 |

## 正交性规则

- `metrics_expressibility.py` 只放无噪表达能力。
- `metrics_noise.py` 才放理想-含噪差异、含噪 KL/MMD、离子阱 error budget 和 noise sensitivity。
- `reward.py` 不直接 import metrics，只聚合 `MetricGroupScore`。
- `evaluator.py` 是唯一负责把四组 metrics 合成为 `ArchitectureScore` 的主线模块。
- `CRLQAS.py`、`PPO_RB.py`、`PPR_DQL.py` 作为候选生成器或 baseline，不反向依赖评分核心。
