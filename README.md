# BBO优化项目基准测试框架

> 本项目提供了一套完整的贝叶斯优化（BBO）算法评测框架，统一封装多种主流BBO算法，支持 CEC 合成函数、MuJoCo 强化学习环境、Lasso-Bench 真实任务（DNA/RCV1）以及可选的 Mopta08 工程约束优化，并实现了自动GPU调度、进度存储、后台任务挂起等辅助功能。

---

## ⚠️ 待完成里程碑

> 以下里程碑已按当前仓库结构重新整理。

| 功能 | 状态 | 说明 |
|------|------|------|
| **NAS-Bench 搜索空间** | 🚧 待完成 | 神经网络架构搜索基准测试集 |
| **Lasso-Bench 真实世界BBO（DNA/RCV1）** | ✅ 已完成（第一阶段） | 已集成统一入口 `run_bbo_lasso.py`，默认任务为 DNA 与 RCV1 |
| **Mopta08 约束优化基准** | 🛠 开发中 | 已加入 `run_bbo_mopta.py` 与 `run_bbo.sh --mopta` 调度选项，待补充标准二进制与大规模实验 |

---

## 一、文件结构与核心脚本

### 1.1 整体目录树

```
BBO_Work/
├── baselines/                     # 所有BBO算法的统一封装接口
│   ├── __init__.py               # 统一导出 + 工厂函数 create_optimizer()
│   ├── base.py                    # 基类 BaseOptimizer 和 StatsFuncWrapper
│   ├── bo.py                     # 贝叶斯优化 (GPyTorch, GPU加速)
│   ├── turbo.py                  # TuRBO信任域优化
│   ├── cmaes.py                  # CMA-ES进化算法
│   ├── nevergrad.py              # NeverGrad元算法集成
│   ├── lamcts_opt.py             # LaMCTS树搜索优化器
│   ├── hesbo.py                  # HeSBO高维嵌入子空间优化
│   ├── baxus.py                  # BAxUS自适应扩展子空间优化
│   ├── saasbo.py                 # SAASBO稀疏轴对齐子空间优化
│   └── lamcts/                   # LaMCTS核心模块
│       ├── MCTS.py               # MCTS树核心
│       ├── Node.py               # 树节点��义
│       ├── Classifier.py         # SVM代理分类器
│       └── utils.py              # 拉丁超立方采样等工具
├── scalpel/                       # Scalpel算法实现
│   ├── scalpel_opt.py            # BaseOptimizer包装器
│   ├── scalpel_core.py           # 核心MCTS rollout逻辑
│   └── original/                 # 原始参考实现
├── functions/                     # 测试函数与环境包装
│   ├── test_functions.py         # 7个经典CEC测试函数
│   ├── mujoco_functions.py        # MuJoCo强化学习环境封装
│   ├── lasso_wrapper.py           # LassoBench任务包装（DNA/RCV1）
│   ├── mopta08_wrapper.py         # Mopta08二进制评估包装
│   └── visualize_policy.py        # 策略可视化
├── nas_bench/                     # 🚧 待完成 - NAS搜索空间基准
│   └── ...                        # 神经网络架构搜索任务集
├── lasso_bench/                    # Lasso任务结果目录
│   └── results/
├── mopta/                         # Mopta08结果目录
│   └── results/
├── utils/                         # 辅助工具
│   ├── gpu_scheduler.py           # GPU调度器（多任务自动分配）
│   ├── benchmark_plotter.py       # 结果绘图
│   └── render_mujoco.py           # MuJoCo动画渲染
├── run_bbo.sh                     # 主启动脚本（shell编排）
├── run_bbo_benchmark.py           # CEC测试函数benchmark入口
├── run_bbo_mujoco.py              # MuJoCo环境benchmark入口
├── run_bbo_lasso.py               # Lasso-Bench benchmark入口（DNA/RCV1）
└── run_bbo_mopta.py               # Mopta08 benchmark入口
```

### 1.2 核心脚本职责

| 脚本 | 职责 |
|------|------|
| `run_bbo.sh` | Shell层编排：CEC + MuJoCo + Lasso + Mopta 多基准并行提交，按算法拆分后台worker并串行执行子任务 |
| `run_bbo_benchmark.py` | CEC函数benchmark主入口：管理进度存储、汇总JSON、绘图 |
| `run_bbo_mujoco.py` | MuJoCo benchmark主入口：额外支持动画渲染（`--render`）、多CPU并行评估 |
| `run_bbo_lasso.py` | Lasso-Bench主入口：默认真实任务 DNA/RCV1，支持统一算法评测与汇总 |
| `run_bbo_mopta.py` | Mopta08主入口：对约束违反加入罚项，输出 penalized objective 与约束指标 |
| `baselines/__init__.py` | **统一工厂函数** `create_optimizer(name, func_wrapper, **kwargs)`，隐藏所有算法细节 |
| `utils/gpu_scheduler.py` | 基于`nvidia-smi`的GPU内存调度，自动分配空闲GPU给等待任务 |

---

## 二、BBO算法Baseline与统一接口

### 2.1 支持的算法列表

项目统一封装了 **9个BBO算法**，均实现 `BaseOptimizer` 接口：

| 算法 | 文件 | 核心原理 | GPU加速 | 特点 |
|------|------|---------|---------|------|
| **BO** | `bo.py` | 贝叶斯优化 + GPyTorch GP代理模型 + EI采集函数 | ✅ GPyTorch ExactGP | 数值稳定：保守配置重训 + 回退上次成功模型 |
| **TuRBO** | `turbo.py` | Trust Region Bayesian Optimization（多TR并行） | ✅ GPyTorch | 动态TR大小：成功扩展/失败收缩，支持restart |
| **CMA-ES** | `cmaes.py` | 协方差矩阵自适应进化策略（pycma实现） | ❌ | IPOP重启策略（popsize翻倍）+ Active CMA加速 |
| **NeverGrad** | `nevergrad.py` | Meta-learner集成（NGOpt等） | ❌ | ask/tell接口，自动选择最优参数化 |
| **HeSBO** | `hesbo.py` | 高维嵌入子空间贝叶斯优化（随机哈希投影到低维） | ✅ GPyTorch | 固定box_size=1，ARD核，适用于D≥100的极高维问题 |
| **BAxUS** | `baxus.py` | 自适应扩展子空间贝叶斯优化（嵌套子空间TR） | ✅ BoTorch | pip install baxus，自适应扩展低维空间，高维问题专用 |
| **SAASBO** | `saasbo.py` | 稀疏轴对齐子空间贝叶斯优化（SAAS先验+HMC推断） | ✅ BoTorch | 使用层次化稀疏先验自动识别重要维度，适合D~100-500 |
| **LaMCTS** | `lamcts_opt.py` | MCTS树搜索 + SVM代理 + BO/TuRBO叶节点采样 | ✅ 可选 | **核心创新**：动态构建树 + 自适应Cp + GPU加速SVM |
| **Scalpel** | `scalpel_opt.py` | MCTS rollout + GP回归替代函数 | ✅ | 与LaMCTS同源但用GP替代SVM，每轮约20个候选点 |

### 2.2 HeSBO 特殊说明

HeSBO（Hyperdimensional Embedded Subspace BO）是处理**极高维问题（D≥100）**的专用算法，通过随机哈希投影将高维空间D映射到低维空间d：

**核心参数**（与原始仓库 https://github.com/aminnayebi/HesBO 完全一致）：
- `eff_dim` (d): 低维嵌入维度，推荐固定值 {2, 5, 10, 20}，与D无关
- `box_size`: 搜索空间边界，**固定为 1.0**（不同于REMBO的sqrt(d)）
- `ARD`: 各向异性核，**默认 True**（不同于REMBO默认False）
- `n_doe`: LHS初始采样数量，默认 d+1
- `n_cands`: EI候选点数量，默认 2000
- `hyper_opt_interval`: 超参优化间隔，默认 20

**自动维度选择策略**：
| 原始维度D | 低维维度d |
|-----------|-----------|
| D ≤ 50 | d = max(5, D) |
| 50 < D ≤ 200 | d = max(10, √D) |
| 200 < D ≤ 1000 | d = max(10, √D) |
| D > 1000 | d = 20（原始仓库推荐固定值）|

### 2.3 BAxUS 特殊说明

BAxUS（Bayesian Optimization with Adaptively Expanding Subspaces）通过**嵌套子空间自适应扩展**处理高维问题。pip 安装：`pip install baxus`

**核心参数**：
- `target_dim`: 低维嵌入维度（None = 自动计算，接近 √D）
- `acquisition_function`: 采集函数，'ts'（Thompson采样）或 'ei'（期望改进），默认 'ts'
- `noise`: 噪声标准差，默认 0.0
- `use_ard`: 是否使用 ARD 核，默认 True
- `max_cholesky_size`: Cholesky分解最大尺寸，默认 2000

**算法流程**：
1. 从低维子空间（如 d=2）开始，在信任域内用 Thompson 采样探索
2. 信任域收缩到最小值后，**自适应扩展**子空间维度
3. 将已有观测迁移到扩展后的空间，继续优化
4. 重复直到达到全维度或 budget 耗尽

**注意事项**：
- pip 包版本的 baxus 封装了 EmbeddedTuRBO，内部使用 BoTorch
- 适合维度极高（D ≥ 100）的稀疏优化问题
- GPU 加速通过 botorch 实现

### 2.4 SAASBO 特殊说明

SAASBO（Sparse Axis-Aligned Subspace BO）通过**层次化稀疏先验**处理高维问题，核心组件在 BoTorch 中。

**核心参数**：
- `warmup_steps`: HMC/NUTS 预热步数，默认 256（不宜低于此值）
- `num_samples`: HMC 样本数，默认 128
- `thinning`: 采样间隔，默认 16
- `noise_var`: 观测噪声方差，默认 1e-6
- `batch_size`: 批量大小，默认 1（**强烈建议保持为1**，因 HMC 推断开销随数据量立方增长）

**SAAS 先验机制**：
- 全局收缩参数 τ ~ Half-Cauchy(β)
- 逆长度尺度 ρ_d ~ Half-Cauchy(τ)，d=1,...,D
- HMC/NUTS 推断自动识别重要维度，稀疏化不重要维度

**性能特点**：
- 适合 D ~ 100-500 维度，evaluation budget ~ 500 以内
- GPU 加速效果显著（推荐使用 GPU）
- 推断开销随数据量立方增长，不适合超长 budget

### 2.5 统一接口设计

所有优化器继承 `BaseOptimizer`，提供两个调用模式：

```python
class BaseOptimizer(ABC):
    def suggest(self, n_suggestions: int) -> np.ndarray:
        """建议新采样点（n_suggestions个）"""

    def observe(self, x: np.ndarray, fx: np.ndarray):
        """反馈外部评估结果"""

    def optimize(self, call_budget: int, batch_size: int):
        """自包含闭环优化（算法内部管理循环）"""

    def reset(self):
        """重置状态"""
```

**工厂函数统一入口**：

```python
from baselines import create_optimizer

# 一个函数创建所有算法
optimizer = create_optimizer(
    name='bo',           # 'bo' | 'turbo' | 'cmaes' | 'nevergrad' | 'hesbo' | 'baxus' | 'saasbo' | 'lamcts' | 'scalpel'
    func_wrapper=wrapper,
    gpu_id=0,
    batch_size=10,
    device=torch.device('cuda:0'),
)
x = optimizer.suggest(10)     # 建议10个点（外部评估）
optimizer.observe(x, fx)      # 反馈结果
```

**函数包装器统一接口**（`FuncWrapper`）：

```python
class FuncWrapper:
    def __init__(self, func, is_minimizing=True):
        self.func = func
        self.lb, self.ub, self.dims  # 边界和维度
        self.sign = 1.0 if is_minimizing else -1.0  # 最大最小化统一

    def __call__(self, x):
        return self.sign * self.func(x)  # 内部统一为最小化问题
```

---

## 三、Benchmark实现与存储方式

### 3.1 四套Benchmark体系

**CEC合成测试函数** (`run_bbo_benchmark.py`)：

- 7个经典函数：`ackley`, `rastrigin`, `rosenbrock`, `griewank`, `michalewicz`, `schwefel`, `levy`
- 3个维度档位：20D / 50D / 100D
- 预算：2000 / 7000 / 12000

**MuJoCo强化学习环境** (`run_bbo_mujoco.py`)：

- 6个环境：`swimmer(16D)`, `hopper(33D)`, `halfcheetah(102D)`, `walker2d(102D)`, `ant(216D)`, `humanoid(6392D)`
- 预算按维度自适应：<50D → 3000, <200D → 5000, ≥200D → 8000

**Lasso-Bench真实任务** (`run_bbo_lasso.py`)：

- 当前任务：`dna(180D)`, `rcv1(19959D)`
- 目标：最小化验证损失，附加记录 `mspe` 与 `fscore`（若可用）
- 说明：已移除 `synt_high` 与 `synt_hard`，避免与 CEC 合成函数评测重复

**Mopta08约束优化** (`run_bbo_mopta.py`)：

- 任务：`mopta08(124D)`，区间约束 `[0,1]^124`
- 目标：最小化 penalized objective
- 罚项：`f_penalized = f_obj + lambda * sum(max(g_i, 0))`

### 3.2 统一存储逻辑

每个 `{算法}/{函数}` 组合产生两个JSON文件：

| 文件 | 内容 | 存储路径 |
|------|------|---------|
| `progress.json` | **覆盖式**存储（每10%更新一次）：`{step, best_fx, elapsed_time}` 数组 | `results/{func}_{dims}d/{algo}/progress.json` |
| `final_result.json` | 最终结果：最优解 `best_x`、最优值 `best_fx`、总时间、`progress_history` | `results/{func}_{dims}d/{algo}/final_result.json` |
| `summary.json` | **合并式**汇总：所有算法×函数的最优值 | `results/summary.json` |

关键设计：**覆盖式 progress + 合并式 summary**

- `progress.json` 每次只保存最新进度（重复运行会覆盖）
- `summary.json` 读取已有内容后合并新结果（不覆盖已有算法数据）

### 3.3 MuJoCo额外存储

```
mujoco/results/{env_name}/{algorithm}/
├── progress.json       # 进度
├── final_result.json   # 最终结果
└── 动画文件（可选，通过 --render 生成）
```

### 3.4 NAS-Bench 搜索空间基准 🚧 待完成

> 神经网络架构搜索（Neural Architecture Search）基准测试集，将搜索空间建模为高维离散/混合优化问题。

**目标**：集成 NAS-Bench-101 / NAS-Bench-201 / NDS 等标准搜索空间，每个任务对应一个预训练好的网络评估数据库，将架构选择问题转化为黑盒优化问题。

**预期特性**：

- 搜索空间维度：~50D（NAS-Bench-101）至 ~200D（NDS搜索空间）
- 评估方式：查表（lookup），无需真实训练，速度极快
- 支持离散操作选择 + 跳连边选择等混合变量
- 可能需要对离散变量进行连续松弛或独热编码

**参考资源**：

- [NAS-Bench-101](https://arxiv.org/abs/1902.09635)
- [NAS-Bench-201](https://arxiv.org/abs/2001.00326)
- [NDS (Network Design Space)](https://arxiv.org/abs/1811.05997)

### 3.5 Lasso-Bench 真实世界BBO ✅ 第一阶段完成

> [Lasso-Bench](https://github.com/Anthony9316/Lasso-Bench) 是一个针对高维稀疏优化问题的BBO评测基准，涵盖真实数据集上的回归/分类任务。

**已完成范围**：

- 已在统一脚本中接入真实任务 `DNA` 与 `RCV1`
- 已纳入 `run_bbo.sh --lasso` 调度体系
- 已统一结果存储与 `summary.json` 合并逻辑
- 已移除 `lasso-synth` 的 `synt_high/synt_hard` 默认任务，避免与 CEC 重复

**下一步扩展建议**：

- 增加 `breast_cancer / diabetes / leukemia` 可选开关
- 为超高维 `RCV1` 增加分档预算策略（例如 warmup + 主优化）
- 增加更细的失败重试与GPU回退策略

**参考资源**：

- [Lasso-Bench GitHub](https://github.com/Anthony9316/Lasso-Bench)
- 对应的论文：[Lasso-Bench: A High-Dimensional Hyperparameter Optimization Benchmark](https://arxiv.org/)

### 3.6 Mopta08 基准规划（已接入入口）

> Mopta08 是经典工程约束优化任务，适合评测高维连续空间下的约束处理能力。

**当前集成形态**：

- 独立脚本 `run_bbo_mopta.py`
- 函数封装 `functions/mopta08_wrapper.py`
- 总调度入口 `run_bbo.sh --mopta`

**实施要点**：

1. 统一使用 `[0,1]^124` 搜索空间。
2. 通过 Mopta08 可执行文件评估目标与约束。
3. 采用罚函数将约束违反折算为单目标优化。
4. 在 `progress.json/final_result.json` 中同时记录 `best_fx`、约束违反数量与最大违反值。

**依赖与数据准备**：

- 需要提供 Mopta08 二进制可执行文件。
- 支持通过 `--mopta-executable` 或环境变量 `MOPTA08_EXECUTABLE` 指定路径。

**参考资源**：

- TuRBO/高维BO社区中常用的 Mopta08 评测协议

---

## 四、辅助功能实现

### 4.1 存储逻辑

`BenchmarkResult` 类管理存储周期：

```python
class BenchmarkResult:
    def add_result(self, step, best_fx, elapsed_time):
        # 每10%的budget保存一次 progress.json
        save_interval = max(total_budget // 10, 1)
        # 每10%的budget打印一次日志（到stdout）
        log_interval = max(total_budget // 10, 1)
        # 完成后保存 final_result.json
```

### 4.2 脚本挂起逻辑

使用Shell的 `nohup` + 后台 `&` + `sh -c "..."` 实现任务链挂起：

```bash
nohup sh -c "
    echo 'Execute func 20D...'
    python run_bbo_benchmark.py --dims 20 --budget 2000 ... > output/20d/func.out 2>&1

    echo 'Execute func 50D...'
    python run_bbo_benchmark.py --dims 50 --budget 7000 ... > output/50d/func.out 2>&1

    echo 'Execute func 100D...'
    python run_bbo_benchmark.py --dims 100 --budget 12000 ... > output/100d/func.out 2>&1
" > output/master_log_gpu${GPU_ID}_${FUNC}.txt 2>&1 &
```

- **nohup**：防止终端断开导致进程退出
- **后台&**：立即返回终端，用户可继续提交其他任务
- **任务链**：`20D → 50D → 100D` 顺序执行，共享同一GPU
- **stdout/stderr重定向**：`.out` 文件捕获输出日志，`.txt` 文件捕获主日志

### 4.3 GPU加速逻辑

**三层GPU管理架构**：

1. **GPU调度器** (`utils/gpu_scheduler.py`)
   - 通过 `nvidia-smi --query-gpu` 查询所有GPU的内存使用情况
   - `wait_for_gpu(min_memory_mb)` 轮询等待，直到有足够空闲内存的GPU可用
   - 按空闲内存排序，选择最空闲的GPU
   - 支持超时控制和检查间隔配置

2. **算法层GPU配置**
   - 每个算法接收 `gpu_id` 和 `device` 参数，在构造时调用 `torch.cuda.set_device()`
   - **BO/TuRBO**：GPyTorch ExactGP在GPU上训练和预测
   - **LaMCTS**：SVM分类器可选GPU加速
   - **Scalpel**：内部使用GPU张量运算
   - 所有算法实现**保险措施**：先 `set_device` 再构造任何GPU对象

3. **内存清理**
   - 每10次函数评估调用一次 `torch.cuda.empty_cache()`
   - 使用 `gc.collect()` 配合清理
   - GPyTorch预测失败时尝试递增的 `jitter` 值（1e-4 → 1e-3 → 1e-2 → 1e-1）

**使用示例**：

```bash
# 自动选择GPU（���存≥8GB）
python run_bbo_benchmark.py --gpu-auto ...

# 指定GPU 3
python run_bbo_benchmark.py --gpu-id 3 ...

# 无GPU模式
python run_bbo_benchmark.py --no-gpu ...
```

---

## 五、数据流总览

```
Shell (run_bbo.sh)
    └─ nohup + sh -c 链式任务
           ├─ GPU调度器 → 分配合适GPU
           ├─ create_optimizer() → 工厂函数
           │      └─ 算法类 (BO/TuRBO/CMAES/NG/LaMCTS/Scalpel)
           │             └─ func_wrapper (FuncWrapper)
           │                    └─ 测试函数 / MuJoCo环境
           ├─ BenchmarkResult → 周期性 save progress.json
           └─ 完成后 save final_result.json + 更新 summary.json
```

---

## 六、快速开始

### 6.1 CEC函数Benchmark

```bash
# 完整评测（后台挂起）
bash run_bbo.sh

# 单函数单算法测试
python run_bbo_benchmark.py \
    --functions ackley rastrigin \
    --algorithms bo lamcts scalpel \
    --dims 50 \
    --budget 5000 \
    --batch 10 \
    --gpu-id 0
```

### 6.2 MuJoCo Benchmark

```bash
# 完整评测
python run_bbo_mujoco.py \
    --algorithms bo turbo cmaes lamcts scalpel \
    --environments swimmer hopper ant \
    --budget 3000 \
    --gpu-id 1 \
    --plot
```

### 6.3 Lasso-Bench Benchmark

```bash
# 默认真实任务（DNA + RCV1）
python run_bbo_lasso.py \
    --algorithms bo turbo cmaes hesbo baxus saasbo scalpel \
    --benchmarks dna rcv1 \
    --batch 1
```

### 6.4 Mopta08 Benchmark

```bash
# 运行前需要提供 Mopta08 可执行文件路径
python run_bbo_mopta.py \
    --algorithms turbo cmaes baxus \
    --budget 1000 --batch 1 \
    --mopta-executable /path/to/mopta08_elf64.bin
```

### 6.5 结果查看

```bash
# 查看汇总结果
cat cec_functions/results/summary.json
cat lasso_bench/results/summary.json
cat mopta/results/summary.json

# 绘制收敛曲线
python run_bbo_benchmark.py --plot

# 查看GPU状态
python utils/gpu_scheduler.py
```
