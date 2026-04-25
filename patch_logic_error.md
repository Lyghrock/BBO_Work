> 审查时间：2026-04-21
> 审查范围：`run_bbo_*.py`、`baselines/`、`functions/`、`cec_functions/`、`lasso_bench/`、`mujoco/`、`run_bbo.sh`、`utils/wandb_sync.py`

> **状态说明：** ✅ 已修复 | ⚠️ 保留（已确认合理） | 🔧 待修复（用户决定修复） | ⏸️ 跳过（用户决定不修复）

---

## 优先级速览

| 优先级 | 数量 | 典型问题 | 状态 |
|---|---|---|---|
| **P0 — 崩溃 / 数据损坏** | 5 | Mopta 异常处理、测试函数返回类型、MuJoCo 重复创建进程、subprocess 无超时、lamcts sign 不一致 | ✅ 全部已修复 |
| **P1 — 逻辑错误 / 结果不正确** | 9 | Scalpel 缺失参数、seed 未生效、cmaes 重复计数、lamcts 树重建 | ✅ 全部已修复（P1-4 跳过） |
| **P2 — 性能 / 正确性** | 6 | GPU device 未传递、HESBO 哈希不可逆、MuJoCo 环境池化、污染全局 RNG | ⏸️ 跳过（用户决定） |
| **P3 — 小问题** | 12 | 死代码、注释错误、命名不一致 | ⏸️ 跳过（低优先级） |

---

## P0 — 必须立即修复（✅ 全部已修复）

### P0-1 `run_bbo_mopta.py` — 异常时 `current_metrics` 未定义导致 `NameError` ✅

**原问题：** `try/except` 吞掉异常后，`current_metrics` 未被赋值，后续访问 `current_metrics.get('objective')` 抛出 `NameError`。

**修复：** 在 try 块之前初始化 `current_metrics = None`，try 块中赋值；后续访问加 None 检查。

---

### P0-2 `functions/test_functions.py` — 所有 `__call__` 返回二元组而非标量 ✅

**原问题：** 7 个测试函数均返回 `(objective, fitness_ratio)` 二元组，与其余 benchmark 返回标量的约定不符。

**修复：** 移除 `fitness_ratio` 返回值，所有函数仅返回标量目标值。

---

### P0-3 `mujoco_functions.py` — 每次 `__call__` 都重新创建环境 ✅

**原问题：** 每次函数评估都调用 `gym.make()`，MuJoCo 模型编译耗时 1–2 秒/次，40k 评估额外增加 10–20 小时。

**修复：** 六个环境类均添加 `self._env = None` 缓存，`_create_env()` 复用现有 env，`reset()` 正确释放。

---

### P0-4 `mopta08_wrapper.py` — subprocess 无超时 ✅

**原问题：** `subprocess.run()` 无 `timeout`，binary 卡死会无限阻塞整个进程。

**修复：** 添加 `timeout=300`（5 分钟），超时时抛出 `TimeoutError`。

---

### P0-5 `baselines/lamcts_opt.py` — `observe()` 的 sign 处理与 MCTS 树期望不一致 ✅

**原问题：** MCTS 树内部使用最大值格式（`curt_best_value = -inf`，越大越好），但 `observe()` 对最小化场景使用 `self.sign = 1`（不做取反），导致 UCT 决策被污染。

**修复：** 最小化时将 `fx` 取负存入 MCTS 树，保证与树内部最大值约定一致。

---

## P1 — 逻辑错误（✅ 全部已修复，P1-4 跳过）

### P1-1 Scalpel 在三个脚本的 `get_optimizer_overrides` 中均无覆盖参数 ✅

**原问题：** `'scalpel'` 列入 `ALGORITHMS`，但三个脚本的 `get_optimizer_overrides` 均无对应分支；`run_bbo_lasso.py` 和 `run_bbo_mopta.py` 也没有 `scalpel_kwargs`。

**修复：** 在三个脚本的 `get_optimizer_overrides` 中添加 `elif algo == 'scalpel': pass`；在 `run_bbo_lasso.py` 和 `run_bbo_mopta.py` 的 `create_optimizer` 调用中添加 `scalpel_kwargs`（传入 `func_name` 和 `use_continuous`）。

---

### P1-2 `run_bbo_benchmark.py` — `seed` 参数被接收但从未应用 ✅

**原问题：** `run_single_algorithm` 接受 `seed` 参数但从不调用 `np.random.seed()` 或 `torch.manual_seed()`。

**修复：** 在 `run_bbo.sh` 增加 `--seed N` 开关（不指定则随机）；四个 benchmark 脚本的 `run_single`/`run_single_algorithm`/`run_benchmark` 均增加 `seed` 参数，并在开头执行 `random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)`。

---

### P1-3 `baselines/cmaes.py` — `call_count` 被重复计数 ✅

**原问题：** `optimize()` 中 `func_wrapper(xi)` 已累加 `func_wrapper.call_count`，随后 `self.call_count += 1` 又累加一次，导致 `call_count` 是实际调用数的两倍。

**修复：** 移除 `optimize()` 中的 `self.call_count += 1`，仅保留 `total_calls += 1`。

---

### P1-4 `baselines/lamcts_opt.py` — 每轮迭代从头重建 MCTS 树 ⏸️

**原问题：** 每轮 `suggest()` 都调用 `self._build_mcts_tree()`，丢弃历史树结构。

**修复：** 用户决定跳过，不修改。

---

### P1-5 `run_bbo_benchmark.py` — unreachable `pass` 死代码 ✅

**原问题：** `if optimizer is None: pass` 之后直接落入下面的 `try` 块，永远不会触发随机搜索（随机搜索实际在第 378 行的 `else` 分支）。

**修复：** 删除该 dead `pass` 代码块和 `if optimizer is None` 检查。

---

### P1-6 `run_bbo_lasso.py` — `gpu_id` 传给 `create_optimizer` ⚠️

**原问题：** `gpu_id` 传入签名可能不兼容。

**修复：** 用户决定保留，GPU 分配逻辑合理。

---

### P1-7 Lasso 和 MuJoCo 的 `result_manager.add_result` 缺少最终保存保护 ✅

**原问题：** `run_bbo_lasso.py` 的 `add_result` 缺少 `or step >= self.total_budget`，loop 中途退出时最后一批结果不持久化。

**修复：** 在 `run_bbo_lasso.py` 的保存条件中添加 `or step >= self.total_budget`（MuJoCo 已有该保护）。

---

### P1-8 `run_bbo_benchmark.py` — `total_evaluations` 始终等于 budget 而非实际调用数 ✅

**原问题：** `save_final` 使用 `self._get_total_budget()`（即传入的 `budget`），而非实际 `total_calls`。

**修复：** `save_final` 新增 `total_evaluations` 参数，调用方传入实际 `total_calls`。

---

## P2 — 性能 / 正确性（⏸️ 全部跳过）

| # | 文件 | 问题 | 跳过原因 |
|---|---|---|---|
| P2-1 | `bo/turbo/baxus/Classifier.py` | GPU device 未传递给 GPyTorch 模型 | 用户决定跳过 |
| P2-2 | `hesbo.py` | 加性随机哈希不可逆 | 已知算法限制 |
| P2-3 | `mujoco_functions.py` | `np.random.seed` 污染全局 RNG | 用户决定跳过 |
| P2-4 | `mujoco_functions.py` | Swimmer 无归一化；Hopper 硬编码统计量 | 用户决定跳过 |

---

## P3 — 小问题（✅ 部分已修复）

### P3-M1 冗余目录 ✅

**修复内容：**
- 删除 `cec_functions/supplement_output/` 目录
- 删除 `lasso_bench/results/synt_high/` 目录（合成 benchmark 已移除）
- 更新 `run_bbo.sh` 删除 `mkdir -p cec_functions/output`（该目录不存在）

### P3-M3 过时注释 ✅

**修复内容：**
- `functions/mujoco_functions.py`：删除废弃的可视化示例代码块（696-714行）
- `functions/visualize_policy.py`：删除不存在的 `Lunarlanding` 导入及注释
- `run_bbo_lasso.py`：删除 `--noise` 参数及相关代码（合成 benchmark 已移除）
- `functions/lasso_wrapper.py`：删除 `noise` 参数的过时说明
- `baselines/lamcts_opt.py`：精简"合成函数"相关注释

### P3-M5 RL 环境创建评估 ✅

**结论：** `run_bbo_mujoco.py` 中每个 algorithm 循环都创建新的 `MuJoCoFuncWrapper`，各算法完全隔离，无需修改。当前 `render_mode='rgb_array'` 不产生实际渲染，无渲染开销。

---

## 已修复汇总（2026-04-21）

| 修复项 | 文件 | 内容 |
|---|---|---|
| Wandb 异步缓冲丢失 | `wandb_sync.py` | `commit=True`；`try/finally`；`WANDB_SYNC_DIR` |
| `run_bbo.sh` 无 `--no-wandb` 开关 | `run_bbo.sh` | `--no-wandb`；`--seed N` |
| 删除多余 output 目录 | 项目根目录 | `cec_functions/output`、`lasso_bench/output`、`mujoco/output`、`mopta/output` |
| Mopta `current_metrics` NameError | `run_bbo_mopta.py` | 初始化 None + None 检查 |
| 测试函数返回二元组 | `functions/test_functions.py` | 全部改为返回标量 |
| MuJoCo 环境每次重建 | `functions/mujoco_functions.py` | 六个类全部加 env 缓存复用 |
| Mopta subprocess 无超时 | `functions/mopta08_wrapper.py` | `timeout=300` |
| lamcts observe sign 不一致 | `baselines/lamcts_opt.py` | 最小化时取负存入 MCTS 树 |
| Scalpel 无超参数覆盖 | 三个 benchmark 脚本 | 添加 elif 分支 + scalpel_kwargs |
| seed 未生效 | `run_bbo.sh` + 四个脚本 | `--seed N` 开关 + 三处设置随机种子 |
| cmaes call_count 重复计数 | `baselines/cmaes.py` | 移除 optimize() 中多余的 `call_count += 1` |
| dead pass 死代码 | `run_bbo_benchmark.py` | 删除 unreachable 代码块 |
| Lasso 最终保存保护 | `run_bbo_lasso.py` | 添加 `or step >= self.total_budget` |
| total_evaluations 报告错误 | `run_bbo_benchmark.py` | 新增参数传入实际 `total_calls` |
| P3-M1 冗余目录 | `cec_functions/supplement_output` 等 | 删除 2 个冗余目录 + 更新 run_bbo.sh |
| P3-M3 过时注释 | `mujoco_functions.py` 等 | 删除废弃可视化代码、`noise` 参数、`Lunarlanding` 导入 |
| P3-M5 RL 环境创建 | `run_bbo_mujoco.py` | 确认正确，无需修改 |

