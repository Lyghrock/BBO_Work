# Scalpel 优化计划

## 修改记录（2026-05-05）

### 实施的修复

#### 1. 修复 test_functions.py — 返回 (f, score) tuple
**文件**: `functions/test_functions.py`

原版 `scalpel/original/main.py` 中的 benchmark 函数返回 `(f, score)` tuple，其中 score 用于 MCTS 训练（越大越好）。

| 函数 | 原 score 计算 | 现已修复 |
|------|-------------|---------|
| Ackley | `100 / (f + 0.01)` | ✓ |
| Rastrigin | `-f` | ✓ |
| Rosenbrock | `100 / (f / (dims * 100) + 0.01)` | ✓ |
| Griewank | `10 / (f / dims + 0.001)` | ✓ |
| Michalewicz | `f` (已是负值，越大越好) | ✓ |
| Schwefel | `-f / 100` | ✓ |
| Levy | `-f` | ✓ |

#### 2. 修复 FuncWrapper — 存储原始 score
**文件**: `run_bbo_benchmark.py`

新增 `get_last_scores()` 方法，返回最近一次批量调用的所有原始 score。

#### 3. 重构 ScalpelOptimizer.optimize() — 每20点批量训练
**文件**: `scalpel/scalpel_opt.py`

**关键改动**:
- 移除 `_TRAIN_INTERVAL` 和 pending 队列机制
- 改为与原版 `main.py` 完全一致的每 20 点批量训练 + rollout 模式
- 批量评估减少接口开销

**原流程**:
```python
# 每次 evaluate 1 点，逐点处理
while total_calls < budget:
    x = optimizer.suggest(1)
    fx = evaluate(x)
    optimizer.observe(x, fx)
    if len(pending) >= 10:  # interval 机制
        train_model()
```

**新流程**:
```python
# 批量 evaluate，每 20 点训练一次
while total_calls < budget:
    # 训练模型
    model = trainer.train(all_X, all_y)
    # MCTS rollout 生成 ~20 点
    new_X = optimizer.rollout(...)
    # 批量 evaluate
    for xi in new_X:
        fx = evaluate(xi)
    # 追加数据
    all_X = concatenate([all_X, new_X])
    all_y = concatenate([all_y, new_scores])
```

#### 4. 添加 BaseOptimizer.observe() scores 参数
**文件**: `baselines/base.py`

新增可选参数 `scores`，允许传入原始 score。

---

## 时间对比测试

### 测试脚本

**Python 脚本**: `test_scalpel_timing.py`
**Shell 脚本**: `test_scalpel_timing.sh`

### 使用方法

```bash
# 基本用法
bash test_scalpel_timing.sh -f ackley -d 10 -b 200 -s 42

# 测试特定函数
bash test_scalpel_timing.sh -f griewank -d 50 -b 1000 -s 42

# 只测试原版
bash test_scalpel_timing.sh -f ackley -d 10 -m original

# 只测试包装版
bash test_scalpel_timing.sh -f ackley -d 10 -m wrapped

# 使用连续模式
bash test_scalpel_timing.sh -f ackley -d 10 -c
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f, --func` | 测试函数 | ackley |
| `-d, --dims` | 问题维度 | 10 |
| `-b, --budget` | 评估预算 | 200 |
| `-s, --seed` | 随机种子 | 42 |
| `-c, --continuous` | 使用连续模式 | False |
| `-m, --mode` | 测试模式 (original/wrapped/both) | both |

### 测试内容

1. **原版 fluxer + main.py**: 直接运行原始实现
2. **包装版 scalpel_core + scalpel_opt.py**: 运行包装后的实现
3. **对比指标**:
   - 总运行时间
   - 最终 best_f(x)
   - 评估次数

### 注意事项

- 已禁用 wandb 和其他日志工具
- 两边使用相同的随机种子
- 两边使用相同的 GPU 配置
- 测试包括初始化 200 个点的开销

---

## 待验证项

- [ ] 原版和包装版的 best_fx 是否在合理范围内（由于随机性，可能不完全一致）
- [ ] 两版的运行时间差异是否在可接受范围内
- [ ] 批量 evaluate 是否正确工作
- [ ] score 计算是否正确传递给 MCTS
