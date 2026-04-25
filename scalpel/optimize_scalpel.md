# Scalpel 优化计划（不改变行为）

## 目标与约束
- 保留全部现有功能和 BBO 执行模式（接口和默认行为不变）。
- 只做运行时间优化，不改变优化逻辑或结果。
- 固定随机种子时尽量保持可复现。

## 当前设计摘要
- 入口结构：
  - [scalpel_core.py](scalpel_core.py) 实现 MCTS rollout 逻辑和代理模型训练。
  - [scalpel_opt.py](scalpel_opt.py) 把 ScalpelCore 封装为 BaseOptimizer。
  - [original/main.py](original/main.py) 是移植逻辑的参考实现。
- 流程（benchmark 模式）：
  1. suggest() 先生成 warm-start 池（200 个 Latin hypercube 点）。
  2. warm-start 结束后，每轮：重训代理模型 -> MCTS rollout -> 返回新候选点。
  3. observe() 保存结果并更新 best。

## 效率问题（封装开销）
- observe() 中的目标函数重复评估：benchmark 已经给了 fx，但 _raw_score() 又会调用一次底层目标函数来恢复 score，造成昂贵函数的重复计算。
- 训练频率偏高：warm-start 后每次 suggest() 都训练，如果 benchmark 以小批量频繁调用，就会比原版“每 20 点训练一次”更频繁。
- 数组反复拼接：在循环中对 _X_all / _y_score_all 使用 np.concatenate，会导致整体时间复杂度接近 O(n^2)。

## 并行性评估
- 核心 rollout 过程是串行的：每一步依赖上一步的 UCT 选择，单条 rollout 无法安全并行而不改变语义。
- 可并行的安全点：
  - 多起点 rollout（非 rastrigin/ackley/levy 的分支）可以用多个独立 ScalpelCore 并行，然后合并候选点。
  - 模型的 predict 已经批量化，应继续保持批量输入。

## 优化方案（保持行为）
1. 避免 raw-score 重算（跨进程不混用）
   - 在 benchmark 侧把 raw_score 记录并传入 observe()，避免 _raw_score() 再次调用底层函数。
   - raw_score 只能在本进程内使用，不跨进程共享，避免不同进程数据混淆。
   - 若 raw_score 不可得，保留当前 fallback 行为。

2. 训练频率调整到每 10 个点一次
   - 参考原版“每 20 点一训”，改成“每 10 点一训”。
   - 训练触发条件以新数据累计为准，避免对同一批数据反复重训。

3. 降低数据追加成本（已实现）
   - 先用 Python 列表累计，再批量堆叠进 numpy 数组。
   - 避免循环里反复 np.concatenate 造成的 O(n^2)。

4. 小型 numpy 操作批量化（保持 device 位置不变）
   - data_process() 可用结构化数组或广播掩码减少 Python 层循环。
   - most_visit_node() 使用 set 追踪已访问点，减少 np.all 的重复扫描。
   - 仅对 CPU 侧 numpy 数据做优化，不强制把 GPU 上的数据搬回 CPU。

## Rollout 逻辑分析与潜在改动点（暂不实施）
- rollout 的核心循环是串行依赖 UCT 选择结果，单条 rollout 无法直接并行。
- 对于多起点分支（非 rastrigin/ackley/levy），存在并行多个独立 rollout 的空间，但会影响随机性和可复现性。
- rollout 中的 boards / boards_rand 生成和去重在 Python 层循环较多，是潜在优化点。
- model.predict 目前是批量调用，仍应保持批量化。

## 风险与验证
- 任何涉及随机性或训练节奏的改动，都可能影响可复现性，默认必须保持一致。
- 建议做轻量回归验证：
  - 相同 seed、函数、预算：比较 best_fx 的轨迹或最终值是否在容差内。
  - 统计运行时间和目标函数调用次数的变化。
