#!/bin/bash
# Scalpel 时间对比测试脚本
# 用法: bash test_scalpel_timing.sh [options]
# 
# Options:
#   -f, --func     测试函数 (ackley, rastrigin, rosenbrock, griewank, michalewicz, schwefel, levy)
#   -d, --dims     问题维度 (默认: 10)
#   -b, --budget   评估预算 (默认: 200)
#   -s, --seed     随机种子 (默认: 42)
#   -c, --continuous 使用连续模式
#   -m, --mode     测试模式 (original, wrapped, both)
#
# 示例:
#   bash test_scalpel_timing.sh -f ackley -d 10 -b 200 -s 42
#   bash test_scalpel_timing.sh -f griewank -d 50 -b 1000 -m both

# 默认参数
FUNC="ackley"
DIMS=10
BUDGET=200
SEED=42
CONTINUOUS=""
MODE="both"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--func)
            FUNC="$2"
            shift 2
            ;;
        -d|--dims)
            DIMS="$2"
            shift 2
            ;;
        -b|--budget)
            BUDGET="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -c|--continuous)
            CONTINUOUS="--continuous"
            shift
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Scalpel 时间对比测试"
echo "========================================"
echo "测试函数: $FUNC"
echo "问题维度: $DIMS"
echo "评估预算: $BUDGET"
echo "随机种子: $SEED"
echo "测试模式: $MODE"
echo "连续模式: $CONTINUOUS"
echo "========================================"
echo ""

cd /home/weijun/BBO_Work

python test_scalpel_timing.py \
    --func "$FUNC" \
    --dims "$DIMS" \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --mode "$MODE" \
    $CONTINUOUS
