#!/bin/bash
#===========================================================
# test_scalpel_timing.sh — SLURM timing test for Scalpel
#
# 使用方式（与 run_slurm.sh 完全一致的工作流）:
#   1. rsync 同步 BBO_Work
#   2. 手动 mv test_scalpel_timing.sh ../
#   3. sbatch test_scalpel_timing.sh
#
# 接受环境变量参数:
#   TEST_FUNC      测试函数 (默认: ackley)
#   TEST_DIMS      问题维度 (默认: 10)
#   TEST_BUDGET    评估预算 (默认: 200)
#   TEST_SEED      随机种子 (默认: 42)
#   TEST_MODE      测试模式: original, wrapped, both (默认: both)
#   TEST_CONTINUOUS 连续模式: 1=启用 (默认: 0)
#
# 示例:
#   sbatch test_scalpel_timing.sh
#   TEST_FUNC=griewank TEST_DIMS=50 sbatch test_scalpel_timing.sh
#   TEST_FUNC=rastrigin TEST_BUDGET=1000 TEST_SEED=123 sbatch test_scalpel_timing.sh
#===========================================================

#---------------- SBATCH directives ----------------
#SBATCH --job-name=scalpel_timing
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/weijun/assigned_runs
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e

#---------------- 工作目录（硬编码，与 run_slurm.sh 一致）----------------
SCRIPT_DIR="/home/weijun/assigned_runs"
WORKSPACE_DIR="$SCRIPT_DIR/BBO_Work"

#---------------- 默认参数 ----------------
TEST_FUNC="${TEST_FUNC:-ackley}"
TEST_DIMS="${TEST_DIMS:-10}"
TEST_BUDGET="${TEST_BUDGET:-200}"
TEST_SEED="${TEST_SEED:-42}"
TEST_MODE="${TEST_MODE:-both}"
TEST_CONTINUOUS="${TEST_CONTINUOUS:-0}"

mkdir -p "$WORKSPACE_DIR/cec_functions/results"

#---------------- conda-free Python setup（与 run_slurm.sh 完全一致）----------------
CONDA_ROOT=""
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_ROOT="${CONDA_PREFIX%/envs/*}"
elif [[ -d "/home/weijun/miniconda3" ]]; then
    CONDA_ROOT="/home/weijun/miniconda3"
elif [[ -d "/home/weijun/anaconda3" ]]; then
    CONDA_ROOT="/home/weijun/anaconda3"
elif [[ -d "$HOME/miniconda3" ]]; then
    CONDA_ROOT="$HOME/miniconda3"
fi

_PYTHON="python"
if [[ -n "$CONDA_ROOT" && -x "${CONDA_ROOT}/bin/conda" ]]; then
    BBO_PYTHON="${CONDA_ROOT}/envs/BBO_Task/bin/python"
    if [[ -x "$BBO_PYTHON" ]]; then
        _PYTHON="$BBO_PYTHON"
        echo "[$(date)] Using BBO_Task Python: $_PYTHON"
    else
        if "${CONDA_ROOT}/bin/conda" run -n BBO_Task python --version > /dev/null 2>&1; then
            _PYTHON="${CONDA_ROOT}/bin/conda run -n BBO_Task python"
            echo "[$(date)] Using conda run -n BBO_Task"
        else
            echo "[$(date)] WARNING: BBO_Task Python not found, using system python"
        fi
    fi
else
    echo "[$(date)] WARNING: conda not found, using system python"
fi

cd "$WORKSPACE_DIR"

echo "[$(date)] Job started on $(hostname)"
echo "[$(date)] WORKSPACE=$WORKSPACE_DIR"
echo "[$(date)] TEST_FUNC=$TEST_FUNC DIMS=$TEST_DIMS BUDGET=$TEST_BUDGET SEED=$TEST_SEED MODE=$TEST_MODE"

#---------------- 运行测试 ----------------
PYTHON_ARGS="-f $TEST_FUNC -d $TEST_DIMS -b $TEST_BUDGET -s $TEST_SEED -m $TEST_MODE"

if [[ "$TEST_CONTINUOUS" == "1" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS -c"
fi

$_PYTHON test_scalpel_timing.py $PYTHON_ARGS

echo "[$(date)] Test finished"
