#!/bin/bash
#===========================================================
# submit_all.sh — 一次性提交全部算法的 SLURM job
# 放在 BBO_Work 同级目录
#
# Usage:
#   ./submit_all.sh                  # 默认跑 cec+mujoco+lasso，1 次随机 seed
#   ./submit_all.sh --all             # 跑全部 benchmark（含 mopta）
#   ./submit_all.sh --no-wandb        # 禁用 wandb
#   ./submit_all.sh --all --no-wandb  # 组合使用
#   ./submit_all.sh --seed 7 622 23 365 98   # 指定固定 seed（多个 seed = 多次 run）
#   ./submit_all.sh --use-GPU-occupied        # 允许 GPU 被占用
#
# 默认行为（不传 --seed）：
#   每个 job 不传 seed，Python 内部自动生成 1 个随机 seed
#
# 文件结构：
#   <some_path>/          ← run_slurm.sh and submit_all.sh live here (BBO_Work 同级)
#   ├── run_slurm.sh
#   ├── submit_all.sh
#   └── BBO_Work/         ← 代码仓库
#===========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_slurm.sh"

# task_id → 算法 映射表（与 run_slurm.sh 保持一致）
declare -a TASK_ID_ALGO=(
    "0:bo"
    "1:turbo"
    "2:cmaes"
    "3:scalpel"
    "4:hesbo"
    "5:baxus"
)

#---------------- parse arguments into seed vs benchmark groups ----------------
SEED_ARGS=()
BENCHMARK_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEED_ARGS+=("$1")
            shift
            # 收集 --seed 之后所有非标志参数，直到遇到下一个 --
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEED_ARGS+=("$1")
                shift
            done
            ;;
        *)
            BENCHMARK_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "==========================================================="
echo "Submitting BBO benchmarks on all 7 algorithms"
echo "==========================================================="
if [[ ${#SEED_ARGS[@]} -gt 0 ]]; then
    echo "Seeds:   ${SEED_ARGS[*]}"
else
    echo "Seeds:   (auto-generated 1 random seed per job)"
fi
[[ ${#BENCHMARK_ARGS[@]} -gt 0 ]] && echo "Args:    ${BENCHMARK_ARGS[*]}"
echo "==========================================================="
echo ""

for entry in "${TASK_ID_ALGO[@]}"; do
    TASK_ID="${entry%%:*}"
    ALGO="${entry#*:}"

    echo ">>> Submitting task_id=$TASK_ID ($ALGO) ..."

    if [[ ! -f "$SBATCH_SCRIPT" ]]; then
        echo "    ERROR: $SBATCH_SCRIPT not found, aborting."
        exit 1
    fi

    # task_id + benchmark 选项 + seed 选项，一起透传给 run_slurm.sh
    JOB_ID=$(sbatch "$SBATCH_SCRIPT" "$TASK_ID" "${BENCHMARK_ARGS[@]}" "${SEED_ARGS[@]}" 2>&1 | grep -oP '(?<=Submitted batch job )\d+')

    if [[ -n "$JOB_ID" ]]; then
        echo "    submitted: job_id=$JOB_ID"
    else
        echo "    ERROR: sbatch failed or returned unexpected output"
    fi
    echo ""
done

echo "==========================================================="
echo "All 7 jobs submitted."
echo "Check status: squeue -u \$USER"
echo "==========================================================="
