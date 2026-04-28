#!/bin/bash
#===========================================================
# run_slurm.sh — SLURM job script for BBO benchmark
# Usage: sbatch run_slurm.sh <task_id> [OPTIONS]
#   <task_id>: 0=bo, 1=turbo, 2=cmaes, 3=scalpel,
#               4=hesbo, 5=baxus, 6=saasbo
#   e.g.  sbatch run_slurm.sh 5
#         sbatch run_slurm.sh 5 --cec --mujoco
#         sbatch run_slurm.sh 5 --all --seed 7 622 23 365 98
#
# File structure:
#   <some_path>/          ← run_slurm.sh and submit_all.sh live here (BBO_Work 同级)
#   ├── run_slurm.sh
#   ├── submit_all.sh
#   └── BBO_Work/         ← 代码仓库
#===========================================================

#---------------- SBATCH directives ----------------
#SBATCH --job-name=bbo_single_algo
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --chdir=/home/weijun/assigned_runs
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e

#---------------- algorithm mapping ----------------
declare -a ALGOS=(
    "bo"
    "turbo"
    "cmaes"
    "scalpel"
    "hesbo"
    "baxus"
)

#---------------- default benchmark selection ----------------
RUN_CEC=true
RUN_MUJOCO=true
RUN_LASSO=true
RUN_MOPTA=true

WANDB_ENABLED=true
WANDB_PROJECT="bbo"
WANDB_ENTITY="hongweijun_jack-peking-university"
WANDB_BASE_GROUP="bbo-main"
WANDB_RUN_TAG="run-$(date +%Y%m%d-%H%M%S)"
USE_GPU_OCCUPIED=false

#---------------- default benchmark tasks ----------------
ALL_FUNCS=(ackley rastrigin rosenbrock griewank michalewicz schwefel levy)
ALL_MUJOCO=(swimmer:16 hopper:33 halfcheetah:102 walker2d:102 ant:216 humanoid:6392)
ALL_LASSO=(dna rcv1)

#---------------- seed: 空字符串 → 自动生成 1 个随机 seed ----------------
SEED_STR=""

#---------------- parse arguments ----------------
usage() {
    echo "Usage: sbatch run_slurm.sh <task_id> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  <task_id>  算法编号："
    echo "              0=bo  1=turbo  2=cmaes  3=scalpel"
    echo "              4=hesbo  5=baxus  6=saasbo"
    echo ""
    echo "Benchmark selection (default: all enabled):"
    echo "  --cec               启用 CEC"
    echo "  --no-cec            禁用 CEC"
    echo "  --mujoco            启用 MuJoCo"
    echo "  --no-mujoco         禁用 MuJoCo"
    echo "  --lasso             启用 Lasso-Bench"
    echo "  --no-lasso          禁用 Lasso-Bench"
    echo "  --mopta             启用 Mopta08"
    echo "  --no-mopta          禁用 Mopta08"
    echo "  --all               启用全部 benchmark"
    echo "  --none              禁用全部 benchmark"
    echo "  (多个 --xxx 可以叠加，例如 --mujoco --lasso --mopta)"
    echo ""
    echo "Other options:"
    echo "  --no-wandb          禁用 wandb 记录"
    echo "  --use-GPU-occupied  允许使用已有进程占用但显存有空闲的 GPU"
    echo "  --seed N [N2 ...]   指定随机种子（不指定时：自动生成 1 个随机 seed）"
    echo ""
    echo "Examples:"
    echo "  sbatch run_slurm.sh 5                     # 默认跑全部 benchmark"
    echo "  sbatch run_slurm.sh 5 --mujoco --lasso --mopta  # 只跑这三个"
    echo "  sbatch run_slurm.sh 5 --no-cec --mopta    # 除 CEC 外跑 mopta"
}

if [[ -z "$1" || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

TASK_ID="$1"
shift

if [[ ! "$TASK_ID" =~ ^[0-6]$ ]]; then
    echo "ERROR: task_id must be 0-6, got: $TASK_ID"
    usage
    exit 1
fi

ALGO="${ALGOS[TASK_ID]}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cec)
            RUN_CEC=true
            shift
            ;;
        --no-cec)
            RUN_CEC=false
            shift
            ;;
        --mujoco)
            RUN_MUJOCO=true
            shift
            ;;
        --no-mujoco)
            RUN_MUJOCO=false
            shift
            ;;
        --lasso)
            RUN_LASSO=true
            shift
            ;;
        --no-lasso)
            RUN_LASSO=false
            shift
            ;;
        --mopta)
            RUN_MOPTA=true
            shift
            ;;
        --no-mopta)
            RUN_MOPTA=false
            shift
            ;;
        --all)
            RUN_CEC=true; RUN_MUJOCO=true; RUN_LASSO=true; RUN_MOPTA=true
            shift
            ;;
        --none)
            RUN_CEC=false; RUN_MUJOCO=false; RUN_LASSO=false; RUN_MOPTA=false
            shift
            ;;
        --no-wandb)
            WANDB_ENABLED=false
            shift
            ;;
        --use-GPU-occupied)
            USE_GPU_OCCUPIED=true
            shift
            ;;
        --seed)
            shift
            SEEDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEEDS+=("$1")
                shift
            done
            SEED_STR="${SEEDS[*]}"
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$RUN_CEC" == false && "$RUN_MUJOCO" == false && "$RUN_LASSO" == false && "$RUN_MOPTA" == false ]]; then
    echo "ERROR: No benchmark selected"
    exit 1
fi

#---------------- working directory ----------------
# NOTE: In SLURM, job always starts in spool dir unless --chdir is set above.
#       ${BASH_SOURCE[0]} may be unreliable in SLURM, so we hardcode the known path.
SCRIPT_DIR="/home/weijun/assigned_runs"
WORKSPACE_DIR="$SCRIPT_DIR/BBO_Work"

mkdir -p "$WORKSPACE_DIR/cec_functions/results"
mkdir -p "$WORKSPACE_DIR/mujoco/results"
mkdir -p "$WORKSPACE_DIR/lasso_bench/results"
mkdir -p "$WORKSPACE_DIR/mopta/results"

#---------------- conda-free Python setup (same logic as run_bbo.sh) ----------------
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
echo "[$(date)] ALGO=$ALGO (task_id=$TASK_ID)"
echo "[$(date)] WORKSPACE=$WORKSPACE_DIR"

#---------------- helper functions ----------------
cec_budget() {
    case $1 in
        20)  echo 2000 ;;
        50)  echo 7000 ;;
        100) echo 12000 ;;
        *)   echo 5000 ;;
    esac
}

mujoco_budget() {
    case $1 in
        16)           echo 2000 ;;
        33)           echo 5000 ;;
        102|216|6392) echo 40000 ;;
        *)            echo 5000 ;;
    esac
}

log() { echo "[$(date)] $*"; }

#---------------- wandb args ----------------
if [[ "$WANDB_ENABLED" == true ]]; then
    WANDB_BASE_ARGS="--wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
else
    WANDB_BASE_ARGS="--no-wandb"
fi

if [[ "$USE_GPU_OCCUPIED" == true ]]; then
    USE_GPU_ARGS="--use-GPU-occupied"
else
    USE_GPU_ARGS=""
fi

# seed: 只在有指定 seed 时才传 --seed（空字符串 → Python 自动生成 1 个随机 seed）
if [[ -n "$SEED_STR" ]]; then
    SEED_ARGS="--seed ${SEED_STR}"
else
    SEED_ARGS=""
fi

#---------------- CEC Functions ----------------
if [[ "$RUN_CEC" == true ]]; then
    log "Running CEC for $ALGO"
    for FUNC in "${ALL_FUNCS[@]}"; do
        for D in 20 50 100; do
            BUDGET=$(cec_budget "$D")
            WANDB_GROUP="${WANDB_BASE_GROUP}/cec/${FUNC}_${D}d"
            WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALGO} cec ${FUNC} ${D}d ${WANDB_RUN_TAG}"
            log "  CEC: ${FUNC}_${D}d (budget=$BUDGET)"
            $_PYTHON run_bbo_benchmark.py \
                --dims "$D" --budget "$BUDGET" --batch 1 \
                --algorithms "$ALGO" --functions "$FUNC" \
                --result-dir cec_functions/results \
                $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS
        done
    done
    log "CEC done"
fi

#---------------- MuJoCo ----------------
if [[ "$RUN_MUJOCO" == true ]]; then
    log "Running MuJoCo for $ALGO"
    for ENV_CFG in "${ALL_MUJOCO[@]}"; do
        ENV_NAME="${ENV_CFG%%:*}"
        DIMS="${ENV_CFG#*:}"
        BUDGET=$(mujoco_budget "$DIMS")
        WANDB_GROUP="${WANDB_BASE_GROUP}/mujoco/${ENV_NAME}"
        WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALGO} mujoco ${ENV_NAME} ${WANDB_RUN_TAG}"
        log "  MuJoCo: ${ENV_NAME} (dim=$DIMS, budget=$BUDGET)"
        $_PYTHON run_bbo_mujoco.py \
            --algorithms "$ALGO" --environments "$ENV_NAME" \
            --budget "$BUDGET" --batch 1 \
            --result-dir mujoco/results \
            --no-per-env-budget \
            $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS
    done
    log "MuJoCo done"
fi

#---------------- Lasso-Bench ----------------
if [[ "$RUN_LASSO" == true ]]; then
    log "Running Lasso for $ALGO"
    for BENCH in "${ALL_LASSO[@]}"; do
        WANDB_GROUP="${WANDB_BASE_GROUP}/lasso/${BENCH}"
        WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALGO} lasso ${BENCH} ${WANDB_RUN_TAG}"
        log "  Lasso: ${BENCH}"
        $_PYTHON run_bbo_lasso.py \
            --algorithms "$ALGO" --benchmarks "$BENCH" \
            --budget 1000 --batch 1 \
            --result-dir lasso_bench/results \
            $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS
    done
    log "Lasso done"
fi

#---------------- Mopta08 ----------------
if [[ "$RUN_MOPTA" == true ]]; then
    log "Running Mopta08 for $ALGO"
    WANDB_GROUP="${WANDB_BASE_GROUP}/mopta/mopta08"
    WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALGO} mopta mopta08 ${WANDB_RUN_TAG}"
    $_PYTHON run_bbo_mopta.py \
        --algorithms "$ALGO" \
        --budget 1000 --batch 1 \
        --result-dir mopta/results \
        $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS
    log "Mopta08 done"
fi

log "All benchmarks finished successfully"
