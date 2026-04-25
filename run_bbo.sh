#!/bin/bash

# ============================================================
# BBO Benchmark Runner
# - 每个算法一个后台进程并行
# - 每个算法内部按任务串行
# - GPU 由 baselines + utils/gpu_scheduler 自动分配
# - Lasso 默认仅真实任务（dna, rcv1）
# - Mopta08 通过 --mopta 或 --all 启用
# - 支持 --seed N1 N2 N3 ... 空格分隔的 seed 列表，每个独立跑一次
# - 未指定 --seed 时自动生成 5 个随机 seed
# ============================================================

set -u
cd "$(dirname "$0")"

RUN_CEC=true
RUN_MUJOCO=true
RUN_LASSO=true
RUN_MOPTA=false

WANDB_ENABLED=true
ALGO_FILTER=()
FUNC_FILTER=()
ENV_FILTER=()
BENCH_FILTER=()

# ALL_ALGOS=(bo turbo cmaes scalpel hesbo baxus saasbo)
ALL_ALGOS=(bo turbo cmaes scalpel hesbo baxus)
ALL_FUNCS=(ackley rastrigin rosenbrock griewank michalewicz schwefel levy)
ALL_MUJOCO=(swimmer:16 hopper:33 halfcheetah:102 walker2d:102 ant:216 humanoid:6392)
ALL_LASSO=(dna rcv1)

WANDB_PROJECT="bbo"
WANDB_ENTITY="hongweijun_jack-peking-university"
WANDB_BASE_GROUP="bbo-main"
WANDB_RUN_TAG="run-$(date +%Y%m%d-%H%M%S)"
USE_GPU_OCCUPIED=false

# seed: 初始化为空字符串，set -u 下不会报错
SEED_STR=""

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Benchmark selection (default: all):"
  echo "  --cec                    只跑 CEC"
  echo "  --mujoco                 只跑 MuJoCo"
  echo "  --lasso                  只跑 Lasso-Bench"
  echo "  --mopta                  只跑 Mopta08"
  echo "  --all                    跑全部"
  echo ""
  echo "Filters (support multiple values):"
  echo "  --algo A1 A2 ...         只跑指定算法"
  echo "  --func F1 F2 ...         只跑指定 CEC 函数"
  echo "  --env E1 E2 ...          只跑指定 MuJoCo 环境"
  echo "  --bench B1 B2 ...        只跑指定 Lasso 任务"
  echo ""
  echo "Wandb options:"
  echo "  --wandb-base-group NAME  统一显示分组前缀（默认: bbo-main）"
    echo "  --wandb-run-tag TAG      本次批次标签（默认: run-时间戳）"
    echo "  --no-wandb               禁用 wandb 记录（加快调试速度）"
    echo "  --seed N [N2 N3 ...]   空格分隔的随机种子"
    echo "                           示例: --seed 14 888 92 473 546"
    echo "                           未指定时：自动生成 5 个随机 seed"
    echo "  --use-GPU-occupied       允许使用已有其他进程占用但显存有空闲的 GPU（默认关闭）"
    echo ""
  echo "Examples:"
  echo "  $0 --cec --algo bo turbo --func ackley rastrigin --seed 42 43 44 45 46"
  echo "  $0 --mujoco --env hopper ant --algo scalpel baxus --seed 0 100 200"
  echo "  $0 --all --seed 14 888 92 473 546   # 自动 5 次 run"
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cec)
      RUN_CEC=true; RUN_MUJOCO=false; RUN_LASSO=false; RUN_MOPTA=false
      shift
      ;;
    --mujoco)
      RUN_CEC=false; RUN_MUJOCO=true; RUN_LASSO=false; RUN_MOPTA=false
      shift
      ;;
    --lasso)
      RUN_CEC=false; RUN_MUJOCO=false; RUN_LASSO=true; RUN_MOPTA=false
      shift
      ;;
    --mopta)
      RUN_CEC=false; RUN_MUJOCO=false; RUN_LASSO=false; RUN_MOPTA=true
      shift
      ;;
    --all)
      RUN_CEC=true; RUN_MUJOCO=true; RUN_LASSO=true; RUN_MOPTA=true
      shift
      ;;
    --algo)
      shift
      ALGO_FILTER=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ALGO_FILTER+=("$1")
        shift
      done
      ;;
    --func)
      shift
      FUNC_FILTER=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        FUNC_FILTER+=("$1")
        shift
      done
      ;;
    --env)
      shift
      ENV_FILTER=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ENV_FILTER+=("$1")
        shift
      done
      ;;
    --bench)
      shift
      BENCH_FILTER=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        BENCH_FILTER+=("$1")
        shift
      done
      ;;
    --wandb-base-group)
      WANDB_BASE_GROUP="$2"
      shift 2
      ;;
    --wandb-run-tag)
      WANDB_RUN_TAG="$2"
      shift 2
      ;;
    --no-wandb)
      WANDB_ENABLED=false
      shift
      ;;
    --seed)
      # 收集所有后续非标志参数作为 seed
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      # 如果没收集到（下一个是 -- 开头的或空了），报错
      if [[ ${#SEEDS[@]} -eq 0 ]]; then
        echo "ERROR: --seed requires at least one integer argument"
        echo "       示例: --seed 14 888 92 473 546"
        exit 1
      fi
      SEED_STR="${SEEDS[*]}"
      ;;
    --run-times)
      echo "WARNING: --run-times is deprecated. Run times is now determined by --seed count (default: 5 random seeds)."
      echo "         Use --seed to explicitly specify seeds, or omit both to auto-generate 5 random seeds."
      shift 2
      ;;
    --use-GPU-occupied)
      USE_GPU_OCCUPIED=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
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

if [[ ${#ALGO_FILTER[@]} -gt 0 ]]; then
  SELECTED_ALGOS=("${ALGO_FILTER[@]}")
else
  SELECTED_ALGOS=("${ALL_ALGOS[@]}")
fi

if [[ ${#FUNC_FILTER[@]} -gt 0 ]]; then
  FUNC_LIST=("${FUNC_FILTER[@]}")
else
  FUNC_LIST=("${ALL_FUNCS[@]}")
fi

if [[ ${#BENCH_FILTER[@]} -gt 0 ]]; then
  LASSO_LIST=("${BENCH_FILTER[@]}")
else
  LASSO_LIST=("${ALL_LASSO[@]}")
fi

MUJOCO_LIST=()
if [[ ${#ENV_FILTER[@]} -gt 0 ]]; then
  for entry in "${ALL_MUJOCO[@]}"; do
    env_name="${entry%%:*}"
    for wanted in "${ENV_FILTER[@]}"; do
      if [[ "$env_name" == "$wanted" ]]; then
        MUJOCO_LIST+=("$entry")
      fi
    done
  done
else
  MUJOCO_LIST=("${ALL_MUJOCO[@]}")
fi

mkdir -p cec_functions/results
mkdir -p mujoco/results
mkdir -p lasso_bench/results
mkdir -p mopta/results
mkdir -p output

# seed 逻辑：传了 seed → 使用这些 seed；没传 → 自动生成 1 个随机 seed
if [[ -n "$SEED_STR" ]]; then
  read -ra _tmp_seeds <<< "$SEED_STR"
  ACTUAL_RUN_TIMES=${#_tmp_seeds[@]}
  ACTUAL_SEEDS=("${_tmp_seeds[@]}")
else
  # 未指定 seed：生成 1 个随机 seed
  ACTUAL_RUN_TIMES=1
  _rand_seed=$(python3 -c "import random; print(random.randint(0, 2147483647))")
  ACTUAL_SEEDS=("$_rand_seed")
  SEED_STR="$_rand_seed"
fi

echo ""
echo "=================================================="
echo "BBO Benchmark - Configuration"
echo "=================================================="
echo "Algorithms: ${SELECTED_ALGOS[*]}"
[[ "$RUN_CEC" == true ]] && echo "CEC funcs: ${FUNC_LIST[*]}"
[[ "$RUN_MUJOCO" == true ]] && echo "MuJoCo envs: ${MUJOCO_LIST[*]}"
[[ "$RUN_LASSO" == true ]] && echo "Lasso tasks: ${LASSO_LIST[*]}"
[[ "$RUN_MOPTA" == true ]] && echo "Mopta: mopta08"
echo "Run times: ${ACTUAL_RUN_TIMES}"
echo "Seeds: ${SEED_STR}"
echo "Wandb: project=${WANDB_PROJECT}, base_group=${WANDB_BASE_GROUP}"
echo "Wandb run tag: ${WANDB_RUN_TAG}"
echo "=================================================="

echo "Submitting ${#SELECTED_ALGOS[@]} algorithm workers..."

for ALG in "${SELECTED_ALGOS[@]}"; do
  MASTER_LOG="output/${ALG}.log"
  OUT_DIR="output/${ALG}"
  mkdir -p "$OUT_DIR"

  echo ">>> Algorithm: $ALG | Log: $MASTER_LOG"

  nohup bash -c '
    ALG="'"$ALG"'"
    OUT_DIR="'"$OUT_DIR"'"
    RUN_CEC="'"$RUN_CEC"'"
    RUN_MUJOCO="'"$RUN_MUJOCO"'"
    RUN_LASSO="'"$RUN_LASSO"'"
    RUN_MOPTA="'"$RUN_MOPTA"'"
    WANDB_ENABLED="'"$WANDB_ENABLED"'"
    WANDB_PROJECT="'"$WANDB_PROJECT"'"
    WANDB_ENTITY="'"$WANDB_ENTITY"'"
    WANDB_BASE_GROUP="'"$WANDB_BASE_GROUP"'"
    WANDB_RUN_TAG="'"$WANDB_RUN_TAG"'"
    SEED_STR="'"$SEED_STR"'"
    USE_GPU_OCCUPIED="'"$USE_GPU_OCCUPIED"'"

    FUNC_LIST_STR="'"${FUNC_LIST[*]}"'"
    MUJOCO_LIST_STR="'"${MUJOCO_LIST[*]}"'"
    LASSO_LIST_STR="'"${LASSO_LIST[*]}"'"

    eval "$(conda shell.bash hook)"
    conda activate BBO_Task
    set -e

    LOCK_DIR="/tmp/bbo_algo_locks"
    mkdir -p "$LOCK_DIR"
    exec 9>"$LOCK_DIR/${ALG}.lock"
    if ! flock -n 9; then
      echo "[$(date +%H:%M:%S)] ${ALG} failed: resources/GPU likely occupied by another run_bbo.sh invocation."
      exit 1
    fi

    log() { echo "[$(date +%H:%M:%S)] $*"; }

    if [[ "$WANDB_ENABLED" == true ]]; then
      WANDB_BASE_ARGS="--wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
    else
      WANDB_BASE_ARGS=""
    fi

    # seed: 空格分隔的字符串，直接传给 Python 脚本（内部解析）
    if [[ -n "$SEED_STR" ]]; then
      SEED_ARGS="--seed ${SEED_STR}"
    else
      SEED_ARGS=""
    fi

    if [[ "$USE_GPU_OCCUPIED" == true ]]; then
      USE_GPU_ARGS="--use-GPU-occupied"
    else
      USE_GPU_ARGS=""
    fi

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
        16)          echo 2000  ;;
        33)          echo 5000  ;;
        102|216|6392) echo 40000 ;;
        *)           echo 5000  ;;
      esac
    }

    read -ra FUNC_LIST <<< "$FUNC_LIST_STR"
    read -ra MUJOCO_LIST <<< "$MUJOCO_LIST_STR"
    read -ra LASSO_LIST <<< "$LASSO_LIST_STR"

    if [[ "$RUN_CEC" == true ]]; then
      log "Running CEC for ${ALG}"
      for FUNC in "${FUNC_LIST[@]}"; do
        for D in 20 50 100; do
          BUDGET=$(cec_budget "$D")
          WANDB_GROUP="${WANDB_BASE_GROUP}/cec/${FUNC}_${D}d"
          WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALG} cec ${FUNC} ${D}d ${WANDB_RUN_TAG}"
          python run_bbo_benchmark.py \
            --dims "$D" --budget "$BUDGET" --batch 1 \
            --algorithms "$ALG" --functions "$FUNC" \
            --result-dir cec_functions/results \
            $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS \
            > "$OUT_DIR/cec_${FUNC}_${D}d.out" 2>&1
        done
      done
    fi

    if [[ "$RUN_MUJOCO" == true ]]; then
      log "Running MuJoCo for ${ALG}"
      for ENV_CFG in "${MUJOCO_LIST[@]}"; do
        ENV_NAME="${ENV_CFG%%:*}"
        DIMS="${ENV_CFG#*:}"
        BUDGET=$(mujoco_budget "$DIMS")
        WANDB_GROUP="${WANDB_BASE_GROUP}/mujoco/${ENV_NAME}"
        WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALG} mujoco ${ENV_NAME} ${WANDB_RUN_TAG}"
        python run_bbo_mujoco.py \
          --algorithms "$ALG" --environments "$ENV_NAME" \
          --budget "$BUDGET" --batch 1 \
          --result-dir mujoco/results \
          --no-per-env-budget \
          $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS \
          > "$OUT_DIR/mujoco_${ENV_NAME}.out" 2>&1
      done
    fi

    if [[ "$RUN_LASSO" == true ]]; then
      log "Running Lasso for ${ALG}"
      for BENCH in "${LASSO_LIST[@]}"; do
        WANDB_GROUP="${WANDB_BASE_GROUP}/lasso/${BENCH}"
        WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALG} lasso ${BENCH} ${WANDB_RUN_TAG}"
        python run_bbo_lasso.py \
          --algorithms "$ALG" --benchmarks "$BENCH" \
          --budget 1000 --batch 1 \
          --result-dir lasso_bench/results \
          $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS \
          > "$OUT_DIR/lasso_${BENCH}.out" 2>&1
      done
    fi

    if [[ "$RUN_MOPTA" == true ]]; then
      log "Running Mopta08 for ${ALG}"
      WANDB_GROUP="${WANDB_BASE_GROUP}/mopta/mopta08"
      WANDB_ARGS="${WANDB_BASE_ARGS} --wandb-group ${WANDB_GROUP} --wandb-tags ${ALG} mopta mopta08 ${WANDB_RUN_TAG}"
        python run_bbo_mopta.py \
          --algorithms "$ALG" \
          --budget 1000 --batch 1 \
          --result-dir mopta/results \
          $WANDB_ARGS $SEED_ARGS $USE_GPU_ARGS \
          > "$OUT_DIR/mopta08.out" 2>&1
    fi

    log "${ALG} done"
  ' > "$MASTER_LOG" 2>&1 &

  echo "    -> submitted (PID: $!)"
done

echo ""
echo "=================================================="
echo "All ${#SELECTED_ALGOS[@]} workers submitted"
echo "Logs: output/{algorithm}.log"
echo "Results: cec_functions/results, mujoco/results, lasso_bench/results, mopta/results"
echo "Watch: tail -f output/${SELECTED_ALGOS[0]}.log"
echo "=================================================="
