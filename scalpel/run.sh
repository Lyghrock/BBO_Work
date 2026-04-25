#!/bin/bash
# Batch submission script for SLURM

echo "=================================================="
echo "🚀 SLURM Batch Job Submission"
echo "=================================================="
echo ""

# Create directories
mkdir -p slurm_logs

# Counter
count=0

# Function to submit a job
submit_job() {
    local func=$1
    local dims=$2
    local samples=$3
    local continuous=$4
    
    if [ "$continuous" = "true" ]; then
        mode="cont"
        mode_flag="--continuous"
    else
        mode="disc"
        mode_flag=""
    fi
    
    job_name="${func}_${dims}d_${mode}"
    
    echo "[$(printf %2d $((++count)))] Submitting: ${job_name}"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=${job_name}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=999:99:99
#SBATCH --output=slurm_logs/${job_name}_%j.out

python -u ../../run_bbo_benchmark.py --algorithms scalpel --functions ${func} --dims ${dims} --budget ${samples} ${mode_flag}
EOF
    
    sleep 0.3
}

echo "Submitting jobs..."
echo ""

# Ackley - Discrete
echo "📐 Ackley (Discrete)..."
submit_job "ackley" 20 10000 "false"
submit_job "ackley" 100 10000 "false"
submit_job "ackley" 500 10000 "false"
submit_job "ackley" 1000 10000 "false"

# Rastrigin - Discrete
echo "📐 Rastrigin (Discrete)..."
submit_job "rastrigin" 20 10000 "false"
submit_job "rastrigin" 100 10000 "false"
submit_job "rastrigin" 500 10000 "false"
submit_job "rastrigin" 1000 10000 "false"

# Rosenbrock - Continuous
echo "✅ Rosenbrock (Continuous)..."
submit_job "rosenbrock" 20 10000 "true"
submit_job "rosenbrock" 100 10000 "true"
submit_job "rosenbrock" 500 10000 "true"
submit_job "rosenbrock" 1000 10000 "true"

# Griewank - Continuous
echo "✅ Griewank (Continuous)..."
submit_job "griewank" 20 10000 "true"
submit_job "griewank" 100 10000 "true"
submit_job "griewank" 500 10000 "true"
submit_job "griewank" 1000 10000 "true"

# Michalewicz - Continuous
echo "✅ Michalewicz (Continuous)..."
submit_job "michalewicz" 20 10000 "true"
submit_job "michalewicz" 100 10000 "true"
submit_job "michalewicz" 500 10000 "true"
submit_job "michalewicz" 1000 10000 "true"

# Schwefel - Continuous
echo "✅ Schwefel (Continuous)..."
submit_job "schwefel" 20 10000 "true"
submit_job "schwefel" 100 10000 "true"
submit_job "schwefel" 500 10000 "true"
submit_job "schwefel" 1000 10000 "true"

# Levy - Continuous
echo "✅ Levy (Continuous)..."
submit_job "levy" 20 10000 "true"
submit_job "levy" 100 10000 "true"
submit_job "levy" 500 10000 "true"
submit_job "levy" 1000 10000 "true"

echo ""
echo "=================================================="
echo "✅ All ${count} jobs submitted!"
echo "=================================================="
echo ""
echo "📊 Check status: squeue -u \$USER"
echo "📋 View logs:    ls slurm_logs/"
echo "🛑 Cancel all:   scancel -u \$USER"
echo ""