#!/bin/bash
#============================================================
# scaling_test.sh — Strong & Weak Scaling for Monte Carlo
#============================================================

SEED=42
DAYS=10        # log2_days = 10 -> 1024 days per sim
BINARY=./test
OUTFILE=scaling_results.csv
LOG=scaling_log.txt

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "============================================" | tee $LOG
echo " Monte Carlo Scaling Benchmark" | tee -a $LOG
echo " Date: $(date)" | tee -a $LOG
echo " Host: $(hostname)" | tee -a $LOG
echo " Binary: $BINARY" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "" | tee -a $LOG

#------------------------------------------------------------
# Helper: run one configuration, extract times, compute metrics
# Args: $1=ranks $2=log2_sims $3=days $4=trial
#------------------------------------------------------------
run_config() {
    local ranks=$1
    local log2_sims=$2
    local days=$3
    local trial=$4
    local total_sims=$((1 << log2_sims))

    local raw
    raw=$(mpirun -np $ranks $BINARY $SEED $log2_sims $days 2>/dev/null)
    local exit_code=$?

    if [ $exit_code -ne 0 ] || [ -z "$raw" ]; then
        echo "FAILED"
        return 1
    fi

    local compute_time io_time
    compute_time=$(echo "$raw" | grep "TOTAL TIME"        | awk '{print $3}')
    io_time=$(     echo "$raw" | grep "PRINT TO FILE TIME" | awk '{print $5}')

    # Fallback if grep finds nothing
    compute_time=${compute_time:-0}
    io_time=${io_time:-0}

    echo "${compute_time},${io_time}"
}

#------------------------------------------------------------
# Average multiple trials
# Args: $1=ranks $2=log2_sims $3=days $4=num_trials
#------------------------------------------------------------
avg_trials() {
    local ranks=$1
    local log2_sims=$2
    local days=$3
    local num_trials=$4

    local sum_compute=0 sum_io=0 success=0

    for trial in $(seq 1 $num_trials); do
        local result
        result=$(run_config $ranks $log2_sims $days $trial)
        if [ "$result" = "FAILED" ]; then
            echo -e "  ${RED}Trial $trial FAILED — skipping${NC}" >&2
            continue
        fi
        local c i
        c=$(echo $result | cut -d',' -f1)
        i=$(echo $result | cut -d',' -f2)
        sum_compute=$(echo "$sum_compute + $c" | bc -l)
        sum_io=$(echo "$sum_io + $i" | bc -l)
        success=$((success + 1))
    done

    if [ $success -eq 0 ]; then
        echo "0,0"
        return
    fi

    local avg_c avg_i
    avg_c=$(echo "scale=6; $sum_compute / $success" | bc -l)
    avg_i=$(echo "scale=6; $sum_io      / $success" | bc -l)
    echo "${avg_c},${avg_i}"
}

#============================================================
# CONFIGURATION
#============================================================
TRIALS=3          # Trials to average per configuration

# Strong scaling: fixed total problem sizes to test
# Each entry is log2_sims (total simulations = 2^N)
STRONG_SIZES=(18 20 22 24)

# Strong scaling rank counts
STRONG_RANKS=(1 2 4 8 16 32)

# Weak scaling: sims per rank (log2)
# Total sims = ranks * 2^WEAK_PER_RANK_LOG2
WEAK_PER_RANK_LOG2=18      # 262144 sims per rank

# Weak scaling rank counts
WEAK_RANKS=(1 2 4 8 16 32)

# Day configurations to sweep (log2_days)
DAY_CONFIGS=(8 10 12)      # 256, 1024, 4096 days

#============================================================
# CSV HEADER
#============================================================
echo "experiment,ranks,log2_sims,total_sims,log2_days,total_days,\
avg_compute_s,avg_io_s,speedup,efficiency_pct,weak_efficiency_pct" \
> $OUTFILE

#============================================================
# STRONG SCALING
#============================================================
echo -e "\n${CYAN}========== STRONG SCALING ==========${NC}" | tee -a $LOG

for log2_days in "${DAY_CONFIGS[@]}"; do
    total_days=$((1 << log2_days))
    echo -e "\n${YELLOW}--- Days = $total_days (log2=$log2_days) ---${NC}" | tee -a $LOG

    for log2_sims in "${STRONG_SIZES[@]}"; do
        total_sims=$((1 << log2_sims))
        echo -e "\n${GREEN}  Problem size: 2^$log2_sims = $total_sims sims${NC}" | tee -a $LOG

        baseline_compute=""

        for ranks in "${STRONG_RANKS[@]}"; do
            # Skip if local_n would be 0
            local_n=$((total_sims / ranks))
            if [ $local_n -lt $ranks ]; then
                echo "  Ranks=$ranks: SKIP (local_n too small)" | tee -a $LOG
                continue
            fi

            echo -n "  Ranks=$ranks, sims=2^$log2_sims, days=$total_days ... " | tee -a $LOG

            result=$(avg_trials $ranks $log2_sims $log2_days $TRIALS)
            avg_c=$(echo $result | cut -d',' -f1)
            avg_i=$(echo $result | cut -d',' -f2)

            # Capture baseline (1 rank)
            if [ -z "$baseline_compute" ]; then
                baseline_compute=$avg_c
            fi

            # Compute speedup and efficiency
            speedup=1.0
            efficiency=100.0
            if [ ! -z "$baseline_compute" ] && [ "$avg_c" != "0" ]; then
                speedup=$(echo "scale=4; $baseline_compute / $avg_c" | bc -l)
                efficiency=$(echo "scale=2; ($speedup / $ranks) * 100" | bc -l)
            fi

            echo "compute=${avg_c}s  io=${avg_i}s  speedup=${speedup}x  eff=${efficiency}%" | tee -a $LOG

            echo "strong,$ranks,$log2_sims,$total_sims,$log2_days,$total_days,\
${avg_c},${avg_i},${speedup},${efficiency}," >> $OUTFILE
        done
    done
done

#============================================================
# WEAK SCALING
#============================================================
echo -e "\n${CYAN}========== WEAK SCALING ==========${NC}" | tee -a $LOG

for log2_days in "${DAY_CONFIGS[@]}"; do
    total_days=$((1 << log2_days))
    echo -e "\n${YELLOW}--- Days = $total_days (log2=$log2_days) ---${NC}" | tee -a $LOG

    baseline_compute=""

    for ranks in "${WEAK_RANKS[@]}"; do
        # Total sims = ranks * 2^WEAK_PER_RANK_LOG2
        # log2(total) = log2(ranks) + WEAK_PER_RANK_LOG2
        # We need integer log2 of ranks
        log2_ranks=0
        tmp=$ranks
        while [ $tmp -gt 1 ]; do
            tmp=$((tmp / 2))
            log2_ranks=$((log2_ranks + 1))
        done

        log2_sims=$((WEAK_PER_RANK_LOG2 + log2_ranks))
        total_sims=$((1 << log2_sims))
        per_rank=$((1 << WEAK_PER_RANK_LOG2))

        echo -n "  Ranks=$ranks, total_sims=2^$log2_sims=$total_sims (${per_rank}/rank), days=$total_days ... " | tee -a $LOG

        result=$(avg_trials $ranks $log2_sims $log2_days $TRIALS)
        avg_c=$(echo $result | cut -d',' -f1)
        avg_i=$(echo $result | cut -d',' -f2)

        # Weak efficiency: T1 / Tn * 100
        if [ -z "$baseline_compute" ]; then
            baseline_compute=$avg_c
            weak_eff=100.0
        else
            if [ "$avg_c" != "0" ]; then
                weak_eff=$(echo "scale=2; ($baseline_compute / $avg_c) * 100" | bc -l)
            else
                weak_eff=0
            fi
        fi

        echo "compute=${avg_c}s  io=${avg_i}s  weak_eff=${weak_eff}%" | tee -a $LOG

        echo "weak,$ranks,$log2_sims,$total_sims,$log2_days,$total_days,\
${avg_c},${avg_i},,,$weak_eff" >> $OUTFILE
    done
done

#============================================================
# GPU UTILIZATION SWEEP (vary blocks/threads)
#============================================================
echo -e "\n${CYAN}========== GPU OCCUPANCY SWEEP (1 rank) ==========${NC}" | tee -a $LOG
echo "" | tee -a $LOG

# These require modifying the binary to accept blocks/threads as args
# OR you can hardcode configs here and recompile between runs
# This section just documents the configs you'd want to test:

cat << 'EOF' | tee -a $LOG
  Suggested GPU occupancy configs to test manually (edit num_blocks/num_threads in start_sims.c):
  Config A: blocks=64,  threads=128  -> 8192  total threads
  Config B: blocks=128, threads=256  -> 32768 total threads  (current)
  Config C: blocks=256, threads=256  -> 65536 total threads
  Config D: blocks=512, threads=256  -> 131072 total threads
  Config E: blocks=256, threads=512  -> 131072 total threads
  Config F: blocks=256, threads=1024 -> 262144 total threads
EOF

#============================================================
# SUMMARY TABLE
#============================================================
echo -e "\n${CYAN}========== RESULTS SUMMARY ==========${NC}" | tee -a $LOG
echo "" | tee -a $LOG
echo "Full results saved to: $OUTFILE" | tee -a $LOG
echo "Full log saved to:     $LOG" | tee -a $LOG
echo "" | tee -a $LOG

echo "--- STRONG SCALING (log2_sims=20, log2_days=10) ---" | tee -a $LOG
printf "%-8s %-14s %-10s %-10s %-12s\n" \
    "Ranks" "Compute(s)" "IO(s)" "Speedup" "Efficiency%" | tee -a $LOG
grep "^strong" $OUTFILE | awk -F',' '$3==20 && $5==10 {
    printf "%-8s %-14s %-10s %-10s %-12s\n", $2, $7, $8, $9, $10
}' | tee -a $LOG

echo "" | tee -a $LOG
echo "--- WEAK SCALING (2^18 sims/rank, log2_days=10) ---" | tee -a $LOG
printf "%-8s %-12s %-14s %-10s %-18s\n" \
    "Ranks" "TotalSims" "Compute(s)" "IO(s)" "WeakEfficiency%" | tee -a $LOG
grep "^weak" $OUTFILE | awk -F',' '$5==10 {
    printf "%-8s %-12s %-14s %-10s %-18s\n", $2, $4, $7, $8, $11
}' | tee -a $LOG

echo "" | tee -a $LOG
echo -e "${GREEN}Done.${NC}" | tee -a $LOG
