#!/bin/bash
# watch_job.sh - nvidia-smi style monitoring with multi-GPU support
# Usage: ./IDUN/watch_job.sh <JOBID>

JOBID=$1
REFRESH=1

if [ -z "$JOBID" ]; then
    echo "Usage: $0 <JOBID>"
    echo "Example: $0 23106207"
    exit 1
fi

NODE=$(squeue -j $JOBID -h -o "%N" 2>/dev/null)

if [ -z "$NODE" ]; then
    echo "Error: Job $JOBID not found or not running"
    echo ""
    echo "Your current jobs:"
    squeue -u $USER -o "%.10i %.20j %.8T %.10M %R"
    exit 1
fi

JOBNAME=$(squeue -j $JOBID -h -o "%j" 2>/dev/null)

# Hide cursor
tput civis
trap 'tput cnorm; echo ""; exit' INT TERM EXIT

# Clear screen once
clear

FIRST_RUN=true
while true; do
    if [ "$FIRST_RUN" = false ]; then
        tput cup 0 0
    fi
    FIRST_RUN=false

    TIMESTAMP=$(date +'%a %b %d %H:%M:%S %Y')

    # Find all PIDs for this job (multi-GPU jobs spawn multiple processes)
    PIDS=$(ssh $NODE "ps -u $USER -o pid,cmd | grep -E '(python|bash.*\.slurm)' | grep -v grep | awk '{print \$1}' 2>/dev/null")

    if [ -z "$PIDS" ]; then
        echo "Job $JOBID - Process not found (may be starting or completed)                    "
        if ! squeue -j $JOBID -h &>/dev/null; then
            echo "Job no longer in queue                                                        "
            break
        fi
        sleep $REFRESH
        continue
    fi

    # Count number of processes
    NUM_PROCS=$(echo "$PIDS" | wc -l)
    MAIN_PID=$(echo "$PIDS" | head -1)

    # Get all GPU info
    GPU_DATA=$(ssh $NODE "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null")

    if [ -z "$GPU_DATA" ]; then
        echo "No GPU data available                                                             "
        sleep $REFRESH
        continue
    fi

    # Count GPUs
    NUM_GPUS=$(echo "$GPU_DATA" | wc -l)

    # Get process info for main PID
    PROC_INFO=$(ssh $NODE "ps -p $MAIN_PID -o %cpu,%mem,rss --no-headers 2>/dev/null")
    CPU=$(echo $PROC_INFO | awk '{print $1}')
    MEM_PERCENT=$(echo $PROC_INFO | awk '{print $2}')
    RSS=$(echo $PROC_INFO | awk '{print $3}')
    RSS_MB=$((RSS / 1024))

    # Get system RAM
    RAM_INFO=$(ssh $NODE "free -h | grep Mem")
    RAM_USED=$(echo $RAM_INFO | awk '{print $3}')
    RAM_TOTAL=$(echo $RAM_INFO | awk '{print $2}')

    # Get all GPU processes for this user
    GPU_PROCS=$(ssh $NODE "nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader 2>/dev/null")

    # Print header
    printf "%s                    \n" "$TIMESTAMP"
    echo "+-----------------------------------------------------------------------------------------+"
    printf "| Job: %-8s | %-70s |\n" "$JOBID" "$JOBNAME"
    printf "| Node: %-40s      PIDs: %-4s  GPUs: %-4s |\n" "$NODE" "$NUM_PROCS" "$NUM_GPUS"
    echo "+-----------------------------------------------------------------------------------------+"
    echo "| GPU  Name                                      Temp   Pwr:Usage/Cap   Memory-Usage   Util |"
    echo "+-----------------------------------------------------------------------------------------+"

    # Print all GPUs
    echo "$GPU_DATA" | while IFS=',' read -r GPU_IDX GPU_NAME GPU_UTIL MEM_USED MEM_TOTAL TEMP POWER POWER_LIMIT; do
        # Trim whitespace
        GPU_NAME=$(echo $GPU_NAME | xargs)
        GPU_UTIL=$(echo $GPU_UTIL | xargs)
        MEM_USED=$(echo $MEM_USED | xargs)
        MEM_TOTAL=$(echo $MEM_TOTAL | xargs)
        TEMP=$(echo $TEMP | xargs)
        POWER=$(echo $POWER | xargs)
        POWER_LIMIT=$(echo $POWER_LIMIT | xargs)

        printf "|  %-2s  %-40s  %3s°C   %3sW / %3sW   %5sMiB / %5sMiB   %3s%% |\n" \
            "$GPU_IDX" "$GPU_NAME" "$TEMP" "$POWER" "$POWER_LIMIT" "$MEM_USED" "$MEM_TOTAL" "$GPU_UTIL"
    done

    echo "+-----------------------------------------------------------------------------------------+"
    echo "                                                                                           "
    echo "+-----------------------------------------------------------------------------------------+"
    echo "| GPU Processes:                                                                          |"
    echo "+-----------------------------------------------------------------------------------------+"

    # Show GPU processes
    if [ -z "$GPU_PROCS" ]; then
        echo "| No GPU processes found                                                                  |"
    else
        echo "$GPU_PROCS" | while IFS=',' read -r BUS_ID PID_GPU MEM_GPU; do
            PID_GPU=$(echo $PID_GPU | xargs)
            MEM_GPU=$(echo $MEM_GPU | xargs)

            # Check if this PID belongs to our job
            if echo "$PIDS" | grep -q "^$PID_GPU$"; then
                # Get GPU index from bus ID
                GPU_IDX=$(ssh $NODE "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader 2>/dev/null | grep '$BUS_ID' | cut -d',' -f1" | xargs)
                printf "| GPU: %-2s   PID: %-7s   Type: Python   GPU Memory: %-8s MiB                          |\n" \
                    "$GPU_IDX" "$PID_GPU" "$MEM_GPU"
            fi
        done
    fi

    echo "+-----------------------------------------------------------------------------------------+"
    echo "                                                                                           "
    echo "+-----------------------------------------------------------------------------------------+"
    echo "| CPU & RAM Usage (Main Process):                                                         |"
    echo "+-----------------------------------------------------------------------------------------+"
    printf "| CPU: %6.1f%%   |   Process RAM: %6.1f%% (%5d MB)   |   System RAM: %5s / %5s      |\n" \
        "$CPU" "$MEM_PERCENT" "$RSS_MB" "$RAM_USED" "$RAM_TOTAL"
    echo "+-----------------------------------------------------------------------------------------+"
    echo "                                                                                           "

    # Calculate average GPU utilization and memory
    AVG_GPU_UTIL=$(echo "$GPU_DATA" | awk -F',' '{sum+=$3; count++} END {printf "%.0f", sum/count}')
    AVG_GPU_MEM=$(echo "$GPU_DATA" | awk -F',' '{used+=$4; total+=$5} END {printf "%.0f", (used/total)*100}')

    # Average GPU Utilization bar
    printf "Avg GPU Util [%3s%%]: " "$AVG_GPU_UTIL"
    BAR_LENGTH=50
    FILLED=$((AVG_GPU_UTIL * BAR_LENGTH / 100))
    for ((i=0; i<FILLED; i++)); do printf "█"; done
    for ((i=FILLED; i<BAR_LENGTH; i++)); do printf "░"; done
    printf "   \n"

    # Average GPU Memory bar
    printf "Avg GPU Mem  [%3s%%]: " "$AVG_GPU_MEM"
    FILLED=$((AVG_GPU_MEM * BAR_LENGTH / 100))
    for ((i=0; i<FILLED; i++)); do printf "█"; done
    for ((i=FILLED; i<BAR_LENGTH; i++)); do printf "░"; done
    printf "   \n"

    # CPU bar
    CPU_INT=$(printf "%.0f" "$CPU")
    if [ "$CPU_INT" -gt 100 ]; then CPU_INT=100; fi
    printf "CPU (Main)   [%3s%%]: " "$CPU_INT"
    FILLED=$((CPU_INT * BAR_LENGTH / 100))
    for ((i=0; i<FILLED; i++)); do printf "█"; done
    for ((i=FILLED; i<BAR_LENGTH; i++)); do printf "░"; done
    printf "   \n"

    echo "                                                                                           "
    echo "Press Ctrl+C to stop monitoring                                                           "

    sleep $REFRESH
done