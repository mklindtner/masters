#!/bin/sh
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:30  # Walltime per trial (adjust if 100k iterations takes longer)
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "parkinSens[1-26]"
#BSUB -o parkinsons_telemonitoring/SGLD_ablation/logs/abl_%J_%I.out
#BSUB -e parkinsons_telemonitoring/SGLD_ablation/logs/abl_%J_%I.err

echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"

mkdir -p parkinsons_telemonitoring/SGLD_ablation/logs
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Hyperparameter Selection ---
LR_PARAM=4e-6
TAU_PARAM=10.0
BS_PARAM=64

# Use a 'case' statement to override one parameter based on the job index
case $LSB_JOBINDEX in
    1)
        # Trial 1: BASELINE RUN - uses the default parameters set above
        echo "Running BASELINE trial"
        ;;
    # --- GROUP 1: Varying Learning Rate (lr) ---
    2)  LR_PARAM=1.00e-02 ;;
    3)  LR_PARAM=3.98e-03 ;;
    4)  LR_PARAM=1.58e-03 ;;
    5)  LR_PARAM=6.31e-04 ;;
    6)  LR_PARAM=2.51e-04 ;;
    7)  LR_PARAM=1.00e-04 ;;
    8)  LR_PARAM=3.98e-05 ;;
    9)  LR_PARAM=1.58e-05 ;;
    10) LR_PARAM=6.31e-06 ;;
    11) LR_PARAM=8.00e-06 ;; # Closest log-step to your requested end-point
    
    # --- GROUP 2: Varying Prior Precision (tau) ---
    12) TAU_PARAM=0.0100 ;;
    13) TAU_PARAM=0.0215 ;;
    14) TAU_PARAM=0.0464 ;;
    15) TAU_PARAM=0.1000 ;;
    16) TAU_PARAM=0.2154 ;;
    17) TAU_PARAM=0.4642 ;;
    18) TAU_PARAM=1.0000 ;;
    19) TAU_PARAM=2.1544 ;;
    20) TAU_PARAM=4.6416 ;;
    21) TAU_PARAM=10.0000 ;;

    # --- GROUP 3: Varying Batch Size (M) ---
    22) BS_PARAM=16 ;;
    23) BS_PARAM=32 ;;
    24) BS_PARAM=64 ;;
    25) BS_PARAM=128 ;;
    26) BS_PARAM=256 ;;
esac

echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Parameters: T=100000, lr=$LR_PARAM, batch_size=$BS_PARAM, tau=$TAU_PARAM"

python -m parkinsons_telemonitoring.parkinsons_SGLD_ablation \
    --T 100000 \
    --batch_size $BS_PARAM \
    --tr_lr $LR_PARAM \
    --weight_decay $TAU_PARAM

echo "--- Trial $LSB_JOBINDEX Finished ---"