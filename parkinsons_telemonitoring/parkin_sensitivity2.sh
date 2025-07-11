#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "StudentTest[1-300]" 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 05:00 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -B
#BSUB -N

### -- Specify unique output/error files for each job in the array --
# %J is the Job ID, %I is the Array Index. This is critical.
#BSUB -o parkinsons_telemonitoring/full_sweep/logs/gpu_%J_%I.out
#BSUB -e parkinsons_telemonitoring/full_sweep/logs/gpu_%J_%I.err


# --- Shared Hyperparameters ---
ITERATIONS=1000000
TAU_PARAM=15.0
POLY_A_PARAM=5.00e-6
POLY_B_PARAM=1000.0
POLY_GAMMA_PARAM=0.56
VAL_STEP=10000
STUDENT_MODE_PARAM="mean_and_variance" 
BATCH_SIZE=256

# --- Select Student Mode based on the Job Array Index ---
# case $LSB_JOBINDEX in
#     1) STUDENT_MODE_PARAM="mean_and_variance" ;;
#     2) STUDENT_MODE_PARAM="variance_only" ;;
#     3) STUDENT_MODE_PARAM="mean_only" ;;
# esac

JOB_INDEX_ZERO_BASED=$((LSB_JOBINDEX - 1))
TAU_CASE=$((JOB_INDEX_ZERO_BASED / 30 + 1))   # Will result in a number from 1-10
POLY_CASE=$((JOB_INDEX_ZERO_BASED % 30 + 1))  # Will result in a number from 1-30


# --- Select TAU based on TAU_CASE ---
case $TAU_CASE in
    1)  TAU_PARAM=15.0 ;;
    2)  TAU_PARAM=21.2 ;;
    3)  TAU_PARAM=29.9 ;;
    4)  TAU_PARAM=42.3 ;;
    5)  TAU_PARAM=59.6 ;;
    6)  TAU_PARAM=84.2 ;;
    7)  TAU_PARAM=118.9 ;;
    8)  TAU_PARAM=167.9 ;;
    9)  TAU_PARAM=237.1 ;;
    10) TAU_PARAM=300.0 ;;
esac

# --- Select Polynomial Decay Parameter based on POLY_CASE ---
case $POLY_CASE in
    # Group 1: Varying 'a' (Initial LR) from 1e-8 to 1e-6
    1)  POLY_A_PARAM=1.00e-08 ;;
    2)  POLY_A_PARAM=1.67e-08 ;;
    3)  POLY_A_PARAM=2.78e-08 ;;
    4)  POLY_A_PARAM=4.64e-08 ;;
    5)  POLY_A_PARAM=7.74e-08 ;;
    6)  POLY_A_PARAM=1.29e-07 ;;
    7)  POLY_A_PARAM=2.15e-07 ;;
    8)  POLY_A_PARAM=3.59e-07 ;;
    9)  POLY_A_PARAM=5.99e-07 ;;
    10) POLY_A_PARAM=1.00e-06 ;;

    # Group 2: Varying 'b' (Stabilizer) from 100 to 10000
    11) POLY_B_PARAM=100 ;;
    12) POLY_B_PARAM=1200 ;;
    13) POLY_B_PARAM=2300 ;;
    14) POLY_B_PARAM=3400 ;;
    15) POLY_B_PARAM=4500 ;;
    16) POLY_B_PARAM=5600 ;;
    17) POLY_B_PARAM=6700 ;;
    18) POLY_B_PARAM=7800 ;;
    19) POLY_B_PARAM=8900 ;;
    20) POLY_B_PARAM=10000 ;;

    # Group 3: Varying 'gamma' (Decay Rate) from 0.56 to 0.80
    21) POLY_GAMMA_PARAM=0.56 ;;
    22) POLY_GAMMA_PARAM=0.587 ;;
    23) POLY_GAMMA_PARAM=0.613 ;;
    24) POLY_GAMMA_PARAM=0.64 ;;
    25) POLY_GAMMA_PARAM=0.667 ;;
    26) POLY_GAMMA_PARAM=0.693 ;;
    27) POLY_GAMMA_PARAM=0.72 ;;
    28) POLY_GAMMA_PARAM=0.747 ;;
    29) POLY_GAMMA_PARAM=0.773 ;;
    30) POLY_GAMMA_PARAM=0.80 ;;
esac

# --- Create a Unique Output Directory for This Trial ---
RUN_NAME="T${ITERATIONS}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
OUTPUT_DIR="parkinsons_telemonitoring/full_sweep/trial_${LSB_JOBINDEX}_${RUN_NAME}"
mkdir -p $OUTPUT_DIR


# --- Environment Setup ---
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate


# --- Run Diagnostics ---
echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Saving all artifacts to: $OUTPUT_DIR"
echo "Parameters: T=$ITERATIONS, tau=$TAU_PARAM, a=$POLY_A_PARAM, b=$POLY_B_PARAM, g=$POLY_GAMMA_PARAM"
nvidia-smi


# --- Execute the Python Script ---
python -m parkinsons_telemonitoring.parkinsons_training \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --batch_size $BATCH_SIZE \
    --student_mode $STUDENT_MODE_PARAM \
    --val_step $VAL_STEP \
    --output_dir $OUTPUT_DIR


# --- Finalize ---
echo "Moving LSF log files to final destination..."
mv parkinsons_telemonitoring/full_sweep/logs/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.out $OUTPUT_DIR/hpc_job_output.out 
mv parkinsons_telemonitoring/full_sweep/logs/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.err $OUTPUT_DIR/hpc_job_error.err  
echo "--- Test Run Finished ---"