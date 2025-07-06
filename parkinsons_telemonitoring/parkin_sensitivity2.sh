#!/bin/sh
#BSUB -q gpua100
#BSUB -J "StudentTest[1-3]" 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -B
#BSUB -N

### -- Specify unique output/error files for each job in the array --
# %J is the Job ID, %I is the Array Index. This is critical.
#BSUB -o parkinsons_telemonitoring/SGLD_sensitivity/logs/gpu_%J_%I.out
#BSUB -e parkinsons_telemonitoring/SGLD_sensitivity/logs/gpu_%J_%I.err


# --- Shared Hyperparameters ---
ITERATIONS=1000000
TAU_PARAM=100.0
POLY_A_PARAM=5.00e-6
POLY_B_PARAM=1000.0
POLY_GAMMA_PARAM=0.55
VAL_STEP=10000

# --- Select Student Mode based on the Job Array Index ---
case $LSB_JOBINDEX in
    1) STUDENT_MODE_PARAM="mean_and_variance" ;;
    2) STUDENT_MODE_PARAM="variance_only" ;;
    3) STUDENT_MODE_PARAM="mean_only" ;;
esac


# --- Create a Unique Output Directory for This Trial ---
RUN_NAME="T${ITERATIONS}_tau${TAU_PARAM}"
OUTPUT_DIR="parkinsons_telemonitoring/SGLD_sensitivity/trial_${LSB_JOBINDEX}_${STUDENT_MODE_PARAM}"
mkdir -p $OUTPUT_DIR/logs # Create log sub-directory


# --- Environment Setup ---
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate


# --- Run Diagnostics ---
echo "--- Starting Test Run for Student Mode: $STUDENT_MODE_PARAM ---"
echo "Job Index: $LSB_JOBINDEX"
echo "All outputs will be saved to: $OUTPUT_DIR"
nvidia-smi


# --- Execute the Python Script ---
python -m parkinsons_telemonitoring.parkinsons_training \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --student_mode $STUDENT_MODE_PARAM \
    --val_step $VAL_STEP \
    --output_dir $OUTPUT_DIR


# --- Finalize ---
echo "Moving LSF log files to final destination..."
mv parkinsons_telemonitoring/SGLD_sensitivity/logs/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.out $OUTPUT_DIR/hpc_job_output.out 
mv parkinsons_telemonitoring/SGLD_sensitivity/logs/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.err $OUTPUT_DIR/hpc_job_error.err  
echo "--- Test Run Finished ---"