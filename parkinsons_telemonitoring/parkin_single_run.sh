#!/bin/sh
#BSUB -q gpua100
#BSUB -J parkinSingle
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 07:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -o parkinsons_telemonitoring/SGLD_single_run/gpu_%J.out
#BSUB -e parkinsons_telemonitoring/SGLD_single_run/gpu_%J.err

# ------------------
ITERATIONS=1000000
TAU_PARAM=15.0
POLY_A_PARAM=6.00e-6 #5.00e-6 prev
POLY_B_PARAM=1000.0
POLY_GAMMA_PARAM=0.56
ST_TYPE="mean_and_variance"
VAL_STEP=10000

RUN_NAME="T${ITERATIONS}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
OUTPUT_DIR="parkinsons_telemonitoring/SGLD_single_run/$RUN_NAME"


# ------------------
mkdir -p $OUTPUT_DIR

module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

echo "--- Starting Single Test Run ---"
echo "All outputs will be saved to: $OUTPUT_DIR"
echo "Parameters: T=$ITERATIONS, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM"
echo "Running on GPU:"
nvidia-smi
# ------------------


python -m parkinsons_telemonitoring.parkinsons_training \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --student_mode $ST_TYPE \
    --val_step $VAL_STEP \
    --output_dir $OUTPUT_DIR



# ------------------
echo "Moving log files to final destination..."
mv parkinsons_telemonitoring/SGLD_single_run/gpu_$LSB_JOBID.out $OUTPUT_DIR/gpu_output.out
mv parkinsons_telemonitoring/SGLD_single_run/gpu_$LSB_JOBID.err $OUTPUT_DIR/gpu_error.err
echo "--- Single Test Run Finished ---"