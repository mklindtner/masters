#!/bin/sh
#BSUB -q gpua100
#BSUB -J parkinVar
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "pkinGrid[1-2]"
#BSUB -o parkinsons_telemonitoring/var_const/singlevar_runs/tmp/gpu_%J_%I.out 
#BSUB -e parkinsons_telemonitoring/var_const/singlevar_runs/tmp/gpu_%J_%I.err


echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"

# ------------------
# --- Default
ITERATIONS=20000
TAU_PARAM=10.0
BATCH_SIZE=256

# --- Poly decay
POLY_A_PARAM=5.00e-6
POLY_B_PARAM=1000.0
POLY_GAMMA_PARAM=0.55

# --- GridSearch for TR


# ------------------
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

case $LSB_JOBINDEX in
    1)  TR_VAR=0.055 ;;
    2)  TR_VAR=0.828 ;;
esac

RUN=2
# RUN_NAME="T${ITERATIONS}_VAR${TR_VAR}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
RUN_NAME="T${ITERATIONS}_VAR${TR_VAR}"
OUTPUT_DIR="parkinsons_telemonitoring/var_const/singlevar_runs/run$RUN/trial$LSB_JOBID_${LSB_JOBINDEX}_${RUN_NAME}"
mkdir -p $OUTPUT_DIR


echo "--- Starting Variance GridSearch Diagonistics ---"
echo "All outputs will be saved to: $OUTPUT_DIR"
echo "Parameters: T=$ITERATIONS, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM"
echo "Running on GPU:"
nvidia-smi
# ------------------


python -m parkinsons_telemonitoring.var_const.parkin_singlevar_train \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --tr_var $TR_VAR \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR



# ------------------
echo "Moving log files to final destination..."
mv parkinsons_telemonitoring/var_const/singlevar_runs/tmp/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.out $OUTPUT_DIR/gpu_output.out
mv parkinsons_telemonitoring/var_const/singlevar_runs/tmp/gpu_${LSB_JOBID}_${LSB_JOBINDEX}.err $OUTPUT_DIR/gpu_error.err
echo "--- Trial $LSB_JOBINDEX Finished ---"