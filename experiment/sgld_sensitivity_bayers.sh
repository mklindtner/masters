#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 7:30
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -J "decayStudy[1-10]"
#BSUB -o experiment/tmp/polydecay/abl_out_%J_%I.out
#BSUB -e experiment/tmp/polydecay/abl_err_%J_%I.err

echo "--- Setting up environment for Job Array Index: $LSB_JOBINDEX ---"
module load cuda/12.2
module load python3/3.11.9
source /zhome/25/e/155273/masters/hpc_venv/bin/activate

# --- Hyperparameter Selection ---
ITERATIONS=1000000
TAU_PARAM=1
POLY_A_PARAM=4.00e-6
POLY_B_PARAM=0
POLY_GAMMA_PARAM=0
BATCH_SIZE=100

# Use a 'case' statement to override one parameter
case $LSB_JOBINDEX in
    1)  TAU_PARAM=1 ;;
    2)  TAU_PARAM=10 ;;
esac

RUN_NAME="T${ITERATIONS}_tau${TAU_PARAM}_a${POLY_A_PARAM}_b${POLY_B_PARAM}_g${POLY_GAMMA_PARAM}"
OUTPUT_DIR="experiment/sensitivity/run7/$RUN_NAME"
mkdir -p $OUTPUT_DIR

echo "--- Starting Trial $LSB_JOBINDEX ---"
echo "Saving all artifacts to: $OUTPUT_DIR"
echo "Parameters: T=1e6, tau=$TAU_PARAM, poly_a=$POLY_A_PARAM, poly_b=$POLY_B_PARAM, poly_g=$POLY_GAMMA_PARAM"

python /zhome/25/e/155273/masters/experiment/experiment1_sensitivity_study.py \
    --iterations $ITERATIONS \
    --tau $TAU_PARAM \
    --tr_poly_a $POLY_A_PARAM \
    --tr_poly_b $POLY_B_PARAM \
    --tr_poly_gamma $POLY_GAMMA_PARAM \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR

echo "--- Python Script Finished ---"

echo "Moving log files to final destination..."
mv experiment/tmp/polydecay/abl_out_${LSB_JOBID}_${LSB_JOBINDEX}.out $OUTPUT_DIR/run_output.out
mv experiment/tmp/polydecay/abl_err_${LSB_JOBID}_${LSB_JOBINDEX}.err $OUTPUT_DIR/run_error.err
echo "--- Trial $LSB_JOBINDEX Fully Completed ---"