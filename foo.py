import pandas as pd

# The filename of the CSV file you want to analyze
#filename = 'experiment/keep/T1000000_tau214_a4.00e-6_b0_g0/MNIST_20250708_111904_T=1000000_M=256_tau=214.0_a=4.0e-06_b=0.0_g=0.0.csv'
filename = 'experiment/single_runs/T1000000_tau400_a5.00e-6_b1000_g0.55_J25548030/MNIST_20250710_193141_T=1000000_M=100_tau=400.0_a=5.0e-06_b=1000.0_g=0.55.csv' 

try:
    # Load the data into a pandas DataFrame
    df = pd.read_csv(filename)

    # --- Teacher NLL Calculation ---
    # Check if the column exists and has enough values
    if 'tr_nll_val' in df.columns and len(df) >= 100:
        # Select the last 100 rows of the 'tr_nll_val' column
        last_100_teacher_nll = df['tr_nll_val'].tail(100)
        
        # Calculate the average of those 100 values
        average_teacher_nll = last_100_teacher_nll.mean()
        
        print(f"Teacher Validation NLL (Average of last 100): {average_teacher_nll:.6f}")
    else:
        print("Could not calculate average teacher NLL. Not enough data or column is missing.")


    # --- Student NLL Calculation ---
    # Check if the column exists and is not empty
    if 'st_nll_val' in df.columns and not df['st_nll_val'].empty:
        # Select the very last value from the 'st_nll_val' column
        final_student_nll = df['st_nll_val'].iloc[-1]
        
        print(f"Student Validation NLL (Final Value):           {final_student_nll:.6f}")
    else:
        print("Could not get final student NLL. Column is missing or empty.")


except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
