import pandas as pd

# The filename of the specific run we are analyzing
filename = 'experiment/MNIST_20250625_132929_T=200000_M=256_tau=440.0_a=2.2e-05_b=1150.0_g=0.55.csv'

try:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # 1. Find the absolute minimum value in the 'tr_nll' (validation) column
    min_validation_nll = df['tr_nll'].min()

    # 2. Calculate the average over the final 20% of the run
    # Find the starting index for the last 20% of the data points
    start_index = int(len(df) * 0.8)
    
    # Select all rows from the start index to the end and calculate the mean
    final_20_percent_avg = df['tr_nll'].iloc[start_index:].mean()

    print(f"--- Analysis of: {filename} ---")
    print(f"The absolute lowest validation NLL achieved was: {min_validation_nll:.6f}")
    print(f"The average validation NLL over the final 20% was: {final_20_percent_avg:.6f}")

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please ensure the script is run from the correct directory.")