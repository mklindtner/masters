import pandas as pd # Import pandas
import torch # Ensure torch is available for type hints or checks if needed
import os

class StudentLogger:
    def __init__(self, log_filepath="student_training_log.csv"):
        self.log_filepath = log_filepath
        self.log_records = [] # Initialize an empty list to store log dictionaries
        self.columns = [
            "StudentUpdateStep", "SGLDIteration", "Loss",
            "w0_Bias", "grad_w0", "w1_Weight", "grad_w1"
        ]

    def log_step(self, student_update_step, sgld_iteration, loss,
                 w0_bias, grad_w0, w1_weight, grad_w1):
        """Stores a single step of student training data as a dictionary."""
        
        # Handle potential None for gradients before formatting
        grad_w0_str = f"{grad_w0:.6e}" if grad_w0 is not None else None # Store as float or None
        grad_w1_str = f"{grad_w1:.6e}" if grad_w1 is not None else None # Store as float or None

        record = {
            "StudentUpdateStep": student_update_step,
            "SGLDIteration": sgld_iteration,
            "Loss": loss, # Store as float or NaN
            "w0_Bias": w0_bias,
            "grad_w0": float(grad_w0_str), # Convert formatted string back to float
            "w1_Weight": w1_weight,
            "grad_w1": float(grad_w1_str) # Convert formatted string back to float
        }
        self.log_records.append(record)

    def get_dataframe(self):
        """Returns the collected log records as a pandas DataFrame."""
        if not self.log_records:
            return pd.DataFrame(columns=self.columns) # Return empty DataFrame with columns if no records
        return pd.DataFrame(self.log_records, columns=self.columns)

    def save_to_csv(self):
        """Converts collected records to a DataFrame and saves to CSV."""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(self.log_filepath, index=False, float_format='%.6e') # Use scientific notation for floats
            print(f"Log data saved to '{self.log_filepath}'")
        else:
            print(f"No data to save for '{self.log_filepath}'. Empty log file created with headers (if new).")
            # Optionally, create an empty file with just headers if df is empty
            pd.DataFrame(columns=self.columns).to_csv(self.log_filepath, index=False)


    def close(self):
        """Saves the collected data to CSV when closing."""
        self.save_to_csv()
        self.log_records = [] # Clear records after saving (optional, good for re-use)

    # For using with 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

log_filename = os.path.join(os.getcwd(),"toy_example/distillation_debug_df.csv") 