import pandas as pd # Import pandas
import torch # Ensure torch is available for type hints or checks if needed
import os
from toydata import T, H

class StudentLogger:
    def __init__(self, log_filepath="student_training_log.csv", review_step=H, T=T):
        self.log_filepath = log_filepath
        self.log_records = [] # Initialize an empty list to store log dictionaries
        self.columns = [
            "g_id",
            "StudentUpdateStep", "SGLDIteration", "Loss",
            "w0_Bias", "grad_w0", "w1_Weight", 
            "grad_w1"
        ]

        self.reviews = [x for x in range(T) if x == 0 or x % review_step == 0]
    
    #For multiple g 
    def logger_step(self, g_id, sgld_step,st_step, loss, w0, w0_grad, w1, w1_grad): 
        if st_step not in self.reviews:
            return None
        
        grad_w0_str = f"{w0_grad:.6e}"
        grad_w1_str = f"{w1_grad:.6e}"
        loss_str = f"{loss.item():.6e}"
        foo = [g_id, st_step, sgld_step, float(loss_str), w0, float (grad_w0_str), w1, float(grad_w1_str)]
        record = dict(zip(self.columns, foo))
        self.log_records.append(record)


    #Should be depcecated
    def add_student_weight(self, student_weight, bias, weight):
        student_weight[0,0] = bias
        student_weight[0,1] = weight

        # logger.student_step5[0,0] = f.fc1.bias.detach().clone()
        #             logger.student_step5[0,1] = f.fc1.weight.detach().clone()


    #For single g
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
        self.save_to_csv()
        self.log_records = [] # Clear records after saving (optional, good for re-use)

    # For using with 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

log_filename = os.path.join(os.getcwd(),"toy_example/distillation_scalable_debug_df.csv") 

#Debugging
student_step5 = torch.empty((1,2))
student_step50 =  torch.empty((1, 2))
student_step1000 = torch.empty((1,2))
student_step2500 = torch.empty((1,2))
student_step5000 = torch.empty((1,2))    
