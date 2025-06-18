# compile_results.py

import os
import json
import pandas as pd
import glob

def compile_ablation_results(base_dir="parkinsons_telemonitoring/SGLD_ablation"):
    """
    Finds all summary.json files, compiles them into a clean, report-ready
    DataFrame with finalized column names, and saves a master CSV file.
    """
    
    cwd = os.getcwd()
    search_path = os.path.join(cwd, base_dir)
    summary_files = glob.glob(os.path.join(search_path, "**", "summary.json"), recursive=True)
    
    if not summary_files:
        print(f"No summary.json files found in '{search_path}'.")
        return

    print(f"Found {len(summary_files)} trial summaries. Compiling and formatting results...")
    
    all_results = []
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not read file {summary_file}. Error: {e}")
            
    if not all_results:
        print("Could not load any valid results. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    
    # Round the NLL column to 5 decimal places using its original name
    results_df['avg_nll_post_burn_in'] = pd.to_numeric(
        results_df['avg_nll_post_burn_in'], errors='coerce'
    ).round(5)
    
    # --- NEW: Rename columns for the final report ---
    column_rename_map = {
        'avg_nll_post_burn_in': 'NLL_avg',
        'burn_in_B': 'B',
        'tr_lr': 'lr',
        'weight_decay': 'tau'
    }
    report_df = results_df.rename(columns=column_rename_map)

    # --- MODIFIED: Define final column order using the new names ---
    report_columns = [
        'T-B', 
        'batch_size', 
        'lr', 
        'tau', 
        'B', 
        'NLL_avg'
    ]
    
    # Select and reorder the DataFrame to match the report format
    report_df = report_df.reindex(columns=report_columns)
    
    # Sort the results by the performance metric (using its new name)
    report_df = report_df.sort_values(by='NLL_avg', ascending=True)
    
    # Save the master CSV file
    master_csv_path = os.path.join(search_path, "master_ablation_summary_final.csv")
    report_df.to_csv(master_csv_path, index=False)
    
    print(f"\nFinal report summary saved to: {master_csv_path}")
    print("\n--- Top 5 Performing Trials (Final Report Format) ---")
    print(report_df.head(5).to_string())

if __name__ == '__main__':
    compile_ablation_results()