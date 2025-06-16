import pandas as pd
import numpy as np
import os

def sample_csv_rows(input_file, output_file=None, sample_fraction=0.5, random_state=42):
    """
    Randomly sample a fraction of rows from a CSV file.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to save the sampled CSV (optional)
    sample_fraction (float): Fraction of rows to sample (0.5 = 50%)
    random_state (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: The sampled dataframe
    """
    
    print(f"Reading CSV file: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    
    # Calculate number of rows to sample
    n_sample = int(len(df) * sample_fraction)
    print(f"Sampling {n_sample} rows ({sample_fraction*100}%)")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Randomly sample rows
    sampled_df = df.sample(n=n_sample, random_state=random_state)
    
    print(f"Sampled dataset shape: {sampled_df.shape}")
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = input_file.rsplit('.', 1)[0]  # Remove .csv extension
        output_file = f"{base_name}_sampled.csv"
    
    # Save the sampled data
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled data saved to: {output_file}")
    
    # Display basic statistics
    print("\n--- Summary ---")
    print(f"Original rows: {len(df):,}")
    print(f"Sampled rows: {len(sampled_df):,}")
    print(f"Reduction: {len(df) - len(sampled_df):,} rows removed")
    
    return sampled_df

# Main execution
if __name__ == "__main__":
    print("=== CSV Row Sampler Debug Info ===")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List all files in current directory
    print("\nFiles in current directory:")
    files = os.listdir(current_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    for file in files:
        print(f"  - {file}")
    
    print(f"\nCSV files found: {csv_files}")
    
    # Configuration
    INPUT_FILE = "pedagogy_data.csv"
    OUTPUT_FILE = "pedagogy_data_half.csv"  # You can change this name
    
    # Check if the specific file exists
    if os.path.exists(INPUT_FILE):
        print(f"\n✓ Found {INPUT_FILE}")
    else:
        print(f"\n✗ {INPUT_FILE} not found!")
        if csv_files:
            print(f"Available CSV files: {csv_files}")
            print("You might need to rename your file or update the INPUT_FILE variable.")
        else:
            print("No CSV files found in this directory.")
        exit()
    
    try:
        print(f"\n=== Starting to process {INPUT_FILE} ===")
        
        # Sample half the rows randomly
        sampled_data = sample_csv_rows(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            sample_fraction=0.5,  # 50% of the data
            random_state=42  # For reproducible results
        )
        
        print("\n--- First 5 rows of sampled data ---")
        print(sampled_data.head())
        
        print("\n✓ Script completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{INPUT_FILE}'")
        print("Make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()