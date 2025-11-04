import pandas as pd
import os

import os
import pandas as pd



# Parameters
stock = 'NIFTY'
strategy = 'NIFTY'
# input_file_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_premium_timewise.xlsx'
input_file_path = f'jay_test/delta_hedging/{stock}/{stock}_premium_timewise_new_selected.xlsx'
dailypnl_base_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
output_file_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_pnl_final_PREMIUM.csv'

# Load the input Excel file (it has multiple sheets for different times)
non_rentry_df = pd.read_excel(input_file_path, sheet_name=None)  # sheet_name=None loads all sheets

# Initialize an empty DataFrame for the final output
final_df = pd.DataFrame()

# Iterate through each sheet (time-based) in the Excel file
for sheet_name, sheet_data in non_rentry_df.items():
    # Initialize the DataFrame for the current sheet
    print(f"Processing sheet: {sheet_name}")
    
    # Iterate through the rows of the sheet (this represents the strategies)
    for index, row in sheet_data.iterrows():
        strategy_file = row['File Name']  # Portfolio file name
        file_set = row['File Set'].split('_set')[0]  # Extract the strategy folder name
        
        # Construct the folder path where the portfolio file is stored
        strategy_folder_path = os.path.join(dailypnl_base_path, file_set)
        strategy_file_path = os.path.join(strategy_folder_path, strategy_file)
        
        print(f"Looking for file: {strategy_file_path}")
        
        try:
            # Load the strategy portfolio file (daily PnL)
            temp_df = pd.read_csv(strategy_file_path)
            temp_df = temp_df.reset_index()
            
            # For the first file in each sheet, initialize the 'Date' column in final_df
            if index == 0 and sheet_name == list(non_rentry_df.keys())[0]:  # First time sheet
                final_df['Date'] = temp_df['Date'].copy()
            
            # Create a new column name as "file_set_file_name" and add the PnL data
            column_name = f"{file_set}_{strategy_file}"
            final_df[column_name] = temp_df['Daily PnL']
            
        except Exception as e:
            print(f"Error processing file {strategy_file_path}: {e}")
            continue

# Save the final DataFrame to a CSV file
final_df.to_csv(output_file_path, index=False)
print(f"Output saved to {output_file_path}")





