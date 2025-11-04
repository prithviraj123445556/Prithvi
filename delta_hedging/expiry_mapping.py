import os
import pandas as pd

import os
import pandas as pd

# Folder paths
type = 'split'  # update this
input_folder = f'/home/newberry4/jay_test/delta_hedging/NIFTY/ND/Trade_Sheets/{type}/'  # update this
output_folder = f'/home/newberry4/jay_test/delta_hedging/NIFTY/ND/Trade_Sheets/{type}'  # update this
market_dates_file = '/home/newberry4/jay_data/NIFTY market dates 2025 updated.xlsx'  # update this

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load market dates mapping
market_dates_df = pd.read_excel(market_dates_file, parse_dates=['Date', 'ExpiryDate'])

# Convert Date column to datetime64 for proper merging
market_dates_df['Date'] = pd.to_datetime(market_dates_df['Date'])

# Process all trade sheet files
for file in os.listdir(input_folder):
    if file.endswith('.csv') or file.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file)

        # Load trade sheet
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=['Date'])
        else:
            df = pd.read_excel(file_path, parse_dates=['Date'])

        # Ensure Date column is in datetime64 format
        df['Date'] = pd.to_datetime(df['Date'])

        # Merge with market_dates_df to add ExpiryDate, DaysToExpiry, Day
        df = df.merge(market_dates_df[['Date', 'ExpiryDate', 'DaysToExpiry', 'Day']], on='Date', how='left')

        # Save processed file
        output_path = os.path.join(output_folder, file)
        if file.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)

        print(f"Processed and saved: {output_path}")

