import os
import pandas as pd

# Define your directory 
strategy = 'SENSEX_apeksha_new'
stock = 'SENSEX'

tradefolder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/ND/Trade_Sheets/'
output_base = f'/home/newberry4/jay_test/delta_hedging/{strategy}/ND/Output_Sheets_iv/'
os.makedirs(output_base, exist_ok=True)
output_folder = os.path.join(output_base, "aggregated_report")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through files in directory
for fname in os.listdir(tradefolder):
    if fname.endswith('.csv'):
        # File path
        input_file = os.path.join(tradefolder, fname)
        df = pd.read_csv(input_file, parse_dates=['entry_date', 'exit_date', 'expiry_date'])

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
        if 'expiry_date'  in df.columns:
            df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        else :
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
        df = df[df['entry_date'].dt.date == df['expiry_date'].dt.date] 

        time_str = fname.split("time_")[1].split(".csv")[0]  # e.g. "09,30"
        hours, minutes = time_str.split(",")
        filter_time = pd.to_datetime(f'{hours}:{minutes}:00').time()

        # Filter for trades matching this time
        filtered = df[df['entry_date'].dt.time == filter_time]
        print(f"Filtered trades for {filter_time} in {fname}:")
        print(filtered)

        # Filter for 09:30 trades for entry price
        # filtered = df[df['entry_date'].dt.time == pd.to_datetime("09:30:00").time()]
        print(filtered)
        
        # Aggregate by date for entry price
        entry_aggregated = filtered.groupby(filtered['entry_date'].dt.date).apply(
    lambda group: pd.Series({ 
        'ce_entry_price': group.loc[group['type'] == 'CE', 'entry_price'].sum(), 
        'pe_entry_price': group.loc[group['type'] == 'PE', 'entry_price'].sum(), 
        'combined_entry_price': group.loc[group['type'].isin(['CE', 'PE']), 'entry_price'].sum()
    })
).reset_index(names='date')

        # entry_aggregated = entry_aggregated.rename(columns={'entry_date': 'date'})

        # Aggregate total_returns by date (ALL trades)
        returns_aggregated = df.groupby(df['entry_date'].dt.date)['Total_Returns'].sum().reset_index()
        returns_aggregated = returns_aggregated.rename(columns={'entry_date': 'date', 'Total_Returns': 'total_returns'})

        # Combine both
        aggregated = pd.merge(entry_aggregated, returns_aggregated, on='date', how='inner')

        # Save to CSV in output directory with SAME NAME
        output_file = os.path.join(output_folder, fname)
        aggregated.to_csv(output_file, index=False)

        print(f"Aggregation completed for {fname}. File Saved to {output_file}")

print("All files processed successfully.")
