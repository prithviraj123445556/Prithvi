### for only non rentry
import pandas as pd
import numpy as np
import psycopg2 as sql
from datetime import datetime, timedelta
import os, sys
import os
import multiprocessing
import pandas as pd
import time
from datetime import datetime
from functools import partial
from functools import partial
from pandarallel import pandarallel
# sys.path.insert(0, r"/home/newberry3/")

from running_pnl_2 import load_options_data, process_file, params, modify_ticker
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
#################################################################################################
stock = 'NIFTY'
# superset = 'COMBINED_WITHOUT_REENTRY'
strategy = 'NIFTY_non_split'
# strategy = 'NIFTY_split'

LOT_SIZE = 75 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (25 if stock == 'FINNIFTY' else (20 if stock == 'SENSEX' else None)))

start_date = '2021-06-01'
end_date = '2025-04-30'

root_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/'
output_folder_path = f'{root_path}/{stock}/1_min_pnl//'
# folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/ND/Trade_Sheets/' 

folder_path = f'/home/newberry4/jay_test/delta_hedging/advance_analytics_split_non_split/tradesheet/'

# expiry_file_path = rf"/home/newberry4/jay_data/Common_Files/{stock} market dates.xlsx"

if stock == 'NIFTY':
    expiry_file_path = rf"/home/newberry4/jay_data/NIFTY market dates 2025 updated.xlsx"
else:
    expiry_file_path = rf"/home/newberry4/jay_data/SENSEX market dates updated 2025.xlsx"

# option_file_path = rf"/home/newberry4/jay_data/Data/NIFTY/NIFTY_OHLCV/NIFTY_OHLCV/"



if stock == 'NIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/Data/{stock}/Current_Expiry/"
    option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/Current_Expiry_OI_OHLC/"
    # option_data_path2 = rf"/home/newberry4/jay_data/Data/NIFTY/2ndweeknext_Expiry_OHLC/"
    # option_data_path2 = rf"/home/newberry4/jay_data/Data/NIFTY/2ndweeknext_Expiry/"
elif stock =='BANKNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/BANKNIFTY_DATA/BANKNIFTY_OHLCV/"
    option_data_path = rf"/home/newberry4/jay_data/Data/BANKNIFTY/monthly_expiry_OI/"
elif stock =='FINNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/FINNIFTY_2/"
    option_data_path = rf"/home/newberry4/jay_data/Data/FINNIFTY/monthly_expiry/"
elif stock =='SENSEX':
    # option_data_path = rf"jay_data/Data/SENSEX/weekly_expiry/"
    option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OHLC/"



os.makedirs(output_folder_path, exist_ok=True)
#################################################################################################

# lots_df = pd.read_excel(f'/home/newberry4/jay_test/SHORT_STRADDLE/1_min_no_rentry/{stock}_premium_lots.xlsx')
# lots_df['Strategy'] = lots_df['Strategy'].str.replace('_p.xlsx', '.csv')

# ########################################## FUNCTIONS ############################################


import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict



import pandas as pd
from collections import defaultdict

from collections import defaultdict
import pandas as pd











import pandas as pd
from collections import defaultdict
import numpy as np

import pandas as pd
from collections import defaultdict
import numpy as np

import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict



# def generate_1min_pnl_with_hedge(trades_df, option_data_df, lot_size, TIME_FRAME='30T'):
#     """
#     Generate PnL for trades, including both primary and hedge legs, and carry forward the PnL until expiry.
    
#     Parameters:
#     - trades_df: DataFrame containing trade info (CE/PE short + hedge)
#     - option_data_df: 1-min OHLC option data ['DateTime', 'Type', 'StrikePrice', 'High']
#     - lot_size: Instrument lot size (e.g., 50 for NIFTY)
#     - TIME_FRAME: Timeframe for resampling, default is 30 minutes ('30T')
    
#     Returns:
#     - final_df: DataFrame with ['DateTime', 'CE pnl', 'PE pnl', 'PnL Combined']
#     """
#     pnl_dict = {'CE pnl': defaultdict(float), 'PE pnl': defaultdict(float)}

#     # Group trades by expiry date
#     grouped_trades = trades_df.groupby('ExpiryDate')

#     # Iterate over each expiry date group
#     for expiry_date, group in grouped_trades:
#         # Sort trades within the expiry group by Entry Time
#         group = group.sort_values(by='Entry Time')

#         # Get the start time from the first trade's entry time (9:30 AM for the first trade)
#         start_time = pd.to_datetime(group.iloc[0]['Entry Time'])  # Entry time of the first trade
#         expiry_time = pd.to_datetime(expiry_date) + pd.Timedelta(hours=15, minutes=30)  # Expiry at 15:30
#         # intervals = pd.date_range(start=start_time, end=expiry_time, freq=TIME_FRAME)

#         # Initialize PnL for each interval from start_time to expiry_time
#         # for interval in intervals:
#         #     pnl_dict['CE pnl'][interval] = 0
#         #     pnl_dict['PE pnl'][interval] = 0

#         last_pnl = 0
#         last_time = None

#         # Process each trade within the expiry group
#         for _, trade in group.iterrows():
#             try:
#                 # Convert all date-related columns to datetime
#                 entry_time = pd.to_datetime(trade['Entry Time'])
#                 exit_time = pd.to_datetime(trade['Exit Time'])
#                 hedge_entry_time = pd.to_datetime(trade['Hedge Entry Time'])
#                 hedge_exit_time = pd.to_datetime(trade['Hedge Exit Time'])
#                 expiry_time = pd.to_datetime(trade['ExpiryDate']) + pd.Timedelta(hours=15, minutes=30)  # Expiry at 15:30

#                 # Extract trade details
#                 main_type = trade['Option']
#                 hedge_type = trade['Hedge Option']
#                 main_strike = int(trade['Strike'])
#                 hedge_strike = int(trade['Hedge Strike'])
#                 entry_price = float(trade['Entry Premium'])
#                 hedge_entry_price = float(trade['Hedge Entry Premium'])

#                 # Main leg data between trade entry date and exit date
#                 main_leg_data = option_data_df[
#                     (option_data_df['Type'] == main_type) &
#                     (option_data_df['StrikePrice'] == main_strike) &
#                     (option_data_df['DateTime'] >= entry_time) &
#                     (option_data_df['DateTime'] <= exit_time).sort_index(inplace=True)  # Filter till exit time
#                 ]
#                 main_leg_data['DateTime'] = pd.to_datetime(main_leg_data['DateTime'])
#                 # Commenting the resample code to use 1-minute data
#                 # main_leg_data = main_leg_data.resample(TIME_FRAME, on='DateTime').agg({
#                 #     'Open': 'first', 
#                 #     'Close': 'last', 
#                 #     'High': 'max', 
#                 #     'Low': 'min'
#                 # }).dropna()

#                 # Hedge leg data between hedge entry date and exit date
#                 hedge_leg_data = option_data_df[
#                     (option_data_df['Type'] == hedge_type) &
#                     (option_data_df['StrikePrice'] == hedge_strike) &
#                     (option_data_df['DateTime'] >= hedge_entry_time) &
#                     (option_data_df['DateTime'] <= hedge_exit_time).sort_index(inplace=True)  # Filter till exit time
#                 ]
#                 hedge_leg_data['DateTime'] = pd.to_datetime(hedge_leg_data['DateTime'])
#                 # Commenting the resample code to use 1-minute data
#                 # hedge_leg_data = hedge_leg_data.resample(TIME_FRAME, on='DateTime').agg({
#                 #     'Open': 'first', 
#                 #     'Close': 'last', 
#                 #     'High': 'max', 
#                 #     'Low': 'min'
#                 # }).dropna()

#                 # Calculate PnL for the main leg (short leg)
#                 for _, row in main_leg_data.iterrows():
#                     pnl = (entry_price - row['Open']) * lot_size
#                     key = 'CE pnl' if main_type == 'CE' else 'PE pnl'
#                     pnl_dict[key][row.name] += round(pnl, 2)

#                 # Calculate PnL for the hedge leg
#                 for _, row in hedge_leg_data.iterrows():
#                     pnl = (row['Open'] - hedge_entry_price) * lot_size
#                     key = 'CE pnl' if hedge_type == 'CE' else 'PE pnl'
#                     pnl_dict[key][row.name] += round(pnl, 2)

#                 # Track the last PnL and time to carry forward PnL to expiry if not yet closed
#                 for _, row in main_leg_data.iterrows():     
#                     last_pnl = pnl_dict['CE pnl'][row.name] if main_type == 'CE' else pnl_dict['PE pnl'][row.name]
#                     last_time = row.name

#                 # Skip if last_time is equal to expiry time (no need to carry forward)
#                 if last_time == expiry_time:
#                     continue

#                 # Carry forward PnL to expiry (if not already closed)
#                 if last_time is not None:
#                     carry_forward = option_data_df[
#                         (option_data_df['DateTime'] > last_time) &
#                         (option_data_df['DateTime'] <= expiry_time) &
#                         ((option_data_df['Type'] == main_type) | (option_data_df['Type'] == hedge_type))
#                     ]

#                     carry_forward = carry_forward.sort_values(by='DateTime')
#                     for dt in carry_forward['DateTime'].unique():
#                         pnl_dict[key][dt] += last_pnl

#             except Exception as e:
#                 print(f"Skipping row due to error: {e}")
#                 continue

#         # Create a final DataFrame with the summed PnLs for each interval in the expiry group
#         result_df = pd.DataFrame(pnl_dict['CE pnl'].items(), columns=['DateTime', 'CE pnl'])
#         result_df['PE pnl'] = pd.Series(pnl_dict['PE pnl']).reindex(result_df['DateTime']).fillna(0)
#         result_df['PnL Combined'] = result_df['CE pnl'] + result_df['PE pnl']

#         # Sort by DateTime to get the correct sequence of intervals
#         result_df.sort_values(by='DateTime', inplace=True)

#     return result_df



import pandas as pd
from collections import defaultdict

def generate_1min_pnl_with_hedge(trades_df, option_data_df, lot_size, TIME_FRAME='30T'):
    """
    Generate PnL for trades, including both primary and hedge legs, and carry forward the PnL until expiry.
    
    Parameters:
    - trades_df: DataFrame containing trade info (CE/PE short + hedge)
    - option_data_df: 1-min OHLC option data ['DateTime', 'Type', 'StrikePrice', 'High']
    - lot_size: Instrument lot size (e.g., 50 for NIFTY)
    - TIME_FRAME: Timeframe for resampling, default is 30 minutes ('30T')
    
    Returns:
    - final_df: DataFrame with ['DateTime', 'CE pnl', 'PE pnl', 'PnL Combined']
    """
    
    pnl_dict = {'CE pnl': defaultdict(float), 'PE pnl': defaultdict(float)}

    # Ensure 'DateTime' is in the correct datetime format for both DataFrames
    # option_data_df['DateTime'] = pd.to_datetime(option_data_df['DateTime'], errors='coerce')
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades_df['Hedge Entry Time'] = pd.to_datetime(trades_df['Hedge Entry Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades_df['Hedge Exit Time'] = pd.to_datetime(trades_df['Hedge Exit Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    # trades_df['ExpiryDate'] = pd.to_datetime(trades_df['ExpiryDate'],format='%Y-%m-%d %H:%M', errors='coerce').dt.date
    # print(trades_df['ExpiryDate'].head())
    # print("trades_df", trades_df['ExpiryDate'])
    # Group trades by expiry date
    grouped_trades = trades_df.groupby('ExpiryDate')
    # print("grouped_trades",grouped_trades)
    # Iterate over each expiry date group

    for expiry_date, group in grouped_trades:
        # Sort trades within the expiry group by Entry Time
        print("expiry_date", expiry_date)
        group = group.sort_values(by='Entry Time')
        # print("group", group)

    #     # Get the start time from the first trade's entry time (9:30 AM for the first trade)
    #     start_time = pd.to_datetime(group.iloc[0]['Entry Time'])  # Entry time of the first trade
        expiry_time = pd.to_datetime(expiry_date, format='%d-%m-%Y') + pd.Timedelta(hours=15, minutes=30)
        # expiry_time_str = expiry_time.strftime('%d-%m-%Y %H:%M')
        # expiry_time = pd.to_datetime(expiry_time, format='%d-%m-%Y %H:%M', errors='coerce')
  # Expiry at 15:30

    #     last_pnl = 0
    #     last_time = None

        # Process each trade within the expiry group
        for _, trade in trades_df.iterrows():
            try:
                # print("trade", trade)
                # Convert all date-related columns to datetime
                entry_time = trade['Entry Time']
                entry_date = trade['Date']
                # print("entry_date", entry_date)
                exit_time = trade['Exit Time']
                # option_data_df['DateTime'] = pd.to_datetime(option_data_df['DateTime'])
                hedge_entry_time = pd.to_datetime(trade['Hedge Entry Time'])
                hedge_exit_time = pd.to_datetime(trade['Hedge Exit Time'])
                # expiry_time = pd.to_datetime(trade['ExpiryDate']) + pd.Timedelta(hours=15, minutes=30)  # Expiry at 15:30

                # print("expiry_time", expiry_time) 
                # print("entry_time", entry_time)   
                # print("exit_time", exit_time) 

                # Ensure all times are in the correct format
                # Extract trade details
                main_type = trade['Option']
                hedge_type = trade['Hedge Option']
                main_strike = trade['Strike']
                # print("main_strike", main_strike)
                hedge_strike = trade['Hedge Strike']
                expiry_date = trade['ExpiryDate']
                entry_price = float(trade['Entry Premium'])
                hedge_entry_price = float(trade['Hedge Entry Premium'])
                main_lots = trade['Quantity']
                hedge_lots = trade['Hedge Quantity']
                # print("expiry_date", expiry_date)
                # print("main_type", main_type)

                print("entry_time", entry_time)
                print("exit_time", exit_time)
                # print("option_data_df['DateTime'].dtype", option_data_df['DateTime'].dtype)

                # Main leg data between trade entry date and exit date
                # print("option_data_df", option_data_df)
                main_leg_data = option_data_df[
                    (option_data_df['Type'] == main_type) &
                    (option_data_df['StrikePrice'] == main_strike) &
                    (option_data.index >= entry_time) &
                    (option_data.index <= exit_time)   # Filter till exit time
                ].copy()
                # print("main_leg_data", main_leg_data)
                # main_leg_data['DateTime'] = pd.to_datetime(main_leg_data['DateTime'], errors='coerce')
                main_leg_data = main_leg_data.sort_index()
                
                print("main_leg_data", main_leg_data)
                # exit()
                # Hedge leg data between hedge entry date and exit date
                hedge_leg_data = option_data_df[
                    (option_data_df['Type'] == hedge_type) &
                    (option_data_df['StrikePrice'] == hedge_strike) &
                    (option_data_df.index >= hedge_entry_time) &
                    (option_data_df.index <= hedge_exit_time) # Filter till exit time
                ].copy()

                # hedge_leg_data['DateTime'] = pd.to_datetime(hedge_leg_data['DateTime'], errors='coerce')
                hedge_leg_data = hedge_leg_data.sort_index()
                # print("hedge_leg_data", hedge_leg_data)
                
                # Calculate PnL for the main leg (short leg)
                
                # Calculate PnL for the main leg (short leg)
                for _, row in main_leg_data.iterrows():
                    pnl = (entry_price - row['Open']) * lot_size * abs(main_lots)
                    key = 'CE pnl' if main_type == 'CE' else 'PE pnl'
                    pnl_dict[key][row.name] += round(pnl, 2)  # Use row.name as the DateTime index

                # Calculate PnL for the hedge leg
                for _, row in hedge_leg_data.iterrows():
                    pnl = (row['Open'] - hedge_entry_price) * lot_size * abs(hedge_lots)
                    key = 'CE pnl' if hedge_type == 'CE' else 'PE pnl'
                    pnl_dict[key][row.name] += round(pnl, 2)

                # Track the last PnL and time to carry forward PnL to expiry if not yet closed
                for _, row in main_leg_data.iterrows():
                    last_pnl = pnl_dict['CE pnl'][row.name] if main_type == 'CE' else pnl_dict['PE pnl'][row.name]
                    last_time = row.name 
                    
                
                print("last_pnl", last_pnl) 
                # Skip if last_time is equal to expiry time (no need to carry forward)
                if last_time == expiry_time:
                    continue
                
                print("last_time", last_time)
                print("expiry_time", expiry_time)
                # print("last_time", type(last_time))
                # print("expiry_time", type(expiry_time))
                # Carry forward PnL to expiry (if not already closed)
                if last_time is not None:
                    carry_forward = option_data_df[
                        (option_data_df.index > last_time) &  # Using the index (DateTime) directly
                        (option_data_df.index <= expiry_time) &
                        ((option_data_df['Type'] == main_type) | (option_data_df['Type'] == hedge_type))
                    ]
                    print("carry_forward", carry_forward)
                    # carry_forward['DateTime'] = pd.to_datetime(carry_forward['DateTime'], errors='coerce')
                    
                    carry_forward = carry_forward.sort_index()
                    # print("carry_forward", carry_forward['ExpiryDate'])
                    for dt in carry_forward.index.unique():  # Accessing DateTime as index directly
                        if dt not in pnl_dict[key]:
                            pnl_dict[key][dt] = last_pnl  # Carry forward PnL to this time step
                        else:
                            pnl_dict[key][dt] += last_pnl 

                    # print(pnl_dict)

            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

        # Create the final DataFrame with the summed PnLs for each interval in the expiry group
        ce_df = pd.DataFrame(pnl_dict['CE pnl'].items(), columns=['DateTime', 'CE pnl'])
        
        pe_df = pd.DataFrame(pnl_dict['PE pnl'].items(), columns=['DateTime', 'PE pnl'])

        # Set 'DateTime' as the index for both dataframes (if it's not already the index)
        # Set 'DateTime' as the index for both dataframes (if it's not already the index)
        ce_df.set_index('DateTime', inplace=True)
        pe_df.set_index('DateTime', inplace=True)

        # Sort both dataframes by 'DateTime'
        ce_df.sort_index(inplace=True)
        pe_df.sort_index(inplace=True)

        # Debug: Check for missing or duplicate DateTime
        print(f"Duplicate DateTime in ce_df: {ce_df.index.duplicated().sum()}")  # Check for duplicates
        print(f"Duplicate DateTime in pe_df: {pe_df.index.duplicated().sum()}")  # Check for duplicates

        # Merge the DataFrames on 'DateTime' using an outer join to keep all DateTime intervals
        final_df = pd.merge(ce_df, pe_df, left_index=True, right_index=True, how='outer')

        # Debug: Print the merged DataFrame before filling NaN
        print("Merged DataFrame before filling NaN:")
        print(final_df.head())

        # Fill missing values with 0 for both 'CE pnl' and 'PE pnl'
        final_df.fillna(0, inplace=True)

        # Calculate the combined PnL
        final_df['PnL Combined'] = final_df['CE pnl'] + final_df['PE pnl']

        # Sort by DateTime to ensure intervals are in the correct order
        final_df.sort_index(inplace=True)

        # Debug: Check if the merged DataFrame has all expected rows
        print("Merged and sorted DataFrame:")
        print(final_df.head())

        # Return the final DataFrame
        return final_df




# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, timedelta
import ast, json, sys, re

# Function to pull options data for specified date range 
def pull_options_data_d(start_date, end_date, option_data_path):
            
    start_time = time.time()
    option_data_files = next(os.walk(option_data_path))[2]
    option_data = pd.DataFrame()

    for file in option_data_files:

        match = re.search(r'(\d+)', file).group(1)
        date1 = datetime.strptime(match[0:4] + match[4:6] + '01', "%Y%m%d").date().strftime("%Y-%m-%d")
        
        if (date1>=start_date) & (date1<=end_date):

            temp_data = pd.read_pickle(option_data_path + file)[['ExpiryDate', 'StrikePrice', 'Type', 'Open', 'Ticker']]
            temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
            temp_data = temp_data.rename_axis('DateTime')
            option_data = pd.concat([option_data, temp_data])

    option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
    option_data['Type'] = option_data['Type'].astype('category')
    
    end_time = time.time()
    print('Time taken to pull Options data :', (end_time-start_time))

    return option_data




# if __name__ == "__main__":
#     # stock = 'FINNIFTY'
#     dte_column = 'DaysToExpiry'

#     # csv_files = Path(folder_path).glob('*.csv')
#     csv_files = next(os.walk(folder_path))[2]
#     print(csv_files)
#     calculated_pnl_dfs = []

#     # print(csv_files[0])
#     # csv_files = [element for element in csv_files if element in lots_df['Strategy'].to_list()]

#     # print(csv_files)
#     # print(len(csv_files))
#     if stock == 'NIFTY':
#         start_date = '2022-06-01'
#         end_date = '2022-12-31'
#     elif stock == 'SENSEX':
#         start_date = '2023-08-01'
#         end_date = '2025-04-30'

#     # stock , expiry_file_path
#     option_data = pull_options_data_d(start_date, end_date, option_data_path)
#     print("option_data", option_data)
#     counter = 1


#     for file in csv_files:
#         file_path = os.path.join(folder_path, file)
#         trades_df = process_file(file_path, strategy)
#         print("trades_df", trades_df)

#         # Convert date strings to actual datetime.date for comparison
#         start_exclude = datetime.strptime("2024-05-31", "%Y-%m-%d").date()
#         end_exclude = datetime.strptime("2024-06-06", "%Y-%m-%d").date()

#         if 'Date' in trades_df.columns:
#             trades_df['Date'] = pd.to_datetime(trades_df['Date'], errors='coerce').dt.date
#             trades_df = trades_df[~((trades_df['Date'] >= start_exclude) & (trades_df['Date'] <= end_exclude))]

#         elif 'entry_date' in trades_df.columns:
#             trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'], errors='coerce')
#             trades_df = trades_df[~((trades_df['entry_date'].dt.date >= start_exclude) &
#                                     (trades_df['entry_date'].dt.date <= end_exclude))]

#         # print(trades_df)
#         trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'], format='%d-%m-%Y %H:%M', errors='coerce')
#         trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'], format='%d-%m-%Y %H:%M', errors='coerce')
#         trades_df.sort_values(by='Entry Time', inplace=True)
#         print("trades_df after sorting", trades_df)
#         print("options_data", option_data)
#         option_data = option_data.sort_index()
#         final_pnl_df = generate_1min_pnl_with_hedge(trades_df, option_data, lot_size=75)

#         print(final_pnl_df)

#         # exit()
        
#         final_pnl_df.to_csv(f'{root_path}/{stock}/1_min_pnl/{file}', index=True)


def process_pnl(file, stock, option_data_path, folder_path, root_path, option_data):
    """
    Process the PnL for each file with the preloaded option_data
    """
    # Set the start and end dates based on stock type


    # Get the full file path
    file_path = os.path.join(folder_path, file)
    
    # Process the file using your existing method
    trades_df = process_file(file_path, strategy)
    start_exclude = datetime.strptime("2024-05-31", "%Y-%m-%d").date()
    end_exclude = datetime.strptime("2024-06-06", "%Y-%m-%d").date()

    if 'Date' in trades_df.columns:
        trades_df['Date'] = pd.to_datetime(trades_df['Date'], errors='coerce').dt.date
        trades_df = trades_df[~((trades_df['Date'] >= start_exclude) & (trades_df['Date'] <= end_exclude))]

    elif 'entry_date' in trades_df.columns:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'], errors='coerce')
        trades_df = trades_df[~((trades_df['entry_date'].dt.date >= start_exclude) &
                                (trades_df['entry_date'].dt.date <= end_exclude))]

    # print(trades_df)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    trades_df.sort_values(by='Entry Time', inplace=True)
    # option_data = option_data.sort_index()
    final_pnl_df = generate_1min_pnl_with_hedge(trades_df, option_data, lot_size=75)
    
    # Save the final PnL to a CSV
    final_pnl_df.to_csv(f'{root_path}/{stock}/1_min_pnl/{file}', index=True)




def run_parallel_processing(csv_files, stock, option_data_path, folder_path, root_path, option_data):
    """
    Run parallel processing for the CSV files with preloaded option data
    """
    # Create a Pool of workers, each processing a different file
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_pnl, [(file, stock, option_data_path, folder_path, root_path, option_data) for file in csv_files])


if __name__ == "__main__":
    # Example usage: Specify the stock, option data path, and root path
    # stock = 'NIFTY'
    # option_data_path = 'path/to/option/data/'
    # root_path = 'path/to/root/folder'
    # folder_path = 'path/to/csv/folder'  # Folder where CSV files are located

    if stock == 'NIFTY':
        start_date = '2022-06-01'
        end_date = '2022-12-31'
    elif stock == 'SENSEX':
        start_date = '2023-08-01'
        end_date = '2025-04-30'

    # Load the option data once at the start
    option_data = pull_options_data_d(start_date, end_date, option_data_path)
    option_data = option_data.sort_index()
    print("Loaded option data")

    # Get the list of CSV files to process
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Call the parallel processing function
    run_parallel_processing(csv_files, stock, option_data_path, folder_path, root_path, option_data)














# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from datetime import datetime, timedelta
# import os
# from multiprocessing import Pool


# # Function to process each trade for pnl calculations
# def process_trade_for_pnl(trade, option_data_df, lot_size, stock, expiry_file_path, option_data_path, TIME_FRAME='15T'):
#     main_type = trade['Option']
#     hedge_type = trade['Hedge Option']
#     main_strike = int(trade['Strike'])
#     hedge_strike = int(trade['Hedge Strike'])
#     entry_price = float(trade['Entry Premium'])
#     hedge_entry_price = float(trade['Hedge Entry Premium'])

#     entry_time = pd.to_datetime(trade['Entry Time'])
#     exit_time = pd.to_datetime(trade['Exit Time'])
#     hedge_entry_time = pd.to_datetime(trade['Hedge Entry Time'])
#     hedge_exit_time = pd.to_datetime(trade['Hedge Exit Time'])
#     expiry_date = pd.to_datetime(trade['ExpiryDate'])

#     # Main leg data for trade entry to expiry
#     main_leg_data = option_data_df[
#         (option_data_df['Type'] == main_type) & 
#         (option_data_df['StrikePrice'] == main_strike) & 
#         (option_data_df['DateTime'] >= entry_time) & 
#         (option_data_df['DateTime'] <= expiry_date)
#     ]
#     main_leg_data = main_leg_data.resample(TIME_FRAME, on='DateTime').agg({
#         'Open': 'first', 
#         'Close': 'last', 
#         'High': 'max', 
#         'Low': 'min'
#     }).dropna()

#     # Hedge leg data for hedge entry to expiry
#     hedge_leg_data = option_data_df[
#         (option_data_df['Type'] == hedge_type) & 
#         (option_data_df['StrikePrice'] == hedge_strike) & 
#         (option_data_df['DateTime'] >= hedge_entry_time) & 
#         (option_data_df['DateTime'] <= expiry_date)
#     ]
#     hedge_leg_data = hedge_leg_data.resample(TIME_FRAME, on='DateTime').agg({
#         'Open': 'first', 
#         'Close': 'last', 
#         'High': 'max', 
#         'Low': 'min'
#     }).dropna()

#     # Calculate PnL for main leg (short leg)
#     ce_pnl_dict = defaultdict(float)
#     for _, row in main_leg_data.iterrows():
#         pnl = (entry_price - row['High']) * lot_size
#         ce_pnl_dict[row.name] += round(pnl, 2)

#     # Calculate PnL for hedge leg (long leg)
#     pe_pnl_dict = defaultdict(float)
#     for _, row in hedge_leg_data.iterrows():
#         pnl = (row['High'] - hedge_entry_price) * lot_size
#         pe_pnl_dict[row.name] += round(pnl, 2)

#     # Combine CE and PE pnl
#     pnl_combined_dict = defaultdict(float)
#     for key in ce_pnl_dict.keys():
#         pnl_combined_dict[key] += ce_pnl_dict[key] + pe_pnl_dict.get(key, 0)

#     print(pnl_combined_dict)

#     return pd.DataFrame(pnl_combined_dict.items(), columns=['DateTime', 'PnL Combined'])


# def generate_1min_pnl_with_hedge(trades_df, option_data_df, lot_size, TIME_FRAME='15T'):
#     # Process each trade in parallel
#     with Pool() as pool:
#         results = pool.starmap(process_trade_for_pnl, [(trade, option_data_df, lot_size, stock, expiry_file_path, option_data_path, TIME_FRAME) for _, trade in trades_df.iterrows()])


#     print("Results from parallel processing:", results)
#     # Combine the individual DataFrames into a single final DataFrame
#     final_df = pd.concat(results).sort_values(by='DateTime').reset_index(drop=True)
    
#     return final_df


# # Main execution flow
# if __name__ == "__main__":
#     # Load necessary data
#     dte_column = 'DaysToExpiry'
#     csv_files = next(os.walk(folder_path))[2]

#     # Load options data (assuming the function exists)
#     option_data = load_options_data(start_date, end_date, stock, expiry_file_path, option_data_path)

#     print("option_data", option_data)
#     for file in csv_files:
#         file_path = os.path.join(folder_path, file)
#         trades_df = process_file(file_path, strategy)
#         print("trades_df", trades_df)
        
#         # Process the trades and generate PnL
#         final_pnl_df = generate_1min_pnl_with_hedge(trades_df, option_data, lot_size=75)

#         # Save final PnL to CSV
#         final_pnl_df.to_csv(f'{root_path}/{stock}/1_min_pnl/{file}', index=False)












































































# def generate_1min_pnl_with_hedge(trades_df, option_data_df, lot_size):
#     """
#     Generate 1-minute PnL for trades that include both primary and hedge legs.
#     The short leg PnL is calculated as: entry_price - high
#     The hedge leg PnL is calculated as: high - hedge_entry_price

#     Parameters:
#     - trades_df: DataFrame containing trade info (CE/PE short + hedge)
#     - option_data_df: 1-min OHLC option data ['DateTime', 'Type', 'StrikePrice', 'High']
#     - lot_size: Instrument lot size (e.g., 50 for NIFTY)

#     Returns:
#     - final_df: DataFrame with ['DateTime', 'CE pnl', 'PE pnl', 'PnL Combined']
#     """
#     pnl_dict = {'CE pnl': defaultdict(float), 'PE pnl': defaultdict(float)}

#     for _, trade in trades_df.iterrows():
#         try:
#             main_type = trade['Option']
#             hedge_type = trade['Hedge Option']
#             main_strike = int(trade['Strike'])
#             hedge_strike = int(trade['Hedge Strike'])
#             entry_price = float(trade['Entry Premium'])
#             hedge_entry_price = float(trade['Hedge Entry Premium'])

#             entry_time = pd.to_datetime(trade['Entry Time'])
#             exit_time = pd.to_datetime(trade['Exit Time'])
#             hedge_entry_time = pd.to_datetime(trade['Hedge Entry Time'])
#             hedge_exit_time = pd.to_datetime(trade['Hedge Exit Time'])

#             # Main short leg
#             main_leg_data = option_data_df[
#                 (option_data_df['Type'] == main_type) &
#                 (option_data_df['StrikePrice'] == main_strike) &
#                 (option_data_df['DateTime'] >= entry_time) &
#                 (option_data_df['DateTime'] <= exit_time)
#             ]

#             for _, row in main_leg_data.iterrows():
#                 pnl = (entry_price - row['High']) * lot_size
#                 key = 'CE pnl' if main_type == 'CE' else 'PE pnl'
#                 pnl_dict[key][row['DateTime']] += round(pnl, 2)

#             # Hedge leg
#             hedge_leg_data = option_data_df[
#                 (option_data_df['Type'] == hedge_type) &
#                 (option_data_df['StrikePrice'] == hedge_strike) &
#                 (option_data_df['DateTime'] >= hedge_entry_time) &
#                 (option_data_df['DateTime'] <= hedge_exit_time)
#             ]

#             for _, row in hedge_leg_data.iterrows():
#                 pnl = (row['High'] - hedge_entry_price) * lot_size
#                 key = 'CE pnl' if hedge_type == 'CE' else 'PE pnl'
#                 pnl_dict[key][row['DateTime']] += round(pnl, 2)

#         except Exception as e:
#             print(f"Skipping row due to error: {e}")
#             continue

#     # Convert to DataFrame
#     ce_df = pd.DataFrame(pnl_dict['CE pnl'].items(), columns=['DateTime', 'CE pnl'])
#     pe_df = pd.DataFrame(pnl_dict['PE pnl'].items(), columns=['DateTime', 'PE pnl'])

#     final_df = pd.merge(ce_df, pe_df, on='DateTime', how='outer')
#     final_df.fillna(0, inplace=True)
#     final_df['PnL Combined'] = final_df['CE pnl'] + final_df['PE pnl']
#     final_df = final_df.sort_values(by='DateTime')

#     return final_df