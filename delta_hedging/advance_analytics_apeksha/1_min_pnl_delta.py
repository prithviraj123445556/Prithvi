### for only non rentry
import pandas as pd
import numpy as np
import psycopg2 as sql
from datetime import datetime, timedelta
import os, sys
from functools import partial
from pandarallel import pandarallel
# sys.path.insert(0, r"/home/newberry3/")

from running_pnl_2 import load_options_data, process_file, params, modify_ticker , process_file_expiry
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
#################################################################################################
# stock = 'NIFTY'
stock = 'SENSEX'
# superset = 'COMBINED_WITHOUT_REENTRY'
# strategy = 'NIFTY'
strategy = 'SENSEX'


LOT_SIZE = 75 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (25 if stock == 'FINNIFTY' else (20 if stock == 'SENSEX' else None)))

start_date = '2021-06-01'
end_date = '2025-08-31'

root_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/'
output_folder_path = f'{root_path}/{stock}/1_min_pnl//'
folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/ND/Trade_Sheets/' 
# 


# folder_path = f'/home/newberry4/jay_test/delta_hedging/advance_analytics_apeksha/tradesheet/' 
# expiry_file_path = rf"/home/newberry4/jay_data/Common_Files/{stock} market dates.xlsx"

if stock == 'NIFTY':        
    expiry_file_path = rf"/home/newberry4/jay_data/NIFTY Market Dates updated 2025.xlsx"
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
    option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OI_OHLC2/"



os.makedirs(output_folder_path, exist_ok=True)
#################################################################################################

# lots_df = pd.read_excel(f'/home/newberry4/jay_test/SHORT_STRADDLE/1_min_no_rentry/{stock}_premium_lots.xlsx')
# lots_df['Strategy'] = lots_df['Strategy'].str.replace('_p.xlsx', '.csv')



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

import pandas as pd



#########  without multiprocessing #######


# def generate_1min_pnl_with_intraday_carryforward(trades_df, option_data_df, lot_size):
#     pnl_dict = {'CE pnl': defaultdict(float), 'PE pnl': defaultdict(float)}

#     for _, trade in trades_df.iterrows():
#         opt_type = trade['type']
#         strike = trade['strike']
#         entry_price = trade['entry_price']
#         entry_dt = pd.to_datetime(trade['entry_date'])
#         print("entry_dt", entry_dt)
#         exit_dt = pd.to_datetime(trade['exit_date'])

#         data = option_data_df[
#             (option_data_df['Type'] == opt_type) &
#             (option_data_df['StrikePrice'] == strike) &
#             (option_data_df['DateTime'] >= entry_dt) &
#             (option_data_df['DateTime'] <= exit_dt)
#         ].copy()

#         data.sort_index(inplace=True)
#         print("data", data)
#         if data.empty:
#             continue

#         data['PnL'] = (entry_price - data['Open']) * lot_size
#         data['PnL'] = data['PnL'].round(2)

#         key = 'CE pnl' if opt_type == 'CE' else 'PE pnl'

#         last_pnl = 0
#         last_time = None

#         for _, row in data.iterrows():
#             pnl_dict[key][row['DateTime']] += row['PnL']
#             last_pnl = row['PnL']
#             last_time = row['DateTime']

#         # Carry forward PnL till end of the same day
#         if last_time is not None:
#             end_of_day = last_time.normalize() + pd.Timedelta(days=1)
#             carry_forward = option_data_df[
#                 (option_data_df['DateTime'] > last_time) &
#                 (option_data_df['DateTime'] < end_of_day) &
#                 (option_data_df['Type'] == opt_type)
#             ]

#             carry_forward = carry_forward.sort_values(by='DateTime')

#             for dt in carry_forward['DateTime'].unique():
#                 pnl_dict[key][dt] += last_pnl

#     # Build final DataFrame
#     ce_df = pd.DataFrame(pnl_dict['CE pnl'].items(), columns=['DateTime', 'CE pnl'])
#     pe_df = pd.DataFrame(pnl_dict['PE pnl'].items(), columns=['DateTime', 'PE pnl'])

#     final_df = pd.merge(ce_df, pe_df, on='DateTime', how='outer')
#     final_df.fillna(0, inplace=True)
#     final_df['PnL Combined'] = final_df['CE pnl'] + final_df['PE pnl']
#     final_df = final_df.sort_values(by='DateTime')

#     return final_df







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
#         start_date = '2021-06-01'
#         end_date = '2025-08-31'
#         LOT_SIZE = 75
#     elif stock == 'SENSEX':
#         start_date = '2023-08-01'
#         end_date = '2025-08-31'
#         LOT_SIZE = 20

#     option_data = load_options_data(start_date, end_date, stock, expiry_file_path, option_data_path)

#     counter = 1


#     for file in csv_files:
#         file_path = os.path.join(folder_path, file)
#         trades_df = process_file_expiry(file_path, strategy)
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
            

#         final_pnl_df = generate_1min_pnl_with_intraday_carryforward(trades_df, option_data, LOT_SIZE)
#         print(final_pnl_df)
#         final_pnl_df.to_csv(f'{root_path}/{stock}/1_min_pnl/{file}', index=False)


#########  without multiprocessing #######











######## same function but for multiprocessing

import pandas as pd
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
from functools import partial

# Define the function that generates PnL with carry forward
def generate_1min_pnl_with_intraday_carryforward(trades_df, option_data_df, lot_size):
    pnl_dict = {'CE pnl': defaultdict(float), 'PE pnl': defaultdict(float)}

    for _, trade in trades_df.iterrows():
        opt_type = trade['type']
        strike = trade['strike']
        entry_price = trade['entry_price']
        entry_dt = pd.to_datetime(trade['entry_date'])
        print("entry_dt", entry_dt)
        exit_dt = pd.to_datetime(trade['exit_date'])

        data = option_data_df[
            (option_data_df['Type'] == opt_type) &
            (option_data_df['StrikePrice'] == strike) &
            (option_data_df['DateTime'] >= entry_dt) &
            (option_data_df['DateTime'] <= exit_dt)
        ].copy()

        data.sort_index(inplace=True)

        if data.empty:
            continue

        data['PnL'] = (entry_price - data['Open']) * lot_size
        data['PnL'] = data['PnL'].round(2)

        key = 'CE pnl' if opt_type == 'CE' else 'PE pnl'

        last_pnl = 0
        last_time = None

        for _, row in data.iterrows():
            pnl_dict[key][row['DateTime']] += row['PnL']
            last_pnl = row['PnL']
            last_time = row['DateTime']

        # Carry forward PnL till end of the same day
        if last_time is not None:
            end_of_day = last_time.normalize() + pd.Timedelta(days=1)
            carry_forward = option_data_df[
                (option_data_df['DateTime'] > last_time) &
                (option_data_df['DateTime'] < end_of_day) &
                (option_data_df['Type'] == opt_type)
            ]

            carry_forward = carry_forward.sort_values(by='DateTime')

            for dt in carry_forward['DateTime'].unique():
                pnl_dict[key][dt] += last_pnl

    # Build final DataFrame
    ce_df = pd.DataFrame(pnl_dict['CE pnl'].items(), columns=['DateTime', 'CE pnl'])
    pe_df = pd.DataFrame(pnl_dict['PE pnl'].items(), columns=['DateTime', 'PE pnl'])

    final_df = pd.merge(ce_df, pe_df, on='DateTime', how='outer')
    final_df.fillna(0, inplace=True)
    final_df['PnL Combined'] = final_df['CE pnl'] + final_df['PE pnl']
    final_df = final_df.sort_values(by='DateTime')

    return final_df


# Function to process each file in parallel
def process_file(file_path, option_data, lot_size, stock, root_path,file):
    trades_df = process_file_expiry(file_path, strategy)  # Assuming process_file_expiry is implemented somewhere
    
    # Filter out rows based on date exclusion
    start_exclude = datetime.strptime("2024-05-31", "%Y-%m-%d").date()
    end_exclude = datetime.strptime("2024-06-06", "%Y-%m-%d").date()

    if 'Date' in trades_df.columns:
        trades_df['Date'] = pd.to_datetime(trades_df['Date'], errors='coerce').dt.date
        trades_df = trades_df[~((trades_df['Date'] >= start_exclude) & (trades_df['Date'] <= end_exclude))]
    elif 'entry_date' in trades_df.columns:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'], errors='coerce')
        trades_df = trades_df[~((trades_df['entry_date'].dt.date >= start_exclude) &
                                (trades_df['entry_date'].dt.date <= end_exclude))]

    final_pnl_df = generate_1min_pnl_with_intraday_carryforward(trades_df, option_data, lot_size)
    
    final_pnl_df.to_csv(f'{root_path}/{stock}/1_min_pnl/{file}', index=False)
    return final_pnl_df


# Main execution with multiprocessing
if __name__ == "__main__":
    dte_column = 'DaysToExpiry'
    csv_files = next(os.walk(folder_path))[2]


    if stock == 'NIFTY':
        start_date = '2021-06-01'
        end_date = '2025-08-31'
        LOT_SIZE = 75
    elif stock == 'SENSEX':
        start_date = '2023-08-01'
        end_date = '2025-08-31'
        LOT_SIZE = 20
    
    # Load options data
    option_data = load_options_data(start_date, end_date, stock, expiry_file_path, option_data_path)
    print("Option data loaded.",option_data.head())
    # Prepare a list of arguments for each file
    args = [(os.path.join(folder_path, file), option_data, LOT_SIZE, stock, root_path, file) for file in csv_files]

    # Use multiprocessing to process the files in parallel
    with Pool() as pool:
        results = pool.starmap(process_file, args)

    # Optionally, combine results if needed
    # final_combined_df = pd.concat(results)

    print("Processing completed for all files.")




















        