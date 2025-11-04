import pandas as pd 
import numpy as np
import os
import re
import pandas as pd
from itertools import product
import sys
import pandas as pd
import os
from datetime import datetime

def get_drawdown(Date, PnL):                      ####daywise drawdown calculation
    max_drawdown = 0
    max_drawdown_date = None
    time_to_recover = None
    peak_date_before_max_drawdown = None
    cum_pnl = 0
    peak = 0

    # Convert Date.iloc[0] to 'YYYY-MM-DD' string if it's a datetime object
    if isinstance(Date.iloc[0], datetime):
        peak_date = Date.iloc[0].strftime('%Y-%m-%d')
    else:
        # If it's a string, ensure it's in 'YYYY-MM-DD' format without time
        if ' ' in Date.iloc[0]:
            peak_date = Date.iloc[0].split()[0]
        else:
            peak_date = Date.iloc[0]

    for date, pnl in zip(Date, PnL):
        cum_pnl += pnl

        # Ensure date is in 'YYYY-MM-DD' format without time
        if ' ' in date:
            date = date.split()[0]

        # Calculate time to recover
        if time_to_recover is None and (cum_pnl >= peak):
            time_to_recover = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(peak_date, "%Y-%m-%d")).days

        # Update peak and peak_date
        if cum_pnl >= peak:
            peak = cum_pnl
            peak_date = date

        drawdown = peak - cum_pnl

        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_date = date
            peak_date_before_max_drawdown = peak_date
            time_to_recover = None 

    return max_drawdown, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown







### Function to calculate drawdown metrics timewise
# def get_drawdown(DateTime, PnL_Combined):
#     DateTime = pd.to_datetime(DateTime)

#     max_drawdown = 0
#     max_drawdown_percentage = 0
#     max_drawdown_date = None
#     time_to_recover = None
#     peak_date_before_max_drawdown = None

#     cum_pnl = 0
#     peak = 0
#     peak_date = DateTime.iloc[0]
#     recovery_start_index = None

#     for i, (dt, pnl) in enumerate(zip(DateTime, PnL_Combined)):
#         cum_pnl += pnl

#         if cum_pnl >= peak:
#             if recovery_start_index is not None:
#                 time_to_recover = round((i - recovery_start_index) / 1440, 2)  # days
#             peak = cum_pnl
#             peak_date = dt
#             recovery_start_index = None

#         drawdown = peak - cum_pnl

#         if drawdown > max_drawdown:
#             max_drawdown = drawdown
#             max_drawdown_date = dt
#             peak_date_before_max_drawdown = peak_date
#             if peak != 0:
#                 max_drawdown_percentage = 100 * drawdown / peak
#             recovery_start_index = i
#             time_to_recover = None  # reset if new drawdown

#     return (
#         max_drawdown,
#         max_drawdown_percentage,
#         max_drawdown_date,
#         time_to_recover,  # now in days
#         peak_date_before_max_drawdown
#     )




def minPnl(Date, df, days_to_expiry):
    monthly40 = {'start': '2021-06-01', 'end': '2025-08-31'}
    monthly4 = {'start': '2025-03-01', 'end': '2025-08-31'}
    monthly12 = {'start': '2024-09-01', 'end': '2025-08-31'}
    # monthly8 = {'start': '2024-01-01', 'end': '2024-09-30'} # 8 month before the 4 month

    def calculate_monthly_pnl(Date, df, days_to_expiry):
        days_to_expiry = 0
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'Daily PnL'].tolist()
        return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly40_pnl = calculate_monthly_pnl(monthly40, df, days_to_expiry)
    Dmonthly12_pnl = calculate_monthly_pnl(monthly12, df, days_to_expiry)
    Dmonthly4_pnl = calculate_monthly_pnl(monthly4, df, days_to_expiry)
    # Dmonthly8_pnl = calculate_monthly_pnl(monthly8, df, days_to_expiry)

    return Dmonthly40_pnl, Dmonthly12_pnl, Dmonthly4_pnl



def calc_sd_loss(loss_series):
    if len(loss_series) == 1:
        # If only one value, SD = sqrt(abs(value))
        return np.sqrt(abs(loss_series.iloc[0]))
    elif len(loss_series) == 0:
        # No values, fallback to 1 or np.nan as needed
        return 1
    else:
        sd_val = loss_series.std()
        return sd_val if sd_val > 0 else 1



def analytics(sub_dataframe_dailypnl, combo_name):
    sub_dataframe_dailypnl['Daily PnL'] = pd.to_numeric(sub_dataframe_dailypnl['Daily PnL'], errors='coerce')
    
    # Filter data before a certain date
    sub_dataframe_dailypnl = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Date'] < "2025-09-01"]
    
    # Subset data for different months
    sub_4_month_df = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Date'] > "2025-03-01"]
    # sub_8_month_df = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Date'] < "2025-04-30"]
    # sub_8_month_df = sub_8_month_df[sub_8_month_df['Date'] >= "2024-09-01"]
    # sub_12_month_df = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Date'] < "2025-04-30"]
    # sub_12_month_df = sub_12_month_df[sub_12_month_df['Date'] >= "2024-05-01"]
    
    # Calculate the total PnL for the entire dataset and subsets
    totalpnl = sub_dataframe_dailypnl['Daily PnL'].sum()
    totalpnl_4 = sub_4_month_df['Daily PnL'].sum()
    # totalpnl_8 = sub_8_month_df['Daily PnL'].sum()
    # totalpnl_12 = sub_12_month_df['Daily PnL'].sum()

    # Call get_drawdown function
    max_drawdown, _, time_to_recover, _ = get_drawdown(
        sub_dataframe_dailypnl['Date'],
        sub_dataframe_dailypnl['Daily PnL']
    )

    # Calculate monthly PnLs
    monthly40pnl, monthly12pnl, monthly4pnl = minPnl(sub_dataframe_dailypnl['Date'], sub_dataframe_dailypnl, days_to_expiry=0)

    # Identify losses for different periods
    Losses = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] <= 0]['Daily PnL']
    Losses_4 = sub_4_month_df[sub_4_month_df['Daily PnL'] <= 0]['Daily PnL']
    # Losses_8 = sub_8_month_df[sub_8_month_df['Daily PnL'] <= 0]['Daily PnL']
    # Losses_12 = sub_12_month_df[sub_12_month_df['Daily PnL'] <= 0]['Daily PnL']

    # Standard deviation of losses for different periods using helper function
    sd_loss = calc_sd_loss(Losses)
    sd_loss_4 = calc_sd_loss(Losses_4)
    # sd_loss_8 = calc_sd_loss(Losses_8)
    # sd_loss_12 = calc_sd_loss(Losses_12)
    sd_loss_40 = sd_loss 

    # Number of winning days and losing days
    num_winners = len(sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] > 0])
    num_losers = len(sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] <= 0])

    # Winning and losing percentages
    total_trades = len(sub_dataframe_dailypnl)
    win_percentage = (num_winners / total_trades) * 100
    loss_percentage = (num_losers / total_trades) * 100

    # Max profit and loss
    max_profit = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] > 0]['Daily PnL'].max()
    max_loss = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] <= 0]['Daily PnL'].min()

    # Median PnL and standard deviation of PnLs
    median_pnl = sub_dataframe_dailypnl['Daily PnL'].median()
    median_profit = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] > 0]['Daily PnL'].median()
    median_loss = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] <= 0]['Daily PnL'].median()

    sd_pnl = sub_dataframe_dailypnl['Daily PnL'].std()
    sd_profit = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] > 0]['Daily PnL'].std()
    sd_loss = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Daily PnL'] <= 0]['Daily PnL'].std()

    # ROI Calculation without Drawdown
    roi = 100 * totalpnl  / max_investment

    # ROI Calculation with Drawdown
    roi_with_dd = 100 * (totalpnl - max_profit) / (max_investment + max_drawdown)

    # Create the result DataFrame with additional metrics
    result_df = pd.DataFrame({
        'File Name': [combo_name],
        'Last 40M Total PnL': [totalpnl],
        '40M SD Loss': [sd_loss_40],  
        'Last 6M Total PnL': [monthly4pnl],
        '6M SD Loss': [sd_loss_4], 
        'Last 12M Total PnL': [monthly12pnl],
        '40 Max Drawdown': [max_drawdown],
        # '4M Before 4M PnL': [totalpnl_8],
        # '4M Before 4M SD Loss': [sd_loss_8],
        # '4M Before 8M PnL': [totalpnl_12],                
        # '4M Before 8M SD Loss': [sd_loss_12],
        'Number of Winning Days': [num_winners],
        'Number of Losing Days': [num_losers],
        'Total Trades': [total_trades],
        'Win %': [win_percentage],
        'Loss %': [loss_percentage],
        'Max Profit': [max_profit],
        'Max Loss': [max_loss],
        'Median PnL': [median_pnl],
        'Median Profit': [median_profit],
        'Median Loss': [median_loss],
        'SD': [sd_pnl],
        'SD Profit': [sd_profit],
        'SD Loss': [sd_loss],
        'max_investment': [max_investment],
        'ROI %': [roi],
        'ROI with DD': [roi_with_dd]
    })
    
    return result_df


# stock = 'NIFTY'
# LOT_SIZE = 75
    
def run_analytics(processed_dfs):
    output_df = pd.DataFrame() 
    for df in processed_dfs:
        set_name = os.path.basename(df['File Path'].iloc[0])
        combo_name = os.path.basename(df['File Name'].iloc[0]) 
        df['Date_Temp'] = df['Date'].astype(str)
        result_df = analytics(df, combo_name)
        output_df = pd.concat([output_df, result_df], ignore_index=True)
    output_df.to_excel(f'{analytics_folder}/{set_name.split(".")[0]}_set_analytics.xlsx', index=False)


strategy = 'NIFTY'

stock = 'NIFTY'
folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
# analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
os.makedirs(analytics_folder, exist_ok=True)
os.makedirs(content_folder, exist_ok=True)


# stock = 'SENSEX'

# if typee  == 'REENTRY':
if stock == 'SENSEX':
    months = 20
    max_investment = 500000
elif stock == 'BANKNIFTY':
    months = 39
    max_investment = 150000
elif stock == 'NIFTY':
    months = 46
    max_investment = 450000 
elif stock == 'FINNIFTY':
    months = 21
    max_investment = 130000



def get_spread_from_range(value, stock, lot_size):
    if stock=='NIFTY' or stock=='FINNIFTY':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
        
    elif stock=='SENSEX':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
          
    for key, range_tuple in range_dict.items():
        start, end = eval(key)
        if start <= abs(value) <= end:
            return abs(range_tuple )
       
    return (abs(value  * 0.3) / 100)


lot_size_dict = {'NIFTY': 75,'SENSEX': 20,
            'BANKNIFTY': 15}
govt_tc_dict = {"NIFTY": 2.25, 'FINNIFTY': 2.25 ,
           "SENSEX": 3}



# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(content_folder, filename.split('.')[0])
        if not os.path.exists(file_path):
            continue
        files_to_process = [file for file in os.listdir(file_path) if file.startswith('pnl_threshold_')]                      ########## for normal
        # files_to_process = [file for file in os.listdir(file_path) if file.startswith('initial_sl_')]               #######for tsl 
        # files_to_process = [file for file in os.listdir(file_path) if file.startswith('daily_pnl_from_threshold')]         ########## for ranges
        processed_dfs = []

        for file_name in files_to_process:
            print("filename", file_name)
            file_path1 = os.path.join(file_path, file_name)
            df = pd.read_csv(file_path1)
            df['File Path'] = filename
            df['File Name'] = file_name
            df['Date'] = df['Date'].astype(str)

            idx_calc = stock
            lot_size = lot_size_dict[idx_calc]
            govt_charge = govt_tc_dict[idx_calc]
            lots = 1

            pnl_list = []
            for idx, row in df.iterrows():
                # print(row)
                row_premium = row['Daily PnL']
                row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size)) + govt_charge *lots
                row_pnl = (row_premium ) - row_tc 
                pnl_list.append(row_pnl)

            df['Daily PnL'] = pnl_list

            # if df['Daily PnL'].any() == 0:
            #     continue

            df.to_csv(os.path.join(file_path, file_name), index=False)
            
            if df is not None:
                processed_dfs.append(df)

        run_analytics(processed_dfs)
        print("All analytics calculations completed and saved.")
                

            

            