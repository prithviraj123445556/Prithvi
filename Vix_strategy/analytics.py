import pandas as pd              
import os     
from datetime import datetime   
import datetime as dt  
import numpy as np  
import time
import glob
# from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import shutil
import multiprocessing
from tqdm import tqdm
import os
import pandas as pd
import re
import ast 


###############################################  FINAL TRADESHEET CREATOR  ###########################################
# INPUT FILES ARE:
# filter_df6
# Trade_Sheets folder

####################################################################################################
# stock = 'NIFTY'
stock = 'SENSEX'
# stock = 'FINNIFTY'                        
# option_type = 'CE'    
option_type = 'ND'

superset = 'Vix_strategy'
dte ='4'                                       ## 1,2,3,4,0.25
# ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
# superset = 'EMA Crossover'
# superset = 'EMA MACD Crossover'
# superset = 'EMA MACD Support'
# superset = 'RSI'

# root_dir = f"/home/newberry/{superset} copy/"
# root_dir = f"/home/newberry4/jay_test/{superset}/"
# filter_df = pd.read_csv(root_dir + f'/{stock}/{option_type}/Filter_Sheets/filter_df8.csv')
# filter_df = pd.read_csv(root_dir + f'temp_data/{stock}/{option_type}/filter_df6.csv')
# tradesheet_folder = root_dir + f'/{stock}/{option_type}/Trade_Sheets/'
# output_folder = root_dir + f'/{stock}/{option_type}/final_tradesheet/'

# day_lots = 4

# if superset == 'Vix_strategy':
#     max_investment = 90000 * 6
# elif superset == 'Vix_strategy_2nd_week':
#     max_investment = 180000 * 13
# elif superset == 'Vix_strategy_3rd_week':
#     max_investment = 180000  * 20

####################################################################################################

# os.makedirs(output_folder, exist_ok=True)

lot_size_dict = {'NIFTY': 75,'FINNIFTY': 25,
            'BANKNIFTY': 15, 'SENSEX': 20}
govt_tc_dict = {"NIFTY": 2.25, 'FINNIFTY': 2.25 ,"SENSEX": 3 , 
           "BANKNIFTY": 3}

if stock =='SENSEX':
    total_months = 14
    Date = {'start': '2024-07-01', 'end': '2025-08-31'}
else:
    total_months = 49
    Date = {'start': '2021-06-01', 'end': '2025-06-30'}


# for index, row in filter_df.iterrows():
#     strategy_file = row['Strategy']
#     start_date = row['Start_Date']
#     end_date = row['End_Date']

#     # Check if the strategy file exists
#     strategy_path = os.path.join(tradesheet_folder, strategy_file)
#     if not os.path.exists(strategy_path + '.csv'):
#         print(f"Strategy file {strategy_file}.csv not found.")
#         continue

#     try:
#         strategy_data = pd.read_csv(strategy_path + '.csv')
#     except:
#         print('Data not available:', strategy_path)
#         continue

#     # Directly add the entire strategy_data to combined_filtered_data without any filtering
#     combined_filtered_data = strategy_data.copy()  # Copying the entire strategy_data

#     # Output file path
#     output_file_path = os.path.join(output_folder, f"{strategy_file}.xlsx")
    
#     # Save combined data to an Excel file
#     combined_filtered_data.to_excel(output_file_path, index=False)
#     print(f"Saved data to {output_file_path}")

#     # Check if the output file already exists
#     if os.path.exists(output_file_path):
#         print(f"File {output_file_path} already exists. Skipping.")
#         continue

#     #saved in final_tradesheet
#     combined_filtered_data.to_excel(output_file_path, index=False)


########################################### daily pnl ##################################
























#################################################################*************************



# Function to extract the number of lots from the filename
def extract_lots_from_filename(filename):
    match = re.search(r"lots_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def extract_entry_time_from_filename(filename):
    match = re.search(r"entry_(\d{2},\d{2})", filename)
    if match:
        return match.group(1)  # Returns the entry time as a string in "HH,MM" format
    return None

# inputfolder_path = root_dir + f'/{stock}/{option_type}/final_tradesheet/'
inputfolder_path = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/'
# outputfolder_path = root_dir + f'/{stock}/{option_type}/dailypnl/'

# outputfolder_path = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/dailypnl_{entry}/'
outputfolder_path = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/dailypnl/'
#####################################################################################################

files = os.listdir(inputfolder_path)
dfs = {}


# def get_spread_from_range(value, stock, lot_size, action):
#     if stock=='NIFTY' or stock=='FINNIFTY':
#         range_dict = {'(0, 10)': 0.05,
#                       '(10.001, 33)': 0.1
#                      }
        
#     elif stock=='BANKNIFTY':
#         range_dict = {'(0, 10)': 0.05,
#                       '(10.001, 33)': 0.1
#                      }
          
#     for key, range_tuple in range_dict.items():
#         start, end = eval(key)
#         if start <= abs(value) <= end:
#             return abs(range_tuple * lot_size)
       
#     return (abs(value * lot_size * 0.3) / 100)


def get_spread_from_range(value, stock, lot_size, action):
    # Apply spread based on the action type
    if action == "Short":
        spread = 0.5 * abs(value) * lot_size / 100
    elif action == "Long":
        spread = 1 * abs(value) * lot_size / 100
    else:
        raise ValueError("Action must be either 'Short' or 'Long'.")

    return spread


for file in files:
    if file.endswith('.csv'):
        entry_time = extract_entry_time_from_filename(file)
        
        # Only process files that contain the specified entry time
        # if entry_time == entry:
        file_path = os.path.join(inputfolder_path, file)
        df = pd.read_csv(file_path)
        dfs[file] = df

for file, df in dfs.items():
    if file.startswith('NIFTY'):
        idx_calc = 'NIFTY'
    elif file.startswith("BANKNIFTY"):
        idx_calc = 'BANKNIFTY'
    elif file.startswith("FINNIFTY"):
        idx_calc = 'FINNIFTY'
    elif file.startswith("SENSEX"):
        idx_calc = 'SENSEX'

    print(file)

    lots = extract_lots_from_filename(file)
        
        # Calculate max_investment based on the number of lots
    if lots:
        lot_number = lots
    else:
        print(f"Could not determine lots for file {file}, skipping.")
        continue
    
    govt_charge = govt_tc_dict[idx_calc]
    lot_size = lot_size_dict[idx_calc]

    pnl_list = []
    for idx, row in df.iterrows():
        row_premium = row['Premium']
        action = row['Action']
        # Helper function to extract the first value if the cell contains a list-like string
        def extract_first_value(cell): 
            if isinstance(cell, str) and cell.startswith('['):
                # Safely convert list-like string to list and take the first element
                return ast.literal_eval(cell)[0]
            return cell  # Return as is if it's already a single value

        # Extract first values for premium columns
        PE_Long_Premium = extract_first_value(row['PE_Long_Premium'])
        PE_Short_Premium = extract_first_value(row['PE_Short_Premium'])
        CE_Short_Premium = extract_first_value(row['CE_Short_Premium'])
        CE_Long_Premium = extract_first_value(row['CE_Long_Premium'])

        # Convert to numeric, ignoring any remaining invalid values (just in case)
        PE_Long_Premium = pd.to_numeric(PE_Long_Premium, errors='coerce')
        PE_Short_Premium = pd.to_numeric(PE_Short_Premium, errors='coerce')
        CE_Short_Premium = pd.to_numeric(CE_Short_Premium, errors='coerce')
        CE_Long_Premium = pd.to_numeric(CE_Long_Premium, errors='coerce')

        # Count how many premiums are less than 1, ignoring NaN values
        low_premium_count = sum([
            (PE_Long_Premium < 1) if pd.notna(PE_Long_Premium) else 0,
            (PE_Short_Premium < 1) if pd.notna(PE_Short_Premium) else 0,
            (CE_Short_Premium < 1) if pd.notna(CE_Short_Premium) else 0,
            (CE_Long_Premium < 1) if pd.notna(CE_Long_Premium) else 0
        ])
    # Calculate transaction cost, adjusting for premiums below 1
        row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size, action)) + (4 - low_premium_count) * lot_number * govt_charge
        row_pnl = (row_premium * lot_size) - (row_tc)
        pnl_list.append(row_pnl)


    df['PnL'] = pnl_list

    result_df = df.groupby(['Date', 'DaysToExpiry']).agg({
        'ExpiryDate': 'first',
        'PnL': 'sum'
    }).reset_index()

    result_df['Date'] = pd.to_datetime(result_df['Date'])
    start_date = pd.to_datetime(Date['start'])
    end_date = pd.to_datetime(Date['end'])
    result_df = result_df[(result_df['Date'] >= start_date) & (result_df['Date'] <= end_date)]

    if not os.path.exists(outputfolder_path):
        os.makedirs(outputfolder_path)
    

    # Save the result_df as an Excel file in the 'dailypnl/' folder
    output_file_path = os.path.join(outputfolder_path, f'{file}')

    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists. Skipping.")
        continue

    selected_columns = ['Date', 'DaysToExpiry', 'ExpiryDate', 'PnL']
    result_df[selected_columns].to_csv(output_file_path, index=False)







#################################################################*************************


















########################################################################################################################3
#filtering according to each period profitability

# strategy_list = []
# dte = [0, 1, 2, 3, 4]

# #32 months
# periods = [
#     {'in_start': '2021-06-07', 'in_end': '2022-05-31', 'out_start': '2022-06-01', 'out_end': '2022-09-31'},
#     {'in_start': '2021-10-01', 'in_end': '2022-09-31', 'out_start': '2022-10-01', 'out_end': '2023-01-31'},
#     {'in_start': '2022-02-01', 'in_end': '2023-01-31', 'out_start': '2023-02-01 ', 'out_end': '2023-05-31'},
#     {'in_start': '2022-06-01', 'in_end': '2023-05-31', 'out_start': '2023-06-01', 'out_end': '2023-09-31'},
#     {'in_start': '2022-10-01', 'in_end': '2023-09-31', 'out_start': '2023-10-01 ', 'out_end': '2024-01-31'},
#     {'in_start': '2023-02-01', 'in_end': '2024-01-31', 'out_start': '2024-02-01 ', 'out_end': '2024-05-10'}
# ]

# #################################################################################################
# # path to the folder
# folder_path = root_dir + f'/{stock}/{option_type}/dailypnl/*.xlsx'
# excel_filename = root_dir + f'{superset}_{stock}_{option_type}.xlsx'
# #################################################################################################

# file_paths = glob.glob(folder_path)

# # for file_path in file_paths:
# #     excel_data = pd.read_excel(file_path)
# #     strategy_list.append(excel_data)

# def read_file(filename):
#     df = pd.read_excel(filename, engine='openpyxl')
#     return df

# # Create a dictionary to store the results
# strategy_dict = {}

# # Use multiprocessing to read files concurrently
# with multiprocessing.Pool(processes=16) as pool:
#     with tqdm(total=len(file_paths), desc='Processing', unit='Iteration') as pbar:
#         for file_path in file_paths:
#             filename = os.path.basename(file_path)
#             result = pool.apply_async(read_file, args=(file_path,))
#             strategy_dict[filename] = result.get()
#             pbar.update(1)


# def calculate_period_pnl(df, dte_value, in_start, in_end, out_start, out_end):
#     # Filter data based on dte_value
#     filtered_data = df[df['DaysToExpiry'] == dte_value]

#     # Filter data based on date ranges
#     in_period_data = filtered_data[(filtered_data['Date'] >= in_start) & (filtered_data['Date'] <= in_end)]
#     out_period_data = filtered_data[(filtered_data['Date'] >= out_start) & (filtered_data['Date'] <= out_end)]

#     # Convert 'PnL' column to numeric to handle non-numeric values
#     in_period_data['PnL'] = pd.to_numeric(in_period_data['PnL'], errors='coerce')
#     out_period_data['PnL'] = pd.to_numeric(out_period_data['PnL'], errors='coerce')

#     # Calculate total PnL for each period
#     in_period_pnl = in_period_data['PnL'].sum()
#     out_period_pnl = out_period_data['PnL'].sum()

#     return in_period_pnl, out_period_pnl

# def walkforwardpnl():
#     results = {dte_val: [] for dte_val in dte}  # Use a dictionary to store results for each dte

#     for file_path, df in strategy_dict.items():
#         unique_dte_values = df['DaysToExpiry'].unique()

#         for dte_value in unique_dte_values:
#             for period in periods:
#                 in_start = period['in_start']
#                 in_end = period['in_end']
#                 out_start = period['out_start']
#                 out_end = period['out_end']

#                 in_pnl, out_pnl = calculate_period_pnl(df, dte_value, in_start, in_end, out_start, out_end)

#                 # if in_pnl <=0.0 or out_pnl <= 0.0:                      ## if want to filter for every part positive
#                 #     all_periods_positive = False
#                 #     break  
#                 # if all_periods_positive:
#                 # result = {
#                 #     'file':os.path.basename(file_path),
#                 #     'dte': dte_value,
#                 # }
#                 result = {
#                     'file': os.path.basename(file_path),
#                     'dte': dte_value,
#                     'in_pnl': in_pnl,
#                     'out_pnl': out_pnl
#                 }
#                 results[dte_value].append(result)  # Append result to the corresponding dte

#     return results


# result_dict = walkforwardpnl()
# # Save results to Excel files, each dte in a separate sheet
# with pd.ExcelWriter(excel_filename) as writer:
#     for dte_value, result_list in result_dict.items():
#         df = pd.DataFrame(result_list)
#         df.to_excel(writer, sheet_name=f'dte_{dte_value}', index=False)





##################################### ANALYTICS CREATOR #########################################

# Date = {'start': '2021-06-07', 'end': '2023-09-30'}
# total_months = 28

# def minPnl(Date, df, days_to_expiry):
#     monthly31 = {'start': '2021-06-01', 'end': '2024-09-30'}
#     monthly11 = {'start': '2023-10-01', 'end': '2024-09-30'}
#     monthly3 = {'start': '2024-06-01', 'end': '2024-09-30'}

#     def calculate_monthly_pnl(Date, df, days_to_expiry):
#         start_date = Date['start']
#         end_date = Date['end']

#         mask = (df['DaysToExpiry'] == days_to_expiry) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
#         monthly_pnls = df.loc[mask, 'PnL'].tolist()

#         return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

#     # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
#     Dmonthly31_pnl = calculate_monthly_pnl(monthly31, df, days_to_expiry) / 40
#     Dmonthly11_pnl = calculate_monthly_pnl(monthly11, df, days_to_expiry) / 12
#     Dmonthly3_pnl = calculate_monthly_pnl(monthly3, df, days_to_expiry) / 4
    
#     all_monthly_pnls = [Dmonthly31_pnl, Dmonthly11_pnl, Dmonthly3_pnl]

#     # Remove None values before finding the overall minimum
#     all_pnls = [pnl for pnl in all_monthly_pnls if pnl is not None]
#     # overall_min_pnl = min(all_pnls, default=None)
#     overall_min_pnl = Dmonthly31_pnl

#     return overall_min_pnl,Dmonthly31_pnl,Dmonthly11_pnl,Dmonthly3_pnl





































#################################################################*************************






import pandas as pd
import os
existing_df = pd.DataFrame()
#tradesheet_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/tradesheet_dte0.csv')
#pnl_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/pnl_dte0_dtecol.csv')
#output_folder = "/home/newberry2/vix2_analytics_report/"


def minPnl(Date, df):
    monthly31 = {'start': '2021-06-01', 'end': '2025-08-31'}
    monthly11 = {'start': '2024-09-01', 'end': '2025-08-31'}
    monthly3 = {'start': '2025-05-01', 'end': '2025-08-31'}


    def calculate_monthly_pnl(Date, df):
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'PnL'].tolist()

        return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly31_pnl = calculate_monthly_pnl(monthly31, df) / 14
    Dmonthly11_pnl = calculate_monthly_pnl(monthly11, df) / 12
    Dmonthly3_pnl = calculate_monthly_pnl(monthly3, df) / 4
    print(Dmonthly31_pnl)
    all_monthly_pnls = [Dmonthly31_pnl, Dmonthly11_pnl, Dmonthly3_pnl]

    # Remove None values before finding the overall minimum
    all_pnls = [pnl for pnl in all_monthly_pnls if pnl is not None]
    # overall_min_pnl = min(all_pnls, default=None)
    overall_min_pnl = Dmonthly31_pnl

    return overall_min_pnl,Dmonthly31_pnl,Dmonthly11_pnl,Dmonthly3_pnl

# Drawdown
def get_drawdown(Date, PnL):
    
    max_drawdown = 0
    max_drawdown_percentage = 0
    max_drawdown_date = None
    time_to_recover = 0
    peak_date_before_max_drawdown = None
    
    cum_pnl = 0
    peak = 0
    peak_date = Date.iloc[0]
    # peak_date = dt.datetime.strptime(Date[0], '%Y-%m-%d')
    
    for date, pnl in zip(Date, PnL):
        print(date)
        cum_pnl += pnl
        if (time_to_recover is None) and (cum_pnl >= peak):
            time_to_recover = (date - peak_date).days
            # time_to_recover = (dt.datetime.strptime(date, '%Y-%m-%d') - peak_date).days
            
        if cum_pnl >= peak:
            peak = cum_pnl
            peak_date = date
            # peak_date = dt.datetime.strptime(date, '%Y-%m-%d')
        
        drawdown = peak - cum_pnl
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            if peak != 0:
                max_drawdown_percentage = 100*max_drawdown/peak
            max_drawdown_date = date
            peak_date_before_max_drawdown = peak_date
            time_to_recover = None
    
    return max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown


def analytics(sub_dataframe_dailypnl, existing_df, filename_str, max_investment):
    # total_months = 23
    sub_dataframe_dailypnl['PnL'] = pd.to_numeric(sub_dataframe_dailypnl['PnL'], errors='coerce')

    # Convert 'Date' column to datetime format if not already in datetime
    sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], errors='coerce')

    totalpnl = sub_dataframe_dailypnl['PnL'].sum()

    # Call get_drawdown function for overall period
    max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
        sub_dataframe_dailypnl['Date'],
        sub_dataframe_dailypnl['PnL']
    )

    overall_min_pnl, Dmonthly31_pnl, Dmonthly11_pnl, Dmonthly3_pnl = minPnl(sub_dataframe_dailypnl['Date'], sub_dataframe_dailypnl)

    # Additional operations
    Profits = sub_dataframe_dailypnl[sub_dataframe_dailypnl['PnL'] > 0]['PnL']
    Losses = sub_dataframe_dailypnl[sub_dataframe_dailypnl['PnL'] <= 0]['PnL']

    total_trades = len(sub_dataframe_dailypnl)
    num_winners = len(Profits)
    num_losers = len(Losses)
    win_percentage = 100 * num_winners / total_trades
    loss_percentage = 100 * num_losers / total_trades

    max_profit = Profits.max() if num_winners > 0 else 0
    max_loss = Losses.min() if num_losers > 0 else 0

    median_pnl = sub_dataframe_dailypnl['PnL'].median()
    median_profit = Profits.median() if num_winners > 0 else 0
    median_loss = Losses.median() if num_losers > 0 else 0

    sd_pnl = sub_dataframe_dailypnl['PnL'].std()
    sd_profit = Profits.std() if num_winners > 0 else 0
    sd_loss = Losses.std() if num_losers > 0 else 0

    # max_investment is already provided
    max_investment = max_investment

    roi_with_dd = 100 * (totalpnl) / (max_investment + max_drawdown)
    roi = 100 * overall_min_pnl * total_months / max_investment

    # Calculate drawdown for the last 15 months from the date range
    date_range_start = pd.to_datetime('2021-06-01')
    date_range_end = pd.to_datetime('2025-08-31')

    # Filter the data for the last 15 months
    filtered_data = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['Date'] >= date_range_start) & (sub_dataframe_dailypnl['Date'] <= date_range_end)]

    # Calculate the drawdown for the filtered data
    last_15_months_max_drawdown, last_15_months_max_drawdown_percentage, last_15_months_max_drawdown_date, last_15_months_time_to_recover, last_15_months_peak_date_before_max_drawdown = get_drawdown(
        filtered_data['Date'],
        filtered_data['PnL']
    )

    # Calculate Monthly Return and Sortino Ratio
    monthly_return = totalpnl / ( max_investment * total_months )
    minimum_return_needed = 0.003
    sd_loss_ratio = sd_loss/max_investment
    sortino_ratio = (monthly_return - minimum_return_needed) / sd_loss_ratio if sd_loss > 0 else 0

    # Create a DataFrame with the metrics
    result_df = pd.DataFrame({
        'Filename': [filename_str],
        'Total PnL': [totalpnl],
        'Max Drawdown': [max_drawdown],
        'Max Drawdown Percentage': [max_drawdown_percentage],
        'All Monthly PnL': [Dmonthly31_pnl],
        '12M Monthly PnL': [Dmonthly11_pnl],
        'Daily 4M PnL ': [Dmonthly3_pnl],
        'min pnl ': [overall_min_pnl],
        'Max Investment': [max_investment],
        'ROI % ': [roi],
        'ROI with DD': [roi_with_dd],
        'Max Drawdown Date': [max_drawdown_date],
        'Time to Recover': [time_to_recover],
        'Peak Date Before Max Drawdown': [peak_date_before_max_drawdown],
        'Total Trades': [total_trades],
        'No. of Winners': [num_winners],
        'No. of Losers': [num_losers],
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
        'Last 15 Months Max Drawdown': [last_15_months_max_drawdown],
        'Last 15 Months Max Drawdown Percentage': [last_15_months_max_drawdown_percentage],
        'Last 15 Months Max Drawdown Date': [last_15_months_max_drawdown_date],
        'Last 15 Months Time to Recover': [last_15_months_time_to_recover],
        'Last 15 Months Peak Date Before Max Drawdown': [last_15_months_peak_date_before_max_drawdown],
        'Sortino Ratio': [sortino_ratio]
    })

    # Drop strategies where total PnL, Dmonthly11_pnl, or Sortino Ratio is negative
    # if totalpnl < 0 or Dmonthly11_pnl < 0 or sortino_ratio <= 0:
    #     return existing_df  # Skip appending this row if the conditions are met

    # Append the result to the existing dataframe
    existing_df = pd.concat([existing_df, result_df], ignore_index=True)


    return existing_df




import os
import pandas as pd
import re
from datetime import datetime

alldte_pnl_files = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/dailypnl/'

# --- Base investment setup ---
if stock == 'NIFTY':
    base_investment_per_lot = 90000
elif stock == 'SENSEX':
    base_investment_per_lot = 90000
else:
    base_investment_per_lot = 90000  # default

# --- Extract lots helper ---
def extract_lots_from_filename(filename):
    match = re.search(r"lots_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


# --- Step 1: Collect file info ---
count_records = []
file_data_list = []

for root, dirs, files in os.walk(alldte_pnl_files):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            pnl_df = pd.read_csv(file_path)

            if pnl_df.empty:
                print(f"[SKIP] Empty file: {file}")
                continue

            if 'Date' not in pnl_df.columns:
                print(f"[WARN] Missing 'Date' column in {file}")
                continue

            pnl_df['Date'] = pd.to_datetime(pnl_df['Date']).dt.date
            row_count = len(pnl_df)

            count_records.append({'Strategy': file, 'RowCount': row_count})
            file_data_list.append((file_path, pnl_df))

            print(f"[INFO] {file} — rows: {row_count}")

        except Exception as e:
            print(f"[ERROR] Failed reading {file}: {e}")


# --- Step 2: Save counts report ---
count_df = pd.DataFrame(count_records)
output_csv = os.path.join(alldte_pnl_files, "strategy_row_counts.csv")
count_df.to_csv(output_csv, index=False)
print(f"\n✅ [INFO] Row counts saved to: {output_csv}")

# --- Step 3: Summary check ---
if not count_df.empty:
    unique_counts = count_df['RowCount'].unique()
    print(f"[INFO] Unique row counts found: {unique_counts}")
    if len(unique_counts) == 1:
        print("✅ All files have the same number of rows.")
    else:
        print("⚠️ Not all files have the same number of rows.")
else:
    print("[WARN] No valid data files processed.")


# --- Step 4: Run analytics for each file ---
existing_df = pd.DataFrame()

for file_path, pnl_df in file_data_list:
    file_name = os.path.basename(file_path)
    lots = extract_lots_from_filename(file_name)

    if not lots:
        print(f"[WARN] Could not determine lots for {file_name}, skipping analytics.")
        continue

    max_investment = base_investment_per_lot * lots
    try:
        existing_df = analytics(pnl_df, existing_df, file_name, max_investment)
        print(f"[ANALYZED] {file_name}")
    except Exception as e:
        print(f"[ERROR] Analytics failed for {file_name}: {e}")

print(f"\n✅ [INFO] Completed analytics for all files.")
print(f"Total strategies analyzed: {len(existing_df)}")





from scipy.stats import zscore

def compute_final_z_scores(final_df):
    # Calculate ratios and z-scores for each required metric
    final_df['Dmonthly12_PnL_Ratio'] = final_df['12M Monthly PnL'] / final_df['Max Investment']   ## 36 month 
    final_df['Max_Drawdown_Ratio'] = final_df['Max Drawdown'] / final_df['Max Investment']
    final_df['Win %'] = final_df['Win %']                                    ## 11 month
    final_df['Sortino Ratio'] = final_df['Sortino Ratio']

    # Calculate z-scores for each metric
    final_df['Dmonthly12_PnL_Z'] = zscore(final_df['Dmonthly12_PnL_Ratio'])
    final_df['Max_Drawdown_Z'] = zscore(final_df['Max_Drawdown_Ratio'])
    final_df['Win%_Z'] = zscore(final_df['Win %'])
    final_df['Sortino_Ratio_Z'] = zscore(final_df['Sortino Ratio'])

    # Standardize z-scores to range [0, 1]
    final_df['Dmonthly12_PnL_Z'] = (final_df['Dmonthly12_PnL_Z'] - final_df['Dmonthly12_PnL_Z'].min()) / (final_df['Dmonthly12_PnL_Z'].max() - final_df['Dmonthly12_PnL_Z'].min())
    final_df['Max_Drawdown_Z'] = (final_df['Max_Drawdown_Z'] - final_df['Max_Drawdown_Z'].min()) / (final_df['Max_Drawdown_Z'].max() - final_df['Max_Drawdown_Z'].min())
    final_df['Win%_Z'] = (final_df['Win%_Z'] - final_df['Win%_Z'].min()) / (final_df['Win%_Z'].max() - final_df['Win%_Z'].min())
    final_df['Sortino_Ratio_Z'] = (final_df['Sortino_Ratio_Z'] - final_df['Sortino_Ratio_Z'].min()) / (final_df['Sortino_Ratio_Z'].max() - final_df['Sortino_Ratio_Z'].min())

    # Apply weights to the z-scores, using (1 - Z) approach for drawdowns
    final_df['Final_Z_Score'] = (
        1 * final_df['Sortino_Ratio_Z'] + 
        1 * final_df['Dmonthly12_PnL_Z'] + 
        1 * (1 - final_df['Max_Drawdown_Z']) + 
        0.5 * final_df['Win%_Z']
    )

    # Sort by Final_Z_Score in descending order
    final_df = final_df.sort_values(by='Final_Z_Score', ascending=False)

    # Check for the top strategy with Sortino Ratio > 1
    top_strategy = None
    for _, row in final_df.iterrows():
        if row['Sortino Ratio'] > 1:
            top_strategy = row
            break

    if top_strategy is None:
        print("[WARNING] No strategy with Sortino Ratio > 1 found.")
    else:
        print(f"[INFO] Selected top strategy with Sortino Ratio > 1: {top_strategy['Filename']}")
        # Move top strategy to the top
        final_df = pd.concat([final_df.loc[[top_strategy.name]], final_df.drop(top_strategy.name)], ignore_index=True)

    return final_df


# Usage
# Assuming `existing_df` is the final DataFrame containing all selected strategies after appending each strategy
final_sorted_df = compute_final_z_scores(existing_df)

final_sorted_df.to_csv(rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/Analytics_{dte}.csv', index=False)



def read_excel_file(file_path):
    return pd.read_csv(file_path)

def calculate_ratio(pnl_value, max_investment):
    if max_investment != 0:
        return (pnl_value / max_investment) * 100
    else:
        return 0



def calculate_correlations_with_top_strategy(analytics_df, dailypnl_folder):
    print("\n[DEBUG] Starting correlation calculation with top strategy")

    # Select the top strategy (first row in the sorted analytics DataFrame)
    top_strategy = analytics_df.iloc[0]
    top_strategy_file = top_strategy['Filename']
    top_strategy_final_z_score = top_strategy['Final_Z_Score']

    # Dictionary to store correlation results
    correlation_dict = {}

    # Read and calculate top strategy's daily PnL %
    top_file_path = os.path.join(dailypnl_folder, top_strategy_file)
    top_df = read_excel_file(top_file_path)
    top_df['PnL %'] = calculate_ratio(top_df['PnL'], top_strategy['Max Investment'])
    top_df = top_df[['Date', 'PnL %']].rename(columns={'PnL %': 'Top Strategy PnL %'})

    for idx, row in analytics_df.iloc[1:].iterrows():  # Skip the top strategy itself
        try:
            strategy_file = row['Filename']
            file_path = os.path.join(dailypnl_folder, strategy_file)

            strategy_df = read_excel_file(file_path)
            strategy_df['PnL %'] = calculate_ratio(strategy_df['PnL'], row['Max Investment'])
            strategy_df = strategy_df[['Date', 'PnL %']].rename(columns={'PnL %': 'Other Strategy PnL %'})

            # Merge dataframes on Date to align the PnL % data for correlation calculation
            merged_df = pd.merge(top_df, strategy_df, on='Date', how='outer').fillna(0)

            correlation_value = merged_df['Top Strategy PnL %'].corr(merged_df['Other Strategy PnL %'])
            if correlation_value is not None:
                correlation_dict[(top_strategy_file, strategy_file)] = correlation_value
                print(f"[DEBUG] Correlation between {top_strategy_file} and {strategy_file}: {correlation_value}")

        except Exception as e:
            print(f"[DEBUG] Error processing correlation between {top_strategy_file} and {strategy_file}: {e}")
            continue

    return correlation_dict, top_strategy_file, top_strategy_final_z_score

def process_correlations_and_create_output(correlation_dict, output_filepath, top_strategy_name, top_strategy_final_z_score):
    # Create a DataFrame from the correlation dictionary
    correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Strategy Pair', 'Correlation'])
    correlation_df[['Top Strategy', 'Correlated Strategy']] = pd.DataFrame(correlation_df['Strategy Pair'].tolist(), index=correlation_df.index)
    correlation_df.drop(columns='Strategy Pair', inplace=True)

    # Calculate Z-score of correlations
    correlation_df['Correlation Z'] = zscore(correlation_df['Correlation'])

    # Standardize the Correlation Z-score to a range [0, 1]
    min_z = correlation_df['Correlation Z'].min()
    max_z = correlation_df['Correlation Z'].max()
    correlation_df['Standardized Correlation Z'] = (correlation_df['Correlation Z'] - min_z) / (max_z - min_z)

    # Calculate Standardized Non Correlation Z Score as 1 - Standardized Correlation Z
    correlation_df['Standardized Non Correlation Z Score'] = 1 - correlation_df['Standardized Correlation Z']

    # Filter for strategies with Standardized Non Correlation Z Score between 0.5 and 1
    non_correlated_df = correlation_df[(correlation_df['Standardized Non Correlation Z Score'] >= 0.5) & 
                                       (correlation_df['Standardized Non Correlation Z Score'] <= 1)]

    # Load the original analytics DataFrame to get the Final_Z_Score and Sortino Ratio for each strategy
    analytics_df = pd.read_csv(analytics_filepath)  # Ensure the path is correctly defined
    non_correlated_df = non_correlated_df.merge(
        analytics_df[['Filename', 'Final_Z_Score', 'Sortino Ratio']], 
        left_on='Correlated Strategy', 
        right_on='Filename', 
        how='left'
    ).drop(columns=['Filename'])

    # Sort the filtered strategies by Final_Z_Score in descending order
    non_correlated_df = non_correlated_df.sort_values(by='Final_Z_Score', ascending=False)

    # Select the top non-correlated strategy with Sortino Ratio > 1
    top_non_correlated_strategy = None
    for _, row in non_correlated_df.iterrows():
        if row['Sortino Ratio'] > 1:
            top_non_correlated_strategy = row
            break

    if top_non_correlated_strategy is None:
        print("[WARNING] No non-correlated strategy with Sortino Ratio > 1 found.")
    else:
        print(f"[INFO] Selected non-correlated strategy with Sortino Ratio > 1: {top_non_correlated_strategy['Correlated Strategy']}")

    # Add the top strategy name and its Final_Z_Score
    non_correlated_df['Top Strategy'] = top_strategy_name
    non_correlated_df['Top Strategy Final_Z_Score'] = top_strategy_final_z_score

    # Save the filtered correlations to a CSV file
    non_correlated_df.to_csv(output_filepath, index=False)
    print(f"[DEBUG] Saved analytics correlation to {output_filepath}")

    # Return the top non-correlated strategy for further analysis if needed
    return top_non_correlated_strategy


def main(analytics_filepath, dailypnl_folder, output_filepath):
    analytics_df = pd.read_csv(analytics_filepath)  # Load final analytics DataFrame

    # Step 1: Calculate correlations with the top strategy
    correlation_dict, top_strategy_file, top_strategy_final_z_score = calculate_correlations_with_top_strategy(analytics_df, dailypnl_folder)

    # Step 2: Process correlations and create output (save to CSV)
    top_non_correlated_strategy = process_correlations_and_create_output(
        correlation_dict, 
        output_filepath, 
        top_strategy_file, 
        top_strategy_final_z_score
    )

    print(f"[DEBUG] Top Non-Correlated Strategy: {top_non_correlated_strategy}")


# File paths for analytics sheet, daily pnl folder, and output file
# entry = entry.replace(',', '_')  # Replace commas with underscores (or any other safe character)

analytics_filepath = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/Analytics_{dte}.csv'
dailypnl_folder = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/dailypnl/'

output_filepath = rf'/home/newberry4/jay_test/{superset}/{stock}/ND/dte/{dte}/Analytics_correlation_{dte}.csv'

# Ensure the directory exists
output_dir = os.path.dirname(output_filepath)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the main function
main(analytics_filepath, dailypnl_folder, output_filepath)








#################################################################*************************








































































# def read_excel_file(file_path):
#     return pd.read_excel(file_path)

# def calculate_ratio(pnl_value, max_investment):
#     if max_investment != 0:
#         return (pnl_value / max_investment) * 100
#     else:
#         return 0

# def calculate_and_store_correlations(output_dataframe, dailypnl_folder):
#     print("\n[DEBUG] Starting correlation calculation")

#     result_dict = {}

#     # Loop through each row in the output DataFrame
#     for index, row in output_dataframe.iterrows():
#         days_to_expiry = row['DTE']
#         file_name_in_dailypnl = row['file']
#         dailypnl_file_path = os.path.join(dailypnl_folder, file_name_in_dailypnl)

#         print(f"[DEBUG] Processing file: {file_name_in_dailypnl} for DTE: {days_to_expiry}")

#         try:
#             # Read the file and filter by days_to_expiry
#             dailypnl_dataframe = read_excel_file(dailypnl_file_path)
#             dailypnl_dataframe = dailypnl_dataframe[dailypnl_dataframe['DaysToExpiry'] == days_to_expiry]

#             if dailypnl_dataframe.empty:
#                 print(f"[DEBUG] Warning: Data for {file_name_in_dailypnl} is empty after filtering for DTE {days_to_expiry}")
#                 continue

#             pnl_value = dailypnl_dataframe.loc[:, 'PnL']
#             max_investment = row['Max Investment']

#             # Calculate the ratio
#             ratio = calculate_ratio(pnl_value, max_investment)
#             dailypnl_dataframe['PnL %'] = ratio

#             # Store the result in the dictionary
#             result_dict[file_name_in_dailypnl] = dailypnl_dataframe[['Date', 'PnL %']]

#         except Exception as e:
#             print(f"[DEBUG] Error processing {file_name_in_dailypnl}: {e}")
#             continue

#     print(f"[DEBUG] Completed result_dict: {list(result_dict.keys())}")

#     # Calculate correlations
#     correlations = {}

#     if not result_dict:
#         print("[DEBUG] result_dict is empty, skipping correlation calculation.")
#         return correlations, None

#     first_key = list(result_dict.keys())[0]
#     first_list = result_dict[first_key]
#     first_list['PnL %_first'] = first_list['PnL %']

#     print(f"[DEBUG] First key for correlation: {first_key}")

#     for other_key in result_dict.keys():
#         if other_key != first_key:
#             other_list = result_dict[other_key]
#             other_list['PnL %_second'] = other_list['PnL %']

#             print(f"[DEBUG] Calculating correlation between {first_key} and {other_key}")

#             # Merge the data on 'Date'
#             merged_df = pd.merge(first_list[['Date', 'PnL %_first']], other_list[['Date', 'PnL %_second']], on='Date', how='outer')

# # Fill NaN values with 0 for missing PnL %
#             merged_df['PnL %_first'].fillna(0, inplace=True)
#             merged_df['PnL %_second'].fillna(0, inplace=True)

# # Sort by 'Date' after the merge
#             merged_df.sort_values(by='Date', inplace=True)

#             if merged_df.empty:
#                 print(f"[DEBUG] No overlapping dates between {first_key} and {other_key}")
#                 continue

#             merged_df = merged_df[~((merged_df['PnL %_first'] == 0) & (merged_df['PnL %_second'] == 0))]

#             # Calculate correlation
#             correlation = merged_df['PnL %_first'].corr(merged_df['PnL %_second'])

#             print(f"[DEBUG] Correlation between {first_key} and {other_key}: {correlation}")

#             if correlation is not None and -0.5 <= correlation <= 0.5:
#                 correlations[(first_key, other_key)] = correlation
#                 print(f"[DEBUG] Storing correlation between {first_key} and {other_key}")

#     print(f"[DEBUG] Completed correlation calculations: {correlations}")
#     return correlations, first_key


# def process_and_calculate_correlations(output_folder, dailypnl_folder):
#     print("\n[DEBUG] Starting to process files in output_folder")

#     all_results_list = []

#     for filename in os.listdir(output_folder):
#         if filename.endswith(".xlsx"):
#             print(f"[DEBUG] Processing file: {filename}")

#             try:
#                 output_file_path = os.path.join(output_folder, filename)
#                 output_dataframe = read_excel_file(output_file_path)

#                 if output_dataframe.empty:
#                     print(f"[DEBUG] Warning: {filename} is empty.")
#                     continue

#                 # Drop duplicates and reset index
#                 output_dataframe = output_dataframe.drop_duplicates().reset_index().drop(columns='index')
#                 print(f"[DEBUG] {filename} loaded successfully with shape: {output_dataframe.shape}")

#                 # Calculate and store correlations
#                 correlations, first_key = calculate_and_store_correlations(output_dataframe, dailypnl_folder)

#                 if correlations:
#                     # Append the dictionary of results to the list
#                     all_results_list.append((filename, correlations, first_key))
#                     print(f"[DEBUG] Correlations stored for {filename}")
#                 else:
#                     print(f"[DEBUG] No correlations found for {filename}")

#             except Exception as e:
#                 print(f"[DEBUG] Error processing {filename}: {e}")

#     print("[DEBUG] Completed processing all files")
#     return all_results_list

# def selected_strategy_info(unique_keys_df, analytics_folder, dailypnl_folder, output_folder):
#     info_list = []

# # CHANGE selected_dailypnl_folder NAME ######################
#     selected_dailypnl_folder = os.path.join(output_folder, f'{superset}_{stock}_{option_type}_dailypnl')
#     os.makedirs(selected_dailypnl_folder, exist_ok=True)

#     # Iterate over all files in the 'Analytics' folder
#     for filename in os.listdir(analytics_folder):
#         if filename.endswith(".xlsx"):
            
#             analytics_file_path = os.path.join(analytics_folder, filename)
#             analytics_dataframe = read_excel_file(analytics_file_path)

#             analytics_dataframe['file_and_DTE'] = analytics_dataframe['file'] + '_' + analytics_dataframe['DTE'].astype(str)
#             unique_keys_df['Keys_and_DTE'] = unique_keys_df['Keys'] + '_' + unique_keys_df['DTE'].astype(str)
            
#             if 'file' in analytics_dataframe.columns:
#                 relevant_rows = unique_keys_df[unique_keys_df['Keys_and_DTE'].isin(analytics_dataframe['file_and_DTE'])]

#                 # Extract relevant information and append to info_list
#                 for index, row in relevant_rows.iterrows():
#                     action_value = option_type
#                     superset_value = superset

#                     info = {
#                         'Strategy': row['Keys'],
#                         'Action': action_value,
#                         'Max Investment': analytics_dataframe.loc[analytics_dataframe['file'] == row['Keys'], 'Max Investment'].iloc[0],
#                         'DaysToExpiry': analytics_dataframe.loc[analytics_dataframe['file'] == row['Keys'], 'DTE'].iloc[0],
#                         'Superset': superset_value,
#                     }
#                     info_list.append(info)

#                     dailypnl_file_name = row['Keys']
#                     dailypnl_source_path = os.path.join(dailypnl_folder, dailypnl_file_name)
#                     if os.path.exists(selected_dailypnl_folder):
#                         dailypnl_dest_path = os.path.join(selected_dailypnl_folder, dailypnl_file_name)
#                         shutil.copyfile(dailypnl_source_path, dailypnl_dest_path)
#                     else:
#                         os.makedirs(selected_dailypnl_folder, exist_ok=True)
#                         shutil.copyfile(dailypnl_source_path, dailypnl_dest_path)
                        
#     strategy_info_df = pd.DataFrame(info_list)


#     # Check if the Excel file exists
#     excel_file_path = os.path.join(output_folder, f'{stock}_{option_type}_strategy_info.xlsx')
#     if os.path.exists(excel_file_path):
#         # Load existing Excel file
#         existing_df = pd.read_excel(excel_file_path)
#         updated_df = pd.concat([existing_df, strategy_info_df], ignore_index=True)
#         updated_df = updated_df.drop_duplicates().reset_index().drop(columns = 'index')
#         updated_df.to_excel(excel_file_path, index=False)
#         print("Data appended to existing Excel file.")
#     else:
#         strategy_info_df = strategy_info_df.drop_duplicates().reset_index().drop(columns = 'index')
#         strategy_info_df.to_excel(excel_file_path, index=False)
#         print("New Excel file created with the appended data.")




# def main():

#     print("Starting main() function")

#     all_results_list = process_and_calculate_correlations(analytics_folder, dailypnl_folder)
#     print("All results list loaded successfully:", all_results_list)

#     unique_keys_df = pd.DataFrame(columns=['Keys', 'DTE'])
#     counter_check = 1
#     for filename, correlations, first_key in all_results_list:
#         counter = 0  # Counter to track the number of correlations printed
#         unique_keys = []
#         dte_list = []
#         DTE = int(filename.split('.xlsx')[0][-1])
        
#         print("\nFilename Check:", filename)
#         print("Counter Check:", counter_check)
#         counter_check += 1

#         print("First Key:", first_key)

#         if not correlations:
#             print("No correlations found for this file. Adding first_key to unique_keys.")
#             unique_keys.append(first_key)
#             dte_list.append(DTE)
#             new_data = pd.DataFrame({'Keys': unique_keys, 'DTE': dte_list})
#             unique_keys_df = pd.concat([unique_keys_df, new_data], ignore_index=True)
#             continue

#         for keys, correlation in correlations.items():
#             print(f"Processing keys: {keys}, correlation: {correlation}")
            
#             if counter < 1:
#                 print("Counter < 1, adding keys to unique_keys")

#                 # Add both keys to the list
#                 unique_keys.append(keys[0])
#                 dte_list.append(DTE)
#                 unique_keys.append(keys[1])
#                 dte_list.append(DTE)

#                 print(f"Filename: {filename}")
#                 print(f"  First Key: {keys[0]}, Other Key: {keys[1]}, Correlation: {correlation}")
#                 counter += 1

#             elif counter == 1:
#                 print("Counter == 1, processing second and third sets")

#                 second_set = unique_keys[1]
#                 third_set = keys[1]

#                 print(f"Second Set: {second_set}, Third Set: {third_set}")

#                 try:
#                     second_pnl = pd.read_excel(dailypnl_folder + '/' + second_set)
#                     second_pnl = second_pnl[second_pnl['DaysToExpiry'] == DTE]

#                     third_pnl = pd.read_excel(dailypnl_folder + "/" + third_set)
#                     third_pnl = third_pnl[third_pnl['DaysToExpiry'] == DTE]

#                     analytics_dataframe = pd.read_excel(analytics_folder + '/' + filename)
#                     second_max = analytics_dataframe[analytics_dataframe['file'] == second_set]['Max Investment'].iloc[0]
#                     third_max = analytics_dataframe[analytics_dataframe['file'] == third_set]['Max Investment'].iloc[0]

#                     print(f"Second Max: {second_max}, Third Max: {third_max}")

#                     second_pnl['PnL %_2'] = (second_pnl['PnL'] / second_max) * 100
#                     third_pnl['PnL %_3'] = (third_pnl['PnL'] / third_max) * 100

#                     merged_df = pd.merge(second_pnl[['Date', 'PnL %_2']], third_pnl[['Date', 'PnL %_3']], on='Date', how='outer')

#                     merged_df['PnL %_2'].fillna(0, inplace=True)
#                     merged_df['PnL %_3'].fillna(0, inplace=True)

#                     merged_df = merged_df[~((merged_df["PnL %_2"] == 0) & (merged_df["PnL %_3"] == 0))]

#                     second_third_correlation = merged_df['PnL %_2'].corr(merged_df['PnL %_3'])
#                     print(f"Correlation between second and third sets: {second_third_correlation}")

#                     if -0.5 < second_third_correlation < 0.5:
#                         print("Correlation is in the range [-0.5, 0.5]. Adding third key to unique_keys.")
#                         unique_keys.append(keys[1])
#                         dte_list.append(DTE)

#                         print(f"Filename: {filename}")
#                         print(f"  First Key: {keys[0]}, Other Key: {keys[1]}, Correlation: {correlation}")
#                         counter += 1
#                         break

#                 except Exception as e:
#                     print(f"Error processing second and third sets: {e}")

#         new_data = pd.DataFrame({'Keys': unique_keys, 'DTE': dte_list})
#         unique_keys_df = pd.concat([unique_keys_df, new_data], ignore_index=True)
        
#         print("Updated unique_keys_df:")
#         print(unique_keys_df)

#     print("Final unique_keys_df:")
#     print(unique_keys_df)

#     selected_strategy_info(unique_keys_df, analytics_folder, dailypnl_folder, results_folder)
#     print("Completed selected_strategy_info")

# if __name__ == "__main__":
#     main()
