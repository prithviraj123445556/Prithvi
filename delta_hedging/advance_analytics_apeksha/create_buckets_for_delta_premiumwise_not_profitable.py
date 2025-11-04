import os
import pandas as pd
import numpy as np

# Define the threshold mapping per variation
# thresholds_by_variation = {
#     "NIFTY_candle_5T_range_[50, 50]_delta_0,3_time_11,45": {
#         10: 486,
#         15: 728.75,
#         20: 1299.276316,
#         25: 1783.206522,
#         30: 923.6079545,
#         35: 1078.875,
#         40: 1110.3125,
#         45: 1570.9375,
#         50: -2613.75,
#         55: 1354.5,
#         60: 3718.125,
#         65: -4027.5,
#         70: 4248.75,
#         75: 521.25,
#         90: 7008.75,
#         95: 3251.25,
#         120: 1815
#     },
#     "NIFTY_candle_5T_range_[50, 50]_delta_0,3_time_13,00": {
#         10: 780.625,
#         15: 1090.9375,
#         20: 695.7421875,
#         25: 787.5,
#         30: 706.40625,
#         35: 1568.4375,
#         40: 1760.625,
#         45: 2793.75,
#         50: 1489.821429,
#         55: 111.25,
#         60: 3350.25,
#         65: -2666.25,
#         70: 318.75
#     },
#     "NIFTY_candle_5T_range_[50, 50]_delta_0,4_time_09,30": {
#         30: -4361.25,
#         35: -1688.75,
#         40: 1358.25,
#         45: -564.2045455,
#         50: 846.09375,
#         55: 4056.5625,
#         60: 242.625,
#         65: -3060.652174,
#         70: -1483.125,
#         75: -1874.134615,
#         90: 1943.75,
#         95: 2876.875,
#         120: -30270
#     },
#     "NIFTY_candle_5T_range_[50, 50]_delta_0,35_time_10,15": {
#         15: -3420,
#         20: -5696.25,
#         25: 349.6875,
#         30: 1431.176471,
#         35: 1105,
#         40: 1499.648438,
#         45: 1941.875,
#         50: 2082.5,
#         55: -1134.715909,
#         60: -412.125,
#         65: 1331.517857,
#         70: 590,
#         75: 3798.333333,
#         120: 10033.125
#     },
#     "NIFTY_candle_5T_range_[50, 50]_delta_0,35_time_13,00": {
#         10: 527.5,
#         15: 1110,
#         20: 283.125,
#         25: 677.1875,
#         30: 1478.602941,
#         35: 1298.611111,
#         40: 1784.765625,
#         45: 377.5,
#         50: 2802.115385,
#         55: 530.625,
#         60: 2456.25,
#         65: 2381.25,
#         70: 3671.25,
#         75: 4644.375,
#         90: 6131.25
#     }
# }


thresholds_by_variation = {
    "SENSEX_candle_5T_range_[50, 50]_delta_0,4_time_11,45": {
        60: 1241,
        80: 1160,
        100: 739.1666667,
        120: 131,
        140: -1336.285714,
        160: 2635.833333,
        180: 1979.375,
        200: 2327.5,
        220: -380.8,
        240: 3333,
        260: 6677.5,
        280: 183,
        300: -5524,
        320: 1782,
        340: 1893,
        400: 3631.5,
        460: -16147,
        520: 3037,
        680: -9947
    },

    "SENSEX_candle_5T_range_[50, 50]_delta_0,5_time_13,00": {
        60: -825,
        80: -259,
        100: 271,
        120: 1517.375,
        140: 1177.875,
        160: -244.7777778,
        180: 2261.75,
        200: 2828.6,
        220: 1391.5,
        240: 2207,
        260: 850,
        280: 259,
        300: 3234,
        320: -284,
        340: -1547,
        360: 2879,
        380: 895.6666667,
        400: 5818,
        420: 843,
        520: 3229,
        600: 2508,
        620: -5259,
        660: 7814
    },

    "SENSEX_candle_5T_range_[66, 66]_delta_0,3_time_10,15": {
        60: 1061.333333,
        80: 2239.333333,
        100: 1466.384615,
        120: 2698.75,
        140: 821.1111111,
        160: 688.125,
        180: 2165.111111,
        200: 1213.5,
        220: 981.6666667,
        240: -3231,
        260: -1653.666667,
        300: 28,
        340: 2068.5,
        400: 6121
    },

    "SENSEX_candle_5T_range_[70, 88]_delta_0,4_time_11,45": {
        60: 1882,
        80: 1798.666667,
        100: 825.1666667,
        120: 718.8571429,
        140: -287.2857143,
        160: 3374,
        180: 2455.5,
        200: 955,
        220: -1319.4,
        240: 4319,
        260: 6677.5,
        280: 646.3333333,
        300: -6114.5,
        320: -829,
        340: -209.5,
        400: 2790,
        460: -3542,
        520: 1644,
        680: -1554
    }
}











stock = 'SENSEX'
strategy = 'SENSEX_apeksha_new'
folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
trade_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/ND/Trade_Sheets/'
analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
# content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content3/'



os.makedirs(folder_path, exist_ok=True)
os.makedirs(trade_folder, exist_ok=True)
os.makedirs(analytics_folder, exist_ok=True)
os.makedirs(content_folder, exist_ok=True)

# Loop through all PnL files
for fname in os.listdir(folder_path):
    if not fname.endswith(".csv"):
        continue

    variation = fname.split(".csv")[0]

    if variation not in thresholds_by_variation:
        print(f"Skipping {variation}, no threshold table found.")
        continue

    threshold_table = thresholds_by_variation[variation]
    pnl_file = os.path.join(folder_path, fname)
    trade_file = os.path.join(trade_folder, fname)  # same filename assumed

    # Check if trade file exists
    if not os.path.exists(trade_file):
        print(f"Trade file missing for {variation}, skipping.")
        continue

    # Load trade data
    df_trade = pd.read_csv(trade_file, parse_dates=['entry_date', 'exit_date', 'expiry_date'], dayfirst=True)
    df_trade['entry_date'] = pd.to_datetime(df_trade['entry_date'], errors='coerce')
    df_trade['expiry_date'] = pd.to_datetime(df_trade['expiry_date'], errors='coerce')
    df_trade = df_trade[df_trade['entry_date'].dt.date == df_trade['expiry_date'].dt.date]

    # Extract time from filename
    try:
        time_str = fname.split("time_")[1].split(".csv")[0]
        hours, minutes = time_str.split(",")
        filter_time = pd.to_datetime(f"{hours}:{minutes}:00").time()
    except:
        print(f"Skipping {fname} due to time parsing issue.")
        continue

    filtered = df_trade[df_trade['entry_date'].dt.time == filter_time]

    if filtered.empty:
        print(f"No trades at {filter_time} in {fname}")
        continue

    # Compute combined entry price
    entry_aggregated = filtered.groupby(filtered['entry_date'].dt.date).apply(
        lambda group: pd.Series({
            'ce_entry_price': group[group['type'] == 'CE']['entry_price'].sum(),
            'pe_entry_price': group[group['type'] == 'PE']['entry_price'].sum(),
            'combined_entry_price': group[group['type'].isin(['CE', 'PE'])]['entry_price'].sum()
        })
    ).reset_index(names='date')

    # Load PnL file
    df_pnl = pd.read_csv(pnl_file)
    if 'PnL Combined' not in df_pnl.columns or 'DateTime' not in df_pnl.columns:
        print(f"Missing columns in {fname}, skipping.")
        continue

    df_pnl['DateTime'] = pd.to_datetime(df_pnl['DateTime'], errors='coerce')
    df_pnl = df_pnl.dropna(subset=['DateTime'])
    df_pnl['date'] = df_pnl['DateTime'].dt.date

    # Compute daily PnL based on crossing threshold
    results = []
    grouped = df_pnl.groupby('date')

    for date, group in grouped:
        group = group[group['PnL Combined'] != 0].reset_index(drop=True)

        row = entry_aggregated[entry_aggregated['date'] == date]
        if row.empty:
            daily_pnl = group['PnL Combined'].iloc[-1] if not group.empty else 0
        else:
            entry_price = row['combined_entry_price'].values[0]
            floored_price = int(np.floor(entry_price / 5.0) * 5)

            if floored_price in threshold_table:
                threshold_pnl = threshold_table[floored_price]
                crossing = group[group['PnL Combined'] >= threshold_pnl]
                daily_pnl = crossing.iloc[0]['PnL Combined'] if not crossing.empty else group['PnL Combined'].iloc[-1]
            else:
                daily_pnl = group['PnL Combined'].iloc[-1] if not group.empty else 0

        results.append({
            'Date': date,
            'Daily PnL': daily_pnl
        })

    final_df = pd.DataFrame(results)
    output_dir = os.path.join(content_folder, variation)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "daily_pnl_from_threshold.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")











#tsl

# import pandas as pd
# import numpy as np
# import os

# stock = 'NIFTY'
# LOT_SIZE = 75 if stock == 'NIFTY' else 20 

# strategy = 'NIFTY_apeksha_new'
# folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
# analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
# # content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
# content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content2/'

# os.makedirs(folder_path, exist_ok=True)
# os.makedirs(analytics_folder, exist_ok=True)
# os.makedirs(content_folder, exist_ok=True)

# # Loop through all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path)

#         if 'PnL Combined' not in df.columns:
#             continue

#         max_threshold = df['PnL Combined'].max()
#         min_threshold = df['PnL Combined'].min()


#         # Generate 9 evenly spaced values in the negative range
#         negative_range = np.linspace(min_threshold, 0, 9)

#         # Filter out 0
#         negative_range = negative_range[negative_range < 0]

#         # Select the one that's closest to 0 (but not 0)
#         closest_to_zero = negative_range[np.abs(negative_range).argmin()]

#         # Generate 9 evenly spaced values in the positive range
#         second_range = np.linspace(0, max_threshold, 9)

#         # Combine both, rounding to 2 decimal places
#         min_val_range = np.round(np.concatenate(([closest_to_zero], second_range)), 2)

#         # Display the range
#         print(min_val_range)


#         df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
#         df = df.dropna(subset=['DateTime'])
#         df['date'] = df['DateTime'].dt.date

#         df['Date_Temp'] = df['date'].astype(str)
#         trades = df.copy(deep=True)

#         # Loop through all threshold values, starting from the 3rd value (index 2)
#         for i in range(2, len(min_val_range)-1):
#             threshold = min_val_range[i]
#             results = []
#             grouped = df.groupby('date')

#             # Set the initial SL as the first element of the range
#             initial_sl1 = min_val_range[0] 
#             initial_sl = min_val_range[0]  # The first element of the list is the initial SL
#             initial_target = min_val_range[i]  # The current threshold is the initial target
#             final_target = min_val_range[-1]  # The last threshold value (final target)

#             last_pnl = 0  # To keep track of the last PnL value
#             daily_pnl = 0  # Initial daily PnL

#             for date, group in grouped:
#                 group = group.reset_index(drop=True)
#                 group = group[group['PnL Combined'] != 0]

#                 # Initialize the variable to track if the target/SL was crossed
#                 target_crossed = False
#                 sl_crossed = False

#                 # Loop through the group and update the PnL
#                 for index, row in group.iterrows():
#                     if row['PnL Combined'] >= initial_target:
#                         daily_pnl = row['PnL Combined']  # Update PnL when crossing target
#                         target_crossed = True
#                         # Move to the next target in the list, update SL to the previous value
#                         initial_target = min_val_range[i + 1] if i + 1 < len(min_val_range) else final_target
#                         initial_sl = min_val_range[i]  # Move the SL to the previous bucket value

#                     elif row['PnL Combined'] <= initial_sl:
#                         daily_pnl = row['PnL Combined']  # Update PnL when crossing SL
#                         sl_crossed = True
#                         break  # Exit loop if SL is hit

#                 # If no crossing occurred, take the last PnL value for the day
#                 if not target_crossed and not sl_crossed:
#                     daily_pnl = group['PnL Combined'].iloc[-1]  # Last PnL value for the day

#                 results.append({
#                     'Date': date,
#                     'Daily PnL': daily_pnl,
#                     'SL': initial_sl,
#                     'Initial Target': initial_target,
#                     'Final Target': final_target
#                 })

#             # Build the final dataframe
#             final_metrics_df = pd.DataFrame(results)
            
#             # Generate the filename for this range
#             threshold_label = f"{threshold:.2f}"
#             name = filename.split('.c')[0]
#             final_folder = f'{content_folder}/{name}/'
#             os.makedirs(final_folder, exist_ok=True)

#             # Save the results to a CSV file
#             output_filename = f'{final_folder}/initial_sl_{int(initial_sl1)}_initial_target_{int(initial_target)}_final_target_{int(final_target)}.csv'
#             final_metrics_df.to_csv(output_filename, index=False)
#             print(f'Results saved to {output_filename}')


























































# Loop through all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path)
        
#         df = df.drop(columns=['CE pnl', 'PE pnl'])
        
#         #match1 = re.search(r"premium_(\d+)", filename)
#         #premium = int(match1.group(1))
#         #stoploss = (filename.split('.c')[0]).split('stoploss_')[1]
#         #stoploss = float(stoploss.replace(',', '.'))  
#         max_threshold = df['PnL Combined'].max()
#         min_threshold = df['PnL Combined'].min()
#         first_range = np.arange(min_threshold, max_threshold+1, 100)
#         second_range = np.linspace(min_threshold, max_threshold, 20)
#         min_val_range = first_range if min(len(first_range), len(second_range)) == len(first_range) else second_range
#         both_legs_open_profit_range = [x for x in min_val_range if x > 0]
#         both_legs_open_loss_range = [x for x in min_val_range if x < 0]
#         one_leg_open_range = min_val_range

#         df['DateTime'] = df['DateTime'].apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M').strftime('%Y-%m-%d %H:%M') if '-' in x and len(x.split('-')[0]) == 2 else pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'))
#         df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M', errors='coerce')
#         df['date'] = df['DateTime'].dt.date
#         df = df[df['date'] > pd.to_datetime('2023-08-03').date()]
#         df['CE Trade_Exit_Status'] = df.groupby(df['date'])['CE Trade_Exit_Status'].cummax()
#         df['PE Trade_Exit_Status'] = df.groupby(df['date'])['PE Trade_Exit_Status'].cummax()
#         #df['CE pnl'] = df.groupby(df['date'])['CE pnl'].cummax()
#         #df['PE pnl'] = df.groupby(df['date'])['PE pnl'].cummax()
#         #df['PnL Combined'] = df['CE pnl'] + df['PE pnl']
        

#         df['Date_Temp'] = df['date'].astype(str)

#         trades = df.copy(deep = True)

#         threshold_combinations = [
#             combo for combo in product(both_legs_open_loss_range, one_leg_open_range, both_legs_open_profit_range)
#             if len(set(combo)) == len(combo) and abs(combo[1]) < abs(combo[2]) and combo[2] !=0 # Unique values and one_leg_open < both_legs_open_loss
#         ]
#         for threshold_min, threshold_max_1, threshold_max_2 in threshold_combinations:
#             total_pnl = 0
#             daily_pnls = []
#             results = []

#             grouped = df.groupby('date')
#             for date, group in grouped:
#                 current_threshold_min = round(threshold_min,2)
#                 current_threshold_max_1 = round(threshold_max_1,2)
#                 current_threshold_max_2 = round(threshold_max_2,2)

#                 min_crossing_count = 0
#                 max_1_crossing_count = 0
#                 max_2_crossing_count = 0
#                 daily_pnl = 0
#                 group = group.reset_index(drop=True)

#                 while True:
#                     # Find crossing rows for all thresholds
#                     min_crossing_row = group[(group['PnL Combined'] <= current_threshold_min)]
#                     max_1_crossing_row = group[(((group['CE Trade_Exit_Status'] != 0) & (group['PE Trade_Exit_Status'] == 0)) | ((group['CE Trade_Exit_Status'] == 0) & (group['PE Trade_Exit_Status'] != 0))) & (group['PnL Combined'] >= current_threshold_max_1)] 
#                     max_2_crossing_row = group[(group['CE Trade_Exit_Status'] == 0) & (group['PE Trade_Exit_Status'] == 0) & (group['PnL Combined'] >= current_threshold_max_2)]
#                     fin_pnl = group[(group['CE Trade_Exit_Status'] == 1) & (group['PE Trade_Exit_Status'] == 1)]['PnL Combined']

#                     # Get the earliest crossing index for each threshold
#                     crossings = {
#                         'min': min_crossing_row.index[0] if not min_crossing_row.empty else float('inf'),
#                         'max_1': max_1_crossing_row.index[0] if not max_1_crossing_row.empty else float('inf'),
#                         'max_2': max_2_crossing_row.index[0] if not max_2_crossing_row.empty else float('inf')
#                     }

#                     # Find the first crossing threshold
#                     earliest_crossing = min(crossings, key=crossings.get)
#                     earliest_index = crossings[earliest_crossing]

#                     if earliest_index == float('inf'): 
#                         # print(filename, date)
#                         group = group[group['PnL Combined'] != 0]
#                         if not group.empty:
#                             daily_pnl = group['PnL Combined'].iloc[-1] 
#                             break
#                         else:
#                             daily_pnl = 0
#                             break
#                         # if earliest_index == float('inf'): 
#                         #     daily_pnl = group['PnL Combined'].iloc[-1] if group['PnL Combined'].iloc[-1] !=0 else fin_pnl.item()# Use the last value
#                         #     break
#                         # # No crossing occurred
#                         # daily_pnl = group['PnL Combined'].iloc[-1] if group['PnL Combined'].iloc[-1] !=0 else fin_pnl.item()# Use the last value
                        

#                     # Update PnL and counts based on the first crossing
#                     if earliest_crossing == 'min':
#                         daily_pnl = group.loc[earliest_index, 'PnL Combined']
#                         min_crossing_count += 1
#                         break
#                     elif earliest_crossing == 'max_1':
#                         daily_pnl = group.loc[earliest_index, 'PnL Combined']
#                         max_1_crossing_count += 1
#                         break
#                     elif earliest_crossing == 'max_2':
#                         daily_pnl = group.loc[earliest_index, 'PnL Combined']
#                         max_2_crossing_count += 1
#                         break
#                     else:
#                         daily_pnl = fin_pnl
#                         break

#                 daily_pnls.append(daily_pnl)
#                 results.append({
#                     'Date': date,
#                     'Daily PnL': daily_pnl,
#                     'No of times Min PnL crossed': min_crossing_count,
#                     'No of times Max_1 PnL crossed': max_1_crossing_count,
#                     'No of times Max_2 PnL crossed': max_2_crossing_count
#                 })

#             final_metrics_df = pd.DataFrame(results)
#             max_1_label = f"{threshold_max_1:.2f}"
#             max_2_label = f"{threshold_max_2:.2f}" 
#             min_label = f"{abs(threshold_min):.2f}" 
#             name = filename.split('.c')[0]
#             final_folder = f'{content_folder}/{name}/'
#             os.makedirs(final_folder, exist_ok=True)

#             output_filename = f'{final_folder}/portfolio_1p{max_1_label}_2p{max_2_label}_2l{min_label}.csv'
            

#             # if ('0k' != min_label) | ('0L' == max_1_label and '0L' == max_2_label):
#             final_metrics_df.to_csv(output_filename, index=False)
#             print(f'Results saved to {output_filename}')# Print the filename
