import pandas as pd 
import numpy as np
import os
import re
import pandas as pd
from itertools import product
import sys
# from Targets_utils import get_drawdown, minPnl, analytics
import pandas as pd
import numpy as np
import os
import re

stock = 'SENSEX'
LOT_SIZE = 75 if stock == 'NIFTY' else 20 

strategy = 'SENSEX'
# root_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{strategy}/'

folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
os.makedirs(analytics_folder, exist_ok=True)
os.makedirs(content_folder, exist_ok=True)


# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        if 'PnL Combined' not in df.columns:
            continue

        max_threshold = df['PnL Combined'].max()
        min_threshold = df['PnL Combined'].min()
        # first_range = np.arange(min_threshold, max_threshold + 1, 100)
        second_range = np.linspace(min_threshold, max_threshold, 20)
        # min_val_range = first_range if len(first_range) <= len(second_range) else second_range
        min_val_range = second_range
        min_val_range = np.round(min_val_range, 2)

        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df['date'] = df['DateTime'].dt.date

        df['Date_Temp'] = df['date'].astype(str)
        trades = df.copy(deep=True)

        for threshold in min_val_range:
            results = []
            grouped = df.groupby('date')
            for date, group in grouped:
                group = group.reset_index(drop=True)
                group = group[group['PnL Combined'] != 0]

                crossing_row = group[group['PnL Combined'] >= threshold]
                if not crossing_row.empty:
                    daily_pnl = crossing_row.iloc[0]['PnL Combined']
                else:
                    daily_pnl = group['PnL Combined'].iloc[-1] if not group.empty else 0

                results.append({
                    'Date': date,
                    'Daily PnL': daily_pnl
                })

            final_metrics_df = pd.DataFrame(results)
            threshold_label = f"{threshold:.2f}"
            name = filename.split('.c')[0]
            final_folder = f'{content_folder}/{name}/'
            os.makedirs(final_folder, exist_ok=True)

            output_filename = f'{final_folder}/pnl_threshold_{threshold_label}.csv'
            final_metrics_df.to_csv(output_filename, index=False)
            print(f'Results saved to {output_filename}')





















##### tsl logic ###############      tried not useful
# import pandas as pd
# import numpy as np
# import os

# stock = 'NIFTY'
# LOT_SIZE = 75 if stock == 'NIFTY' else 20 

# strategy = 'NIFTY_apeksha_new'
# folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
# analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
# content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content/'
# # content_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Content2/'

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
