import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os, re


stock = 'NIFTY' # 'BANKNIFTY', 'NIFTY', 'FINNIFTY'
strategy = 'NIFTY' # typee = 'NO REENTRY'

# if typee  == 'REENTRY':
if stock == 'SENSEX':
    months = 20
    investment = 500000
    target_return_40_months = 0.09
elif stock == 'BANKNIFTY':
    months = 39
    investment = 150000
elif stock == 'NIFTY':
    months = 46
    investment = 450000
    target_return_40_months = 0.18
elif stock == 'FINNIFTY':
    months = 21
    investment = 130000


target_return_6_months = 0.03

# target_return_40_months = 0.18


final_df = pd.DataFrame()
final_strategies = pd.DataFrame()


folder_path = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl'
analytics_folder = f'/home/newberry4/jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/Analytics/'
# analytics_folder = f'/home/newberry4/jay_test/SHORT_STRADDLE/1_Min_Pnl/{stock}/Analytics/'








#####old ruchika logic #####

# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Initialize a DataFrame to collect results
# final_strategies = pd.DataFrame()

# # Iterate through each file in the analytics folder
# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# final_strategies = pd.DataFrame()


# all_data = []

# # 1. Load all files first and concatenate
# for filename in os.listdir(analytics_folder):
#     file_path = os.path.join(analytics_folder, filename)
#     df = pd.read_excel(file_path)
#     df['File Set'] = file_path.split('Analytics/')[1].split('.')[0]
#     all_data.append(df)

# # Concatenate all data into one DataFrame
# all_df = pd.concat(all_data, ignore_index=True)

# # 2. Calculate Sortino Ratios for all rows together
# all_df['6M SD Loss'] = all_df['6M SD Loss'].round(2)
# investment = investment  # your variable here
# target_return_6_months = target_return_6_months  # your variable here

# all_df['6M Sortino Ratio'] = ((all_df['Last 6M Total PnL'] / investment - target_return_6_months) / 
#                               (all_df['6M SD Loss'] / investment))

# all_df['40M Sortino Ratio'] = ((all_df['Last 40M Total PnL'] / investment - target_return_6_months) / 
#                                (all_df['40M SD Loss'] / investment))

# # 3. Filter out rows with negative Sortino Ratios globally
# all_df = all_df[(all_df['6M Sortino Ratio'] > 0) & (all_df['40M Sortino Ratio'] > 0)].copy()

# # 4. Define columns for log transform
# columns_to_standardize = ['6M Sortino Ratio', '40 Max Drawdown', '40M Sortino Ratio']

# # 5. Apply log transform on whole dataset
# all_df_logs = all_df[columns_to_standardize].applymap(lambda x: np.log10(x) if x > 0 else np.nan)

# # Drop rows with NaNs from log transform
# all_df_logs = all_df_logs.dropna()
# all_df = all_df.loc[all_df_logs.index]  # keep df rows aligned with logs

# # 6. Apply MinMax scaling on entire log-transformed dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# all_scaled_logs = pd.DataFrame(
#     scaler.fit_transform(all_df_logs),
#     columns=[f'{col}_Log' for col in columns_to_standardize],
#     index=all_df_logs.index
# )

# # 7. Invert drawdown scaled log
# all_scaled_logs['40 Max Drawdown_Log'] = 1 - all_scaled_logs['40 Max Drawdown_Log']

# # 8. Join scaled logs back to main df
# all_df = all_df.join(all_scaled_logs)

# # 9. Calculate final weighted score (weights can be adjusted)
# weights = {
#     '6M Sortino Ratio_Log': 1,
#     '40 Max Drawdown_Log': 1,
#     '40M Sortino Ratio_Log': 1
# }

# all_df['Final Weighted Score'] = all_df[list(weights.keys())].mul(pd.Series(weights)).sum(axis=1)

# # 10. Sort globally by weighted score descending
# all_df_sorted = all_df.sort_values(by='Final Weighted Score', ascending=False)

# # 11. Save full combined & scored dataset
# all_df_sorted.to_excel(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_premium.xlsx', index=False)


#####old ruchika logic #####








##### for timewise analysis

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize a dictionary to store data grouped by time
time_grouped_data = {}

# 1. Load all files first and concatenate
for filename in os.listdir(analytics_folder):
    file_path = os.path.join(analytics_folder, filename)
    df = pd.read_excel(file_path)
    df['File Set'] = file_path.split('Analytics/')[1].split('.')[0]              ### for analytics 3

    # 2. Extract time from 'File Set' using regex to capture time like '13,00'
    time = df['File Set'].str.extract(r'_time_(\d{2},\d{2})')[0].unique()[0]

    # 3. Group data by time (create time-specific sheets)
    if time not in time_grouped_data:
        time_grouped_data[time] = []

    time_grouped_data[time].append(df)

# Process each time group
for time, group_files in time_grouped_data.items():
    # Concatenate all data for this time period
    all_df = pd.concat(group_files, ignore_index=True)
    print(f"Processing time group: {time} with {len(all_df)} records")

    # 4. Handle outliers in '6M SD Loss' before calculating Sortino ratios
    if len(all_df['6M SD Loss'].unique()) > 1:  # Ensure there are at least 2 unique values
        sorted_sd_loss = np.sort(all_df['6M SD Loss'].unique())
        second_lowest_sd = sorted_sd_loss[1]
        print(second_lowest_sd)  # second-lowest SD loss value

        # Identify rows where '6M SD Loss' is the minimum value (outliers)
        all_df['6M SD Loss'] = all_df['6M SD Loss'].apply(lambda x: second_lowest_sd if x == sorted_sd_loss[0] else x)

    # 5. Calculate Sortino Ratios after handling outliers
    investment = investment  # your variable here
    # target_return_6_months = target_return_6_months  # your variable here

    all_df['6M Sortino Ratio'] = ((all_df['Last 6M Total PnL'] / investment - target_return_6_months) / 
                                  (all_df['6M SD Loss'] / investment))

    all_df['40M Sortino Ratio'] = ((all_df['Last 40M Total PnL'] / investment - target_return_40_months) / 
                                   (all_df['40M SD Loss'] / investment))

    # 6. Filter out rows with negative Sortino Ratios globally
    all_df = all_df[(all_df['6M Sortino Ratio'] > 0) & (all_df['40M Sortino Ratio'] > 0)].copy()

    # 7. Apply log transformation on the columns
    columns_to_standardize = ['6M Sortino Ratio', '40 Max Drawdown', '40M Sortino Ratio','Win %']
    df_logs = all_df[columns_to_standardize].applymap(lambda x: np.log10(x) if x > 0 else np.nan)

    # 8. Drop rows with NaNs from log transformation
    df_logs = df_logs.dropna()
    all_df = all_df.loc[df_logs.index]  # Keep df rows aligned with logs

    # 9. Apply MinMax scaling on the log-transformed data
    if df_logs.empty:
        print("No data left after dropping Nans.")
        continue
        # Handle gracefully here
        # e.g.: continue, break, or raise a custom error
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled_logs = pd.DataFrame(
            scaler.fit_transform(df_logs),
            columns=[f'{col}_Log' for col in columns_to_standardize],
            index=df_logs.index
        )

    # 10. Invert '40 Max Drawdown_Log' (lower drawdown is better)
    df_scaled_logs['40 Max Drawdown_Log'] = 1 - df_scaled_logs['40 Max Drawdown_Log']

    # 11. Join scaled logs back to the original df
    all_df = all_df.join(df_scaled_logs)

    # 12. Add raw and scaled values to the df for sorting and weights
    all_df['6M Sortino Ratio_Log_Raw'] = df_logs['6M Sortino Ratio']
    all_df['40M Sortino Ratio_Log_Raw'] = df_logs['40M Sortino Ratio']

    # 13. Calculate final weighted score (weights can be adjusted)
    weights = {
        '6M Sortino Ratio_Log': 1,
        '40 Max Drawdown_Log': 1,
        '40M Sortino Ratio_Log': 1,
        'Win %_Log': 1
    }

    all_df['Final Weighted Score'] = all_df[list(weights.keys())].mul(pd.Series(weights)).sum(axis=1)

    # 14. Sort by the final weighted score (descending)
    all_df_sorted = all_df.sort_values(by='Final Weighted Score', ascending=False)

    # 15. Append the sorted data to the final strategies DataFrame
    if not all_df_sorted.empty:
        final_strategies = pd.concat([final_strategies, all_df_sorted], ignore_index=True)

# 16. Save the results for each time group in separate sheets
with pd.ExcelWriter(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_premium_timewise_new.xlsx') as writer:
    # Iterate through each time group in time_grouped_data
    for time, group_files in time_grouped_data.items():
        # Filter out rows from the final strategies DataFrame for the current time group
        time_df_sorted = final_strategies[final_strategies['File Set'].str.contains(time)]
        
        # Sort the dataframe by 'Final Weighted Score' in descending order and get the top row
        # top_row = time_df_sorted.sort_values(by='Final Weighted Score', ascending=False).head(1)
        top_row = time_df_sorted.sort_values(by='Final Weighted Score', ascending=False)       ### all 
        print("top_row is ", top_row)
        print("top_row is ", top_row)

        print("completed")
        
        # Save the top row to the respective sheet
        top_row.to_excel(writer, sheet_name=f'Time_{time}_Top', index=False)













