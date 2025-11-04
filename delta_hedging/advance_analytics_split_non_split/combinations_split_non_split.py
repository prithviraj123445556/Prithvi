# import pandas as pd
# import os
# import glob
# from itertools import product
# from helpers import analytics

# #margin = 2lakh
# #margin = 2.2lakh friday


# folder_path = "/home/newberry3/Delta Hedging/NIFTY/Trade_Sheets/No_Strike_Split/"
# file_paths = glob.glob(os.path.join(folder_path, "*.csv"))

# trade_data = {}

# for file_path in file_paths:
#     target_point = int(file_path.split("_")[-1].split(".")[0])  
#     df = pd.read_csv(file_path)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df[df['Date'] >= '2022-01-07']
#     df['Weekday'] = df['Date'].dt.strftime("%A")
#     df['PnL'] = (df['Exit Premium'] - df['Entry Premium']) * df['Quantity']
#     df['Brokerage'] = 6*df['Quantity']
#     df['Entry Slippage'] = (0.2/100)*df['Entry Premium']
#     df['Realised PnL'] = df['PnL'] - df['Brokerage'] - df['Entry Slippage']
        
#     if target_point not in trade_data:
#         trade_data[target_point] = df
#     else:
#         trade_data[target_point] = pd.concat([trade_data[target_point], df])

# aggregated_results = {}

# # Iterate over target points and aggregate PnL for each weekday
# for target, df in trade_data.items():
#     grouped_pnl = df.groupby('Weekday')['PnL'].sum().reset_index()
#     for _, row in grouped_pnl.iterrows():
#         weekday = row['Weekday']
#         pnl = row[' Realised PnL']
#         if (weekday, target) not in aggregated_results:
#             aggregated_results[(weekday, target)] = 0
#         aggregated_results[(weekday, target)] += pnl


# results_df = pd.DataFrame(
#     [(weekday, target, pnl) for (weekday, target), pnl in aggregated_results.items()],
#     columns=['Weekday', 'Target_Point', 'Total_PnL']
# )

# results_df = results_df.sort_values(by="Total_PnL", ascending=False)
# results_df.to_csv("weekday_target_pnl.csv", index=False)

# def run_analytics(processed_dfs):
#     output_df = pd.DataFrame() 
#     result_df = analytics(df)
#     output_df = pd.concat([output_df, result_df], ignore_index=True)
#     output_file_path = '/home/newberry3/Delta Hedging/analytics.xlsx'
#     output_df.to_excel(output_file_path, index=False)

# file_path = '/home/newberry3/total trades.csv'
# df = pd.read_csv(file_path)

# run_analytics(df)
# print("All analytics calculations completed and saved.")




##########done from may 2022 #############

# import pandas as pd
# import os
# import glob
# from helpers import analytics
# superset = 'delta_hedging'
# stock = 'NIFTY'
# option_type = 'ND'
# split_type = 'non_split'

# root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
# folder_path = rf'{root_path}/dailypnl_{split_type}/'
# file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))

# trade_data = {}

# for file_path in file_paths:
#     target_point = float(file_path.split("_")[-1].split(".")[0].replace(",", "."))
#     df = pd.read_excel(file_path)
#     print("file_path", file_path)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df[df['Date'] >= '2021-06-01']
#     df['Weekday'] = df['Day']
#     # df['PnL'] = (
#     #         ((df['Exit Premium'] - df['Entry Premium']) * df['Quantity']) +
#     #         ((df['Hedge Exit Premium'] - df['Hedge Entry Premium']) * df['Hedge Quantity'])
#     #     )
#     # df['Brokerage'] = 6 * df['Quantity']
#     # df['Entry Slippage'] = (0.2 / 100) * df['Entry Premium']
#     # df['Realised PnL'] = df['PnL'] - df['Brokerage'] - df['Entry Slippage']
        
#     if target_point not in trade_data:
#         trade_data[target_point] = df 
#     else:
#         trade_data[target_point] = pd.concat([trade_data[target_point], df])

# aggregated_results = []

# for target, df in trade_data.items():
#     grouped_pnl = df.groupby('Weekday')['Pnl'].sum().reset_index()
#     print("grouped_pnl", grouped_pnl)
#     for _, row in grouped_pnl.iterrows():
#         weekday = row['Weekday']
#         pnl = row['Pnl']
#         analytics_results = analytics(df, 'Pnl')  # Run analytics on the data
#         analytics_results.insert(0, 'Weekday', weekday)
#         analytics_results.insert(1, 'Target_Point', target)
#         analytics_results.insert(2, 'Total_PnL', pnl)
#         aggregated_results.append(analytics_results)

# results_df = pd.concat(aggregated_results, ignore_index=True)
# results_df.to_csv(f"weekday_target_pnl_{split_type}.csv", index=False)

# output_file_path = rf'{root_path}/analytics_{split_type}.xlsx'
# results_df.to_excel(output_file_path, index=False)

# print("All analytics calculations completed and saved.")






##########done from may 2022 #############


# import pandas as pd
# import os
# import glob
# from helpers import analytics , compute_final_z_scores

# superset = 'delta_hedging'
# stock = 'NIFTY'
# option_type = 'ND'
# split_type = 'non_split'

# root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
# folder_path = rf'{root_path}/dailypnl_{split_type}/'
# file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))

# aggregated_results = []

# for file_path in file_paths:
#     filename = os.path.basename(file_path).replace(".xlsx", "")
    
#     try:
#         # Example: NIFTY_candle_10T_hedge_0,2_target_0,6
#         parts = filename.split('_')
#         timeframe = parts[2]  # '10T'
#         hedge = parts[4]      # '0,2'
#         target_point = parts[-1].replace(",", ".")  # '0.6'
#         target_point = float(target_point)
#     except Exception as e:
#         print(f"Skipping file {filename}, unable to extract metadata: {e}")
#         continue

#     df = pd.read_excel(file_path)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df[df['Date'] >= '2022-05-01']
#     df['Weekday'] = df['Day']

#     # Group by weekday first
#     for weekday, weekday_df in df.groupby('Weekday'):
#         weekday_df = weekday_df.copy()

#         # 🔽 Sort by date before passing to analytics
#         weekday_df = weekday_df.sort_values('Date')

#         total_pnl = weekday_df['Pnl'].sum()

#         # Run analytics for just this weekday
#         analytics_results = analytics(weekday_df, 'Pnl')

#         # Add metadata columns
#         analytics_results.insert(0, 'Target_Point', target_point)
#         analytics_results.insert(1, 'Hedge', hedge)
#         analytics_results.insert(2, 'Timeframe', timeframe)
#         analytics_results.insert(3, 'Weekday', weekday)
#         analytics_results.insert(4, 'Total_PnL', total_pnl)

#         aggregated_results.append(analytics_results)


# # Final aggregation
# results_df = pd.concat(aggregated_results, ignore_index=True)

# # Save to files
# # results_df.to_csv(f"weekday_target_pnl_{split_type}.csv", index=False)
# final_sorted_df = compute_final_z_scores(results_df)

# output_file_path = rf'{root_path}/analytics_{split_type}.xlsx'
# results_df.to_excel(output_file_path, index=False)

# print("All analytics calculations completed and saved.")





import pandas as pd
import os
import glob
from helpers import analytics , compute_final_z_scores
from datetime import datetime, timedelta


superset = 'delta_hedging'
stock = 'NIFTY'
option_type = 'ND'
split_type = 'split'

# root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
strategy = 'NIFTY_apeksha'
root_path = rf"/home/newberry4/jay_test/delta_hedging/{strategy}/ND"
drawdown_folder = f'jay_test/delta_hedging/{strategy}/1_Min_Pnl/{stock}/1_min_pnl/'
folder_path = rf'{root_path}/dailypnl/{split_type}/'
file_paths = glob.glob(os.path.join(folder_path, "*.csv"))

aggregated_results = []


for file_path in file_paths:
    filename = os.path.basename(file_path).replace(".csv", "")
    
    try:
        # Example: NIFTY_candle_10T_hedge_0,2_target_0,6
        parts = filename.split('_')
        timeframe = parts[2]  # '10T'
        hedge = parts[4]      # '0,2'
        target_point = parts[-1].replace(",", ".")  # '0.6'
        target_point = float(target_point)
    except Exception as e:
        print(f"Skipping file {filename}, unable to extract metadata: {e}")
        continue

    df = pd.read_excel(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= '2022-05-01']                     #### added since its positive from here ###
    start_exclude = datetime.strptime("2024-05-31", "%Y-%m-%d").date()
    end_exclude = datetime.strptime("2024-06-06", "%Y-%m-%d").date()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df = df[~((df['Date'] >= start_exclude) & (df['Date'] <= end_exclude))]

    elif 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
        df = df[~((df['entry_date'].dt.date >= start_exclude) &
                                (df['entry_date'].dt.date <= end_exclude))]
        
    df['Weekday'] = df['Day']

    # Group by weekday first
    for weekday, weekday_df in df.groupby('Weekday'):
        weekday_df = weekday_df.copy()

        # 🔽 Sort by date before passing to analytics
        weekday_df = weekday_df.sort_values('Date')

        total_pnl = weekday_df['Pnl'].sum()

        # Run analytics for just this weekday
        analytics_results = analytics(weekday_df, 'Pnl')

        # Add metadata columns
        analytics_results.insert(0, 'Filename', filename)
        analytics_results.insert(0, 'Target_Point', target_point)
        analytics_results.insert(1, 'Hedge', hedge)
        analytics_results.insert(2, 'Timeframe', timeframe)
        analytics_results.insert(3, 'Weekday', weekday)
        analytics_results.insert(4, 'Total_PnL', total_pnl)
        aggregated_results.append(analytics_results)


# Final aggregation
results_df = pd.concat(aggregated_results, ignore_index=True)
print(results_df)

# === Get top 10 per weekday using your compute_final_z_scores ===
top_strategies_list = []

for weekday, group_df in results_df.groupby('Weekday'):
    group_df = group_df.copy()
    print("group_df", group_df)
    scored_df = compute_final_z_scores(group_df)
    # top_10 = scored_df.head(10)
    top_strategies_list.append(scored_df)

final_top_10_df = pd.concat(top_strategies_list, ignore_index=True)

# === Save outputs ===
results_df.to_excel(rf'{root_path}/analytics_all_{split_type}.xlsx', index=False)
final_top_10_df.to_excel(rf'{root_path}/analytics_top10_{split_type}.xlsx', index=False)

print("✅ All analytics + top 10 per weekday saved successfully.")






