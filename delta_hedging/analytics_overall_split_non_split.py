import pandas as pd
import os
existing_df = pd.DataFrame()
#tradesheet_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/tradesheet_dte0.csv')
#pnl_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/pnl_dte0_dtecol.csv')
#output_folder = "/home/newberry2/vix2_analytics_report/"


def minPnl(Date, df):
    monthly31 = {'start': '2021-06-01', 'end': '2025-02-28'}
    monthly11 = {'start': '2024-02-01', 'end': '2025-02-28'}
    monthly3 = {'start': '2024-11-01', 'end': '2025-02-28'}

    def calculate_monthly_pnl(Date, df):
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'Pnl'].tolist()

        return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly31_pnl = calculate_monthly_pnl(monthly31, df) / 44
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
import pandas as pd

def get_drawdown(Date, PnL):
    # Ensure Date is in datetime format
    Date = pd.to_datetime(Date)

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
    total_months = 44
    #sheet_name = 'analytics_report_alldte'
    sub_dataframe_dailypnl['Pnl'] = pd.to_numeric(sub_dataframe_dailypnl['Pnl'], errors='coerce')
    '''
    if sheet_name.startswith('dte_'):
        days_to_expiry = int(sheet_name.split('_')[1])  # Extract the numeric part from sheet_name
    '''    
    # Convert 'Date' column to datetime format if not already in datetime
    '''sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], errors='coerce')'''
    sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], errors='coerce')

    totalpnl = sub_dataframe_dailypnl['Pnl'].sum()

    # Call get_drawdown function
    max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
        sub_dataframe_dailypnl['Date'],
        sub_dataframe_dailypnl['Pnl']
    )

    overall_min_pnl,Dmonthly31_pnl,Dmonthly11_pnl,Dmonthly3_pnl = minPnl(sub_dataframe_dailypnl['Date'],sub_dataframe_dailypnl)

    # Additional operations
    Profits = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Pnl'] > 0]['Pnl']
    Losses = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Pnl'] <= 0]['Pnl']

    total_trades = len(sub_dataframe_dailypnl)
    num_winners = len(Profits)
    num_losers = len(Losses)
    win_percentage = 100 * num_winners / total_trades
    loss_percentage = 100 * num_losers / total_trades

    max_profit = Profits.max() if num_winners > 0 else 0
    max_loss = Losses.min() if num_losers > 0 else 0

    median_pnl = sub_dataframe_dailypnl['Pnl'].median()
    median_profit = Profits.median() if num_winners > 0 else 0
    median_loss = Losses.median() if num_losers > 0 else 0

    sd_pnl = sub_dataframe_dailypnl['Pnl'].std()
    sd_profit = Profits.std() if num_winners > 0 else 0
    sd_loss = Losses.std() if num_losers > 0 else 0

    # Monthly PnL calculation
    # monthly_pnl = totalpnl / total_months
    # max investment Directional
    '''max_investment = -((sub_dataframe_tradesheet[sub_dataframe_tradesheet['DaysToExpiry'] == days_to_expiry].loc[sub_dataframe_tradesheet['Action'] == 'long', 'Premium'].min()) * 50)'''
    # max_investment = sub_dataframe_dailypnl['ENTRY Prm'].max() * sub_dataframe_dailypnl['Qty'][0]
    max_investment = max_investment


    #  ROI = (PNL - highest profit)/(Investment + Drawdown)
    # roi_with_dd=100*overall_min_pnl*total_months/(max_investment + max_drawdown)

    roi_with_dd = 100 * (totalpnl-max_profit)/(max_investment + max_drawdown)
    roi = 100*overall_min_pnl*total_months/max_investment
    #action = sub_dataframe_dailypnl[sub_dataframe_dailypnl['DaysToExpiry'] == days_to_expiry]['Type'].unique()
    

    # Create a DataFrame with file, Total PnL, Max Drawdown, and additional metrics
    result_df = pd.DataFrame({
        'Filename': [filename_str],
        'Total PnL': [totalpnl],
        'Max Drawdown': [max_drawdown],
        'Max Drawdown Percentage': [max_drawdown_percentage],
        '31M Monthly PnL' : [Dmonthly31_pnl],
        '11M Monthly PnL' : [Dmonthly11_pnl],
        'Daily 3M PnL ' : [Dmonthly3_pnl],
        'min pnl ' : [overall_min_pnl],
        'Max Investment': [max_investment],
        'ROI % ' : [roi],
        'ROI with DD' : [roi_with_dd],
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
        'SD Loss': [sd_loss]
    })
    '''
    # Save the DataFrame to an Excel file for each 'dte' sheet
    output_file_path = os.path.join(output_folder, f"{sheet_name}.xlsx")

    # If the file already exists, append the new result to it
    if os.path.isfile(output_file_path):
        existing_df = pd.read_excel(output_file_path)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_excel(output_file_path, index=False)
    '''
    
    existing_df = pd.concat([existing_df, result_df], ignore_index=True)
    return existing_df


def get_spread_from_range(value, stock, lot_size):
    if stock=='NIFTY' or stock=='FINNIFTY':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
        
    elif stock=='BANKNIFTY':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
          
    for key, range_tuple in range_dict.items():
        start, end = eval(key)
        if start <= abs(value) <= end:
            return abs(range_tuple * lot_size)
       
    return (abs(value * lot_size * 0.3) / 100)

lot_size_dict = {'NIFTY': 75,'FINNIFTY': 25,
            'BANKNIFTY': 15}
govt_tc_dict = {"NIFTY": 2.25, 'FINNIFTY': 2.25 ,
           "BANKNIFTY": 3}


stock = 'NIFTY'
max_investment = 15744000
type = 'split' 
alldte_pnl_files = f'/home/newberry4/jay_test/delta_hedging/{stock}/ND/Trade_Sheets/{type}/'
outputfolder_path = f'/home/newberry4/jay_test/delta_hedging/{stock}/ND/dailypnl/{type}/'
drawdown_folder = f'jay_test/delta_hedging/advance_analytics/1_Min_Pnl/NIFTY/1_min_pnl'
os.makedirs(outputfolder_path, exist_ok=True)
existing_df = pd.DataFrame()

# # Phase 1: Process and Save PnL Files
for root, dirs, files in os.walk(alldte_pnl_files):
    for file in files:
        file_path = os.path.join(root, file)
        print(f"Processing: {file_path}")
        
        try:
            pnl_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        if pnl_df.empty:
            print(f"Skipping empty file: {file}")
            continue

        # Compute PnL
        pnl_df['Pnl'] = (
    ((pnl_df['Exit Premium'].fillna(0) - pnl_df['Entry Premium'].fillna(0)) * pnl_df['Quantity'].fillna(0)) +
    ((pnl_df['Hedge Exit Premium'].fillna(0) - pnl_df['Hedge Entry Premium'].fillna(0)) * pnl_df['Hedge Quantity'].fillna(0))
)

        idx_calc = 'NIFTY'
        lot_size = lot_size_dict[idx_calc]
        govt_charge = govt_tc_dict[idx_calc]
        lots = 10

        pnl_list = []
        for idx, row in pnl_df.iterrows():
            row_premium = row['Pnl']
            row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size)) + govt_charge * lots
            row_pnl = (row_premium * lot_size) - row_tc 
            pnl_list.append(row_pnl)

        pnl_df['Pnl'] = pnl_list

        pnl_df = pnl_df.groupby(['Date']).agg({
            'Type': 'first',
            'ExpiryDate': 'first',
            'DaysToExpiry': 'first',
            'Day': 'first',
            'Pnl': 'sum'
        }).reset_index()

        if not os.path.exists(outputfolder_path):
            os.makedirs(outputfolder_path)

        output_file_path = os.path.join(outputfolder_path, file.replace('.csv', '.xlsx'))

        if os.path.exists(output_file_path):
            print(f"File {output_file_path} already exists. Skipping.")
            continue

        pnl_df.to_excel(output_file_path, index=False)
        print(f"Saved to {output_file_path}")

# Phase 2: Reload and Run Analytics
print("\n--- Starting Analytics Phase ---")
for file in os.listdir(outputfolder_path):
    if file.endswith('.xlsx'):
        file_path = os.path.join(outputfolder_path, file)
        try:
            pnl_df = pd.read_excel(file_path)
            filename_str = str(file)
            existing_df = analytics(pnl_df, existing_df, filename_str, max_investment)
        except Exception as e:
            print(f"Failed analytics for {file}: {e}")

# Save Final Analytics Output
analytics_output_path = '/home/newberry4/jay_test/delta_hedging/Analytics.xlsx'
existing_df.to_excel(analytics_output_path, index=False)
print(f"Analytics saved to: {analytics_output_path}")












# def get_spread_from_range(value, stock, lot_size):
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



# for file in files:
#     if file.endswith('.xlsx') or file.endswith('.xls'):
#         file_path = os.path.join(inputfolder_path, file)
#         #print(file_path)
#         df = pd.read_excel(file_path)
#         dfs[file] = df

# for file, df in dfs.items():
#     if file.startswith('NIFTY'):
#         idx_calc = 'NIFTY'
#     elif file.startswith("BANKNIFTY"):
#         idx_calc = 'BANKNIFTY'
#     elif file.startswith("FINNIFTY"):
#         idx_calc = 'FINNIFTY'

#     govt_charge = govt_tc_dict[idx_calc]
#     lot_size = lot_size_dict[idx_calc]

#     pnl_list = []
#     for idx, row in df.iterrows():
#         row_premium = row['Premium']
#         row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size)) + govt_charge
#         row_pnl = (row_premium * lot_size) - (row_tc)
#         pnl_list.append(row_pnl)

#     df['PnL'] = pnl_list

#     result_df = df.groupby(['Date', 'WeeklyDaysToExpiry']).agg({
#         'Type': 'first',
#         'ExpiryDate': 'first',
#         'PnL': 'sum'
#     }).reset_index()

#     if not os.path.exists(outputfolder_path):
#         os.makedirs(outputfolder_path)
    

#     # Save the result_df as an Excel file in the 'dailypnl/' folder
#     output_file_path = os.path.join(outputfolder_path, f'{file}')

#     if os.path.exists(output_file_path):
#         print(f"File {output_file_path} already exists. Skipping.")
#         continue

#     # selected_columns = ['Date', 'WeeklyDaysToExpiry', 'Type', 'ExpiryDate', 'PnL']
#     result_df.to_excel(output_file_path, index=False)