import time,sys
sys.path.insert(0, r"/home/newberry4/jay_data/")
from Common_Functions.modified_utils import compare_month_and_year, postgresql_query
import pandas as pd
import os
from datetime import timedelta


def find_hedge_strike(option_data, option_type, target_premium, time):
    """
    Find the strike price for hedging based on target premium.
    """
    # if option_type == 'CE':
    #     eligible_strikes = option_data[option_data['StrikePrice'] > base_strike]
    # else:  # 'Put'
    #     eligible_strikes = option_data[option_data['StrikePrice'] < base_strike]
    print('option_data:', option_data)
    print('option_type:', option_type)
    
    option_data = option_data[(option_data['Type'] == option_type) & (option_data['Time'] == time)]
    if option_data.empty:
        while option_data[(option_data['Type'] == option_type) & (option_data['Time'] == time)].empty:
            time += timedelta(minutes=1)
        option_data = option_data[(option_data['Type'] == option_type) & (option_data['Time'] == time)]
        
    hedge_strike = option_data.iloc[(option_data['Open'] - target_premium).abs().argmin()]['StrikePrice']
    return hedge_strike



def find_hedge_strike2(option_data, option_type, target_premium, time):
    """
    Find the strike price for hedging based on target premium.
    Uses the index instead of 'Time' column for filtering.
    If the exact time is missing, it increments by 1 minute until a match is found.
    """
    # Ensure index is a DatetimeIndex
    print("time", time)
    # print("target_premium", target_premium)
    if not isinstance(option_data.index, pd.DatetimeIndex):
        option_data = option_data.set_index(pd.to_datetime(option_data.index))
    # print('option_data:', option_data)
    # print('option_type:', option_type)
    # Filter based on option type
    option_data_filtered = option_data[option_data['Type'] == option_type]
    # print('option_data_filtered:', option_data_filtered)

    # If option_data_filtered is empty, return None
    if option_data_filtered.empty:
        print(f"Warning: No data available for {option_type}. Returning None.")
        return None

    # Get the last available timestamp to prevent OutOfBounds errors
    last_available_time = option_data_filtered.index[-1]

    # Increment time by 1 minute until a valid timestamp is found
    while time not in option_data_filtered.index:
        print("yesss")
        time += timedelta(minutes=1)

        # Stop if we exceed the last available timestamp
        if time > last_available_time:
            print(f"Warning: No valid strike price found for {option_type} at or after {time}")
            return None

    # Filter for the available time
    option_data_filtered = option_data_filtered.loc[time]
    # print('option_data_filtered:', option_data_filtered)

    # Find the closest strike price based on target premium
    hedge_strike = option_data_filtered.iloc[
        (option_data_filtered['Open'] - target_premium).abs().argmin()
    ]['StrikePrice']
    
    return hedge_strike



def is_expiry(timestamp, expiry_date):
    # Check if the current timestamp is the expiry date
    print("timestamp", timestamp)   
    print("expiry_date", expiry_date)
    return timestamp == pd.to_datetime(expiry_date).date()


def resample_data(data, TIME_FRAME):
    # Ensure the index is a DateTimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Filter data to start from 9:15 AM
    data = data[data.index.time >= pd.to_datetime("09:15").time()]

    # Resample with an offset to start from 9:15 AM
    resampled_data = data.resample(TIME_FRAME, origin="start_day", offset="15min").agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'ExpiryDate': 'first'
    }).dropna()

    # Add Date and Time columns
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    return resampled_data


def pull_options_data_d(start_date, end_date, option_data_path, stock):

    start_time = time.time()
    option_data_files = next(os.walk(option_data_path))[2]
    option_data = pd.DataFrame()

    for file in option_data_files:

        file1 = compare_month_and_year(start_date, end_date, file, stock)
              
        if not file1:
            continue

        temp_data = pd.read_pickle(option_data_path + file)[['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open','Low','Close' , 'High', 'Ticker']]
        temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
        temp_data = temp_data.rename_axis('DateTime')
        option_data = pd.concat([option_data, temp_data])

    print('Option data columns :', option_data.columns)
    option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
    option_data['Type'] = option_data['Type'].astype('category')
    option_data = option_data.sort_index()
    end_time = time.time()
    print('Time taken to pull Options data :', (end_time-start_time))

    return option_data





def pull_index_data(start_date_idx, end_date_idx, stock, mapped_days):

    start_time = time.time()
    print(start_date_idx, end_date_idx)
    table_name = stock + '_IDX'
    data = postgresql_query(f'''
                            SELECT "Open", "High", "Low", "Close", "Ticker"
                            FROM "{table_name}"
                            WHERE "Date" >= '{start_date_idx}'
                            AND "Date" <= '{end_date_idx}'
                            AND "Time" BETWEEN '09:15' AND '15:29'
                            ''')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time taken to get Index Data:', elapsed_time)

    column_names = ['Open', 'High', 'Low', 'Close', 'Ticker']
    index_data = pd.DataFrame(data, columns = column_names)
    index_data['Date'] = pd.to_datetime(index_data['Ticker'].str[0:8], format = '%Y%m%d').astype(str)

    df = index_data.merge(mapped_days, on = 'Date')
    df.index = pd.to_datetime(df['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df


def get_latest_price(index_data):
    """
    Returns the latest index price from the index data.
    """
    latest_row = index_data.iloc[-1]  # Get the most recent row
    return latest_row['Open']


def get_drawdown(Date, PnL):
    max_drawdown = 0
    max_drawdown_percentage = 0
    max_drawdown_date = None
    time_to_recover = 0
    peak_date_before_max_drawdown = None
    cum_pnl = 0
    peak = 0
    peak_date = Date.iloc[0]

    for date, pnl in zip(Date, PnL):
        cum_pnl += pnl
        if (time_to_recover is None) and (cum_pnl >= peak):
            time_to_recover = (date - peak_date).days
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

    return max_drawdown, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown

def minPnl(df):
    monthly36 = {'start': '2022-01-07', 'end': '2024-12-19'}
    monthly18 = {'start': '2023-06-01', 'end': '2024-12-19'}
    monthly4 = {'start': '2024-08-01', 'end': '2024-12-19'}

    def calculate_monthly_pnl(Date, df):
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'Premium'].tolist()

        return sum(monthly_pnls) 

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly36_pnl = calculate_monthly_pnl(monthly36, df) 
    Dmonthly18_pnl = calculate_monthly_pnl(monthly18, df)
    Dmonthly4_pnl = calculate_monthly_pnl(monthly4, df)

    #all_monthly_pnls = [Dmonthly36_pnl, Dmonthly12_pnl]
    #all_pnls = [pnl for pnl in all_monthly_pnls if pnl is not None]

    return Dmonthly36_pnl, Dmonthly18_pnl, Dmonthly4_pnl


def analytics(sub_dataframe_dailypnl, x):
    max_investment = 15744000
    sub_dataframe_dailypnl['Pnl'] = pd.to_numeric(sub_dataframe_dailypnl[x], errors='coerce')
    i = 1
    # if sheet_name.endswith('.csv'):
    if i == 1:
        sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], format ='mixed')
        sub_dataframe_dailypnl = sub_dataframe_dailypnl[sub_dataframe_dailypnl['Pnl'] != 0]

        totalpnl = (sub_dataframe_dailypnl['Pnl'].sum())
        print("totalpnl", totalpnl)

        # Call get_drawdown function
        max_drawdown,max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
            sub_dataframe_dailypnl['Date'],
            sub_dataframe_dailypnl['Pnl']
        )
        
        total_months = 40
        monthly_return = totalpnl / ( max_investment * total_months)
        minimum_return_needed = 0.003
        # sd_loss_ratio = sd_loss/max_investment
        # sortino_ratio = (monthly_return - minimum_return_needed) / sd_loss_ratio if sd_loss > 0 else 0
        # Dmonthly36_pnl, Dmonthly18_pnl, Dmonthly4_pnl = minPnl(sub_dataframe_dailypnl)
        sub_dataframe_dailypnl_18 = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['Date'] >= '2024-04-01')]
        Dmonthly18_pnl = sub_dataframe_dailypnl_18['Pnl'].sum()
        print("Dmonthly18_pnl", Dmonthly18_pnl)
        sub_dataframe_dailypnl_4 = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['Date'] >= '2024-11-01')]
        Dmonthly4_pnl = sub_dataframe_dailypnl_4['Pnl'].sum()

        # Additional operations
        Profits = sub_dataframe_dailypnl[ (sub_dataframe_dailypnl['Pnl'] > 0)]['Pnl']
        Losses = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['Pnl'] < 0)]['Pnl']

        total_trades = (len(sub_dataframe_dailypnl))
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


        sd_loss_ratio = sd_loss/max_investment
        sortino_ratio = (monthly_return - minimum_return_needed) / sd_loss_ratio if sd_loss > 0 else 0
        # max_investment = 15744000/5

        roi_with_dd=100*totalpnl/(max_investment + max_drawdown)
        roi_real = 100*totalpnl/max_investment
        #action = sub_dataframe_dailypnl[sub_dataframe_dailypnl['DaysToExpiry'] == days_to_expiry]['Type'].unique()
        
        result_df = pd.DataFrame({
            'totalpnl': [totalpnl],
            'Max Drawdown': [max_drawdown],
            '11M Monthly PnL' : [Dmonthly18_pnl],
            '6M Monthly PnL' : [Dmonthly4_pnl],
            'Max Investment': [max_investment],
            'Actual ROI % ' : [roi_real],
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
            'SD Loss': [sd_loss] ,
            'Sortino Ratio': [sortino_ratio]
        })
        return result_df
    


def analytics2(combined_df, combined_expiry_df, x):
    max_investment = 15744000

    # Convert and clean PnL for both DataFrames
    combined_df['Pnl'] = pd.to_numeric(combined_df[x], errors='coerce')
    combined_expiry_df['Pnl'] = pd.to_numeric(combined_expiry_df[x], errors='coerce')

    # Ensure Date column is in datetime format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed')
    combined_expiry_df['ExpiryDate'] = pd.to_datetime(combined_expiry_df['ExpiryDate'], format='mixed')

    # Remove rows where PnL is 0
    combined_df = combined_df[combined_df['Pnl'] != 0]
    combined_expiry_df = combined_expiry_df[combined_expiry_df['Pnl'] != 0]

    # === Metrics from combined_df ===
    totalpnl = combined_df['Pnl'].sum()
    total_months = 40
    monthly_return = totalpnl / (max_investment * total_months)
    minimum_return_needed = 0.003

    combined_df_18 = combined_df[combined_df['Date'] >= '2024-04-01']
    Dmonthly18_pnl = combined_df_18['Pnl'].sum()

    combined_df_4 = combined_df[combined_df['Date'] >= '2024-11-01']
    Dmonthly4_pnl = combined_df_4['Pnl'].sum()

    roi_real = 100 * totalpnl / max_investment

    # === Metrics from combined_expiry_df ===
    max_drawdown, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
        combined_expiry_df['ExpiryDate'],
        combined_expiry_df['Pnl']
    )

    Profits = combined_expiry_df[combined_expiry_df['Pnl'] > 0]['Pnl']
    Losses = combined_expiry_df[combined_expiry_df['Pnl'] < 0]['Pnl']

    total_trades = len(combined_expiry_df)
    num_winners = len(Profits)
    num_losers = len(Losses)

    win_percentage = 100 * num_winners / total_trades if total_trades > 0 else 0
    loss_percentage = 100 * num_losers / total_trades if total_trades > 0 else 0

    max_profit = Profits.max() if num_winners > 0 else 0
    max_loss = Losses.min() if num_losers > 0 else 0

    median_pnl = combined_expiry_df['Pnl'].median()
    median_profit = Profits.median() if num_winners > 0 else 0
    median_loss = Losses.median() if num_losers > 0 else 0

    sd_pnl = combined_expiry_df['Pnl'].std()
    sd_profit = Profits.std() if num_winners > 0 else 0
    sd_loss = Losses.std() if num_losers > 0 else 0

    sd_loss_ratio = sd_loss / max_investment
    sortino_ratio = (monthly_return - minimum_return_needed) / sd_loss_ratio if sd_loss > 0 else 0
    roi_with_dd = 100 * totalpnl / (max_investment + max_drawdown)

    # === Final result ===
    result_df = pd.DataFrame({
        'totalpnl': [totalpnl],
        'Max Drawdown': [max_drawdown],
        '11M Monthly PnL': [Dmonthly18_pnl],
        '6M Monthly PnL': [Dmonthly4_pnl],
        'Max Investment': [max_investment],
        'Actual ROI %': [roi_real],
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
        'Sortino Ratio': [sortino_ratio]
    })

    return result_df




from scipy.stats import zscore


def compute_final_z_scores(final_df):
    # Calculate ratios and z-scores for each required metric
    final_df['Dmonthly36_PnL_Ratio'] = final_df['Total_PnL'] / final_df['Max Investment']   ## 36 month 
    final_df['Max_Drawdown_Ratio'] = final_df['Max Drawdown'] / final_df['Max Investment']
    final_df['Dmonthly11_PnL'] = final_df['11M Monthly PnL']                                    ## 11 month
    final_df['Sortino Ratio'] = final_df['Sortino Ratio']

    # Calculate z-scores for each metric
    final_df['Dmonthly36_PnL_Z'] = zscore(final_df['Dmonthly36_PnL_Ratio'])
    final_df['Max_Drawdown_Z'] = zscore(final_df['Max_Drawdown_Ratio'])
    final_df['Dmonthly11_PnL_Z'] = zscore(final_df['Dmonthly11_PnL'])
    final_df['Sortino_Ratio_Z'] = zscore(final_df['Sortino Ratio'])

    # Standardize z-scores to range [0, 1]
    final_df['Dmonthly36_PnL_Z'] = (final_df['Dmonthly36_PnL_Z'] - final_df['Dmonthly36_PnL_Z'].min()) / (final_df['Dmonthly36_PnL_Z'].max() - final_df['Dmonthly36_PnL_Z'].min())
    final_df['Max_Drawdown_Z'] = (final_df['Max_Drawdown_Z'] - final_df['Max_Drawdown_Z'].min()) / (final_df['Max_Drawdown_Z'].max() - final_df['Max_Drawdown_Z'].min())
    final_df['Dmonthly11_PnL_Z'] = (final_df['Dmonthly11_PnL_Z'] - final_df['Dmonthly11_PnL_Z'].min()) / (final_df['Dmonthly11_PnL_Z'].max() - final_df['Dmonthly11_PnL_Z'].min())
    final_df['Sortino_Ratio_Z'] = (final_df['Sortino_Ratio_Z'] - final_df['Sortino_Ratio_Z'].min()) / (final_df['Sortino_Ratio_Z'].max() - final_df['Sortino_Ratio_Z'].min())

    # Apply weights to the z-scores, using (1 - Z) approach for drawdowns
    final_df['Final_Z_Score'] = (
        1 * final_df['Sortino_Ratio_Z'] + 
        1 * final_df['Dmonthly36_PnL_Z'] + 
        1 * (1 - final_df['Max_Drawdown_Z']) + 
        0.5 * final_df['Dmonthly11_PnL_Z']
    )

    final_df = final_df.sort_values(by='Final_Z_Score', ascending=False).reset_index(drop=True)

    # Filter strategies where Sortino Ratio > 1
    # top_strategies = final_df[final_df['Sortino Ratio'] > 1].copy()

    # if top_strategies.empty:
    #     print("[WARNING] No strategy with Sortino Ratio > 1 found.")
    top_strategies = final_df.head(10).copy()
    # else:
    #     top_strategies = top_strategies.head(10).copy()

#     top_strategies['Strategy_Name'] = (
#     "Target_" + top_strategies['Target_Point'].astype(str) + "_" +
#     "Hedge_" + top_strategies['Hedge'].astype(str) + "_" +
#     "TF_" + top_strategies['Timeframe'].astype(str) + "_" +
#     "WD_" + top_strategies['Weekday'].astype(str) + "_" +
#     top_strategies['Filename']
# )
    top_strategies['Strategy_Name'] = (
    top_strategies['Filename'] + "_" +
    "WD_" + top_strategies['Weekday'].astype(str)
)

    print("[INFO] Top 10 strategy names (Sortino Ratio > 1):")
    print(top_strategies['Strategy_Name'].tolist())

    return top_strategies


# Usage
# Assuming `existing_df` is the final DataFrame containing all selected strategies after appending each strategy
# final_sorted_df = compute_final_z_scores(existing_df)
