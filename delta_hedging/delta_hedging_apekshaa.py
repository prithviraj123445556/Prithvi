# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
# from datetime import datetime, timedelta
import os
from openpyxl import load_workbook
import ast
import json
from functools import partial
import warnings
import pandas as pd
from copy import deepcopy
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/newberry4')
import concurrent.futures
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
# sys.path.insert(0, r"/home/newberry3")
import multiprocessing
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
# from datetime import datetime, timedelta
import ast, json, sys, re
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
# from datetime import datetime, time
from copy import deepcopy
import datetime
import time




sys.path.insert(0, r"/home/newberry4/jay_data/")
from jay_data.Common_Functions.utils import TSL_PREMIUM, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes_d
from jay_data.Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, get_target, get_stoploss, compare_month_and_year





from helpers import find_hedge_strike, is_expiry, pull_options_data_d, pull_index_data, resample_data , find_hedge_strike2



# TODO: remove 30th may to 6th june
# TODO: find out max drawdown in the week minwise
# TODO: findout target points like short straddle
# TODO: signals from upstoxs





# take 33% hedge here also 


def black_scholes_price(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "PE":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(option_price, S, K, T, r, option_type):
    # print("option_price", option_price)
    # print("S", S)
    # print("K", K)
    # print("T", T)
    # print("r", r)
    # print("option_type", option_type)
    try:
        return brentq(lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_price, 0.01, 5.0)
    except (ValueError, RuntimeError):
        return None


def black_scholes_greeks(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "CE" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (
        (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        if option_type == "CE"
        else (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    )
    rho = (
        K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        if option_type == "CE"
        else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    )

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }


############old code not much precise######################
# def calculate_time_to_expiry(manual_datetime_str, expiry_date_str):
#     # print("manual_datetime_str" ,type(manual_datetime_str))
#     # print("expiry_date_str" ,type(expiry_date_str))
#     # now = datetime.strptime(manual_datetime_str, "%Y-%m-%d %H:%M:%S")
#     now = manual_datetime_str
#     expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

#     market_open = datetime.time(9, 15)
#     market_close = datetime.time(15, 30)
#     today = now.date()
#     days_left = (expiry_date - today).days
#     if days_left <= 0:
#         days_left += 1

#     total_trading_minutes = (market_close.hour * 60 + market_close.minute) - (market_open.hour * 60 + market_open.minute)
#     current_minutes_since_open = (now.hour * 60 + now.minute) - (market_open.hour * 60 + market_open.minute)

#     if current_minutes_since_open < 0:
#         T = round(days_left / 365, 6)
#     elif current_minutes_since_open >= total_trading_minutes:
#         T = round(max(0, (days_left - 1) / 365), 6)
#     else:
#         fraction_of_day_passed = current_minutes_since_open / total_trading_minutes
#         T = round((days_left - fraction_of_day_passed) / 365, 6)

#     return T



def calculate_time_to_expiry(current_dt, expiry_date_str):
    expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    print("current_dt", current_dt.date())
    if current_dt.date() > expiry_date:
        return 0.0

    total_minutes = 0.0
    if current_dt.date() == expiry_date:
        market_close = datetime.time(15, 30)
        if current_dt.time() >= market_close:
            return 0.0
        minutes_until_close = (market_close.hour*60 + market_close.minute) - (current_dt.hour*60 + current_dt.minute)
        total_minutes += minutes_until_close
    else:
        days_left = (expiry_date - current_dt.date()).days
        market_close = datetime.time(15, 30)
        market_open = datetime.time(9, 15)

        current_minutes_in_day = (current_dt.time().hour*60 + current_dt.time().minute)
        if current_minutes_in_day < (market_open.hour*60 + market_open.minute):
            minutes_today = (market_close.hour*60 + market_close.minute) - (market_open.hour*60 + market_open.minute)
        elif current_minutes_in_day >= (market_close.hour*60 + market_close.minute):
            minutes_today = 0
        else:
            minutes_today = (market_close.hour*60 + market_close.minute) - current_minutes_in_day

        full_trading_days = max(0, days_left - 1)
        minutes_per_full_day = (market_close.hour*60 + market_close.minute) - (market_open.hour*60 + market_open.minute)
        total_minutes = minutes_today + (full_trading_days * minutes_per_full_day)

    trading_minutes_per_year = (15*60 + 30 - 9*60 - 15) * 252
    T = max(0.0, total_minutes / trading_minutes_per_year)
    return T




def create_option_chain_data_with_delta(row, DELTA):
    try:
        En_Date = pd.to_datetime(row['En_Date'])
        atm_strike = row['Atm_Strike']
        expiry = row['Expiry']
        underlying_price = row['Spot_En_close']
        # manual_time_str = row['Manual_Time']

        T = calculate_time_to_expiry(En_Date, expiry)
        print("T", T)
        strikes = [atm_strike + i * roundoff for i in range(-10, 11)]
        delta_CE = []
        delta_PE = []
        CE_ltp_data = []
        PE_ltp_data = []
        strike_type = []

        for strike in strikes:
            strike_type.append(
                'Call OTM' if strike >= atm_strike else 'Put OTM'
            )
            # print(type(En_Date))
            # print(type(option_data))
            df_ce = deepcopy(option_data.loc[En_Date])
            # print(df_ce)
            df_ce = df_ce[(df_ce["StrikePrice"] == strike) & (df_ce["ExpiryDate"] == expiry) & (df_ce["Type"] == "CE")]
            CE_ltp = df_ce['Open'].values[0] if not df_ce.empty else 0
            CE_ltp_data.append(CE_ltp)

            iv_ce = implied_volatility(CE_ltp, underlying_price, strike, T, risk_free_rate, "CE")
            greeks_ce = black_scholes_greeks(underlying_price, strike, T, risk_free_rate, iv_ce, "CE") if iv_ce else None
            delta_CE.append(greeks_ce["Delta"] if greeks_ce else 0)

            df_pe = deepcopy(option_data.loc[En_Date])
            df_pe = df_pe[(df_pe["StrikePrice"] == strike) & (df_pe["ExpiryDate"] == expiry) & (df_pe["Type"] == "PE")]
            PE_ltp = df_pe['Open'].values[0] if not df_pe.empty else 0
            PE_ltp_data.append(PE_ltp)

            iv_pe = implied_volatility(PE_ltp, underlying_price, strike, T, risk_free_rate, "PE")
            greeks_pe = black_scholes_greeks(underlying_price, strike, T, risk_free_rate, iv_pe, "PE") if iv_pe else None
            delta_PE.append(greeks_pe["Delta"] if greeks_pe else 0)

        df_chain = pd.DataFrame({
            'Strike': strikes,
            'Call_LTP': CE_ltp_data,
            'Call_Delta': delta_CE,
            'Type': strike_type,
            'Put_LTP': PE_ltp_data,
            'Put_Delta': delta_PE,
        }).set_index('Strike')

        # Select the closest delta to 0.35 for CEs and -0.35 for PEs
        selected_CE_strike = df_chain.iloc[(df_chain['Call_Delta'] - DELTA).abs().argmin()].name
        selected_PE_strike = df_chain.iloc[(df_chain['Put_Delta'] + DELTA).abs().argmin()].name

        # Print or return the selected strikes
        print(f"Selected Call OTM Strike: {selected_CE_strike}")
        print(f"Selected Put OTM Strike: {selected_PE_strike}")

        df_chain['Selected_Put_OTM_Strike'] = selected_PE_strike
        df_chain['Selected_Call_OTM_Strike'] = selected_CE_strike

        selected_CE_delta = df_chain.loc[selected_CE_strike, 'Call_Delta']
        selected_PE_delta = df_chain.loc[selected_PE_strike, 'Put_Delta']

        df_chain['Selected_Put_Delta'] = selected_PE_delta
        df_chain['Selected_Call_Delta'] = selected_CE_delta

        # Return the result
        print(f"Data collected for row {row.name}")
        return {row.name: df_chain}
    except Exception as e:
        print("Error occurred while collecting data for row", row.name, ":", e)
        return {}
    


def get_option_chain_data(df_ts , option_data, DELTA):
    start = time.perf_counter()
    result_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        df_ts_list = [row for _, row in df_ts.iterrows()]
        results = executor.map(lambda row: create_option_chain_data_with_delta(row, DELTA), df_ts_list)
    for result in results:
        result_dict.update(result)
        print("Intermediate result collected",result)
    finish = time.perf_counter()
    print(f"Data collection completed in {np.round(finish - start, 2)} seconds")
    return result_dict




def new_df(df_ts):
    # Make a copy of the original DataFrame
    df_copy = df_ts.copy()

    # Fill missing values in 'Pe_Short_Atm_Strike' and 'Pe_Short_Atm_Strike' with 'Atm_Strike'
    df_copy['Pe_Short_Atm_Strike'].fillna(df_copy['Atm_Strike'], inplace=True)
    df_copy['Pe_Short_Atm_Strike'].fillna(df_copy['Atm_Strike'], inplace=True)

    # Fill any remaining NaN values in the DataFrame with 0
    df_copy.fillna(0, inplace=True)

    # # Convert `Pe_En_Date` and `Pe_En_Date` to datetime for consistency
    # df_copy['Pe_En_Date'] = pd.to_datetime(df_copy['Pe_En_Date'], errors='coerPe')
    # df_copy['Pe_En_Date'] = pd.to_datetime(df_copy['Pe_En_Date'], errors='coerPe')

    # Drop rows where both 'Pe_En_Date' and 'Pe_En_Date' are 0 (not NaN)
    df_copy = df_copy[~((df_copy['Pe_En_Date'] == 0) & (df_copy['Pe_En_Date'] == 0))]

    return df_copy





def createtradesheet(data, data_fut ,TIME):
    col_list = [
        'Strategy', 'Date', 'En_Date','Ex_Date' ,'Max_Ex_Date', 'Expiry',
        'Pe_En_Date', 'Pe_Ex_Date', 'Spot_En_Price', 'Atm_Strike',
        'Pe_Short_Atm_Strike', 'Pe_Short_Atm_En_Price', 'Pe_Short_Atm_Ex_Price',
    ]
    # vix_df = pd.read_csv("/home/newberry4/jay_test/options_strangle/_INDIAVIX__202410251224.csv")

    df = pd.DataFrame(columns=col_list)
    
    # Ensure 'Date' is a datetime.date object
    df["Date"] = list(set(data.index.date))

    hours, minutes = map(int, TIME.split(":"))
    
    # Set En_Date with dynamic time from TIME parameter
    df['En_Date'] = deepcopy(pd.to_datetime(df["Date"]).map(
        lambda t: t.replace(hour=hours, minute=minutes, second=0)))
    
    # df['Ex_Date'] = deepcopy(pd.to_datetime(df["Date"]).map(lambda t: t.replace(hour=15, minute=10, second=0)))

    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Total Trading Days - ", len(df))
    df['Max_Ex_Date'] = deepcopy(df['En_Date'].map(lambda t: t.replace(hour=15, minute=20 , second=0)))

    # Ensure 'Date' is a datetime.date object in both dataframes
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    data_fut['Date'] = pd.to_datetime(data_fut['Date']).dt.date
    
    # Create a dictionary for mapping expiry dates
    date_to_expiry = data_fut.set_index('Date')['ExpiryDate'].to_dict()

    # date_to_expiry = data_fut.set_index('Date')['MonthlyExpiry'].to_dict()
    # Map expiry dates to tradesheet DataFrame``
    df['Expiry'] = df['Date'].map(date_to_expiry)

    # Debugging: Check for missing expiry dates
    missing_dates = df[df['Expiry'].isna()]
    if not missing_dates.empty:
        print("Dates with missing expiry:", missing_dates['Date'].unique())

    # Optional steps
    # df = assign_expiry(df, df_expiry)  # Assign Expiry
    df = assign_strike(df,data, data_fut)  # Assign Futures Entry Price and Strikes

    if df is None:
        print("Error: assign_strike returned None. Unable to proceed.")
    else:
        df.reset_index(drop=True, inplace=True)
        df["Strategy"] = "DELTA_HEDGING"  # Update strategy name

    return df







def assign_strike(df_ts,data, data_fut):
    df = deepcopy(df_ts.copy())
    prev_spot_ex_close = None
    # Ensure 'DateTime' is the index in data_fut
    # Change 1: Set 'DateTime' as index
    # data_fut.index = pd.to_datetime(data_fut['DateTime'])
    
    for i, r in df.iterrows():
        try:
            En_Date = r['En_Date']
            # Ex_Date = r['Ex_Date']
            # print(En_Date, Ex_Date)
            Expiry = r['Expiry']
            
            # Check if 'DateTime' is an index
            if data_fut.index.name != 'DateTime':
                print(f"'DateTime' index is not correctly set at index {i}")
                continue
            
            # Filter daily spot data
            En_filtered_data = data_fut[data_fut.index == En_Date]
            print("processing for", En_Date)
            # print(En_filtered_data)
            # Ex_filtered_data = data_fut[data_fut.index == Ex_Date]
            # print(Ex_filtered_data)

            # Debug outPE
            Spot_En_open = En_filtered_data.iloc[0]['Open']
            print("open", Spot_En_open)
            Spot_En_close = En_filtered_data.iloc[0]['Close']

            if pd.isna(Spot_En_close):
                raise ValueError("Spot_En_close is NaN or missing!")
            print("close", Spot_En_close)
            # Spot_Ex_open = Ex_filtered_data.iloc[0]['Open']
            # Spot_Ex_close = Ex_filtered_data.iloc[0]['Close']

            # Process Pe Sell Signal
            # if prev_spot_ex_close is not None:
            #     Atm_Strike = np.round(prev_spot_ex_close, -2)
            # else:
            #     Atm_Strike = np.round(Spot_Ex_close, -2)
                
            # prev_spot_ex_close = Spot_Ex_close
            Spot_En_close = int(Spot_En_close)
            # Atm_Strike = np.round(Spot_En_close, -2)
            # print(Spot_En_close)
            Atm_Strike = nearest_multiple(Spot_En_close, roundoff)
            # print(Atm_Strike)

            # Validate that data is a DataFrame and not a list
            if not isinstance(data, pd.DataFrame):
                print(f"'data' is not a DataFrame at index {i}")
                continue

            df_options_chain = data.loc[r.En_Date]
            # print(df_options_chain)

            # Check if Expiry is valid
            if pd.isna(Expiry):
                print(f"Missing Expiry at index {i}")
                continue
            

            if isinstance(Expiry, str):
                Expiry = pd.to_datetime(Expiry)

            df_options_chain = df_options_chain.loc[df_options_chain["ExpiryDate"] == Expiry.strftime("%Y-%m-%d")]

            # Ensure options chain is not empty
            if df_options_chain.empty:
                print(f"Empty options chain at index {i}")
                continue

            df_CE_chain = df_options_chain.loc[(df_options_chain["StrikePrice"] == Atm_Strike) & (df_options_chain["Type"] == "CE")]
            # print(df_CE_chain)
            df_PE_chain = df_options_chain.loc[(df_options_chain["StrikePrice"] == Atm_Strike) & (df_options_chain["Type"] == "PE")]

            # Ensure CE and PE chains are not empty
            if df_CE_chain.empty or df_PE_chain.empty:
                print(f"Empty CE/PE chain at index {i}")
                continue

            Ce_En_Date = df_options_chain.iloc[0].name
            Pe_En_Date = df_options_chain.iloc[0].name

            Atm_Strike_Call = df_CE_chain.iloc[0].StrikePrice
            Atm_Strike_PE = df_PE_chain.iloc[0].StrikePrice

            df.at[i, "Spot_En_open"] = Spot_En_open
            df.at[i, "Spot_En_close"] = Spot_En_close
            # df.at[i, "Spot_Ex_open"] = Spot_Ex_open
            # df.at[i, "Spot_Ex_close"] = Spot_Ex_close

            df.at[i, "Atm_Strike"] = Atm_Strike
            df.at[i, "Ce_En_Date"] = Ce_En_Date
            df.at[i, "Pe_En_Date"] = Pe_En_Date
            df.at[i, "Ce_Short_Atm_Strike"] = Atm_Strike_Call
            df.at[i, "Pe_Short_Atm_Strike"] = Atm_Strike_PE
            
        except Exception as e:
            print(f"Error at index {i}: {e}")
    # df.dropna(subset=["Spot_En_Price"], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    return df




def list_func(df_ts):
    df_list = []
    for i in range(len(df_ts)):
        df = df_ts.iloc[i]
        df_list.apPend(df)
    return df_list


def data_collector_opt(row, data, df):
    try:
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a DataFrame, not a list or other type.")
        
        # Ensure ExpiryDate is a datetime column
        data['ExpiryDate'] = pd.to_datetime(data['ExpiryDate'])
        
        # Ensure indices are DateTimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Convert row date columns to datetime
        En_Date = pd.to_datetime(row['En_Date'])
        Max_Ex_Date = pd.to_datetime(row['Max_Ex_Date'])
        Expiry = pd.to_datetime(row['Expiry'])
        Pe_Short_Atm_Strike = row['Pe_Short_Atm_Strike']
        Ce_Short_Atm_Strike = row['Ce_Short_Atm_Strike']

        # Filter options data for the date range and specific strike and expiry
        df_trade = data.loc[En_Date:Max_Ex_Date]
        df_trade_ce = df_trade[
            (df_trade["StrikePrice"] == Ce_Short_Atm_Strike) & 
            (df_trade["ExpiryDate"] == Expiry)
        ]


        df_trade_pe = df_trade[
            (df_trade["StrikePrice"] == Pe_Short_Atm_Strike) & 
            (df_trade["ExpiryDate"] == Expiry) 
        ]

        # Process CE (Call Option) data
        df_ce_open = df_trade_ce[df_trade_ce["Type"] == "CE"][['Open']].rename(columns={'Open': 'Open_Ce'})
        df_ce_close = df_trade_ce[df_trade_ce["Type"] == "CE"][['Close']].rename(columns={'Close': 'Close_Ce'})
        df_ce = pd.concat([df_ce_open, df_ce_close], axis=1)
        df_ce = df_ce.loc[~df_ce.index.duplicated(keep='first')].sort_index()

        # print(df_ce)
        # Process PE (Put Option) data
        df_pe_open = df_trade_pe[df_trade_pe["Type"] == "PE"][['Open']].rename(columns={'Open': 'Open_Pe'})
        df_pe_close = df_trade_pe[df_trade_pe["Type"] == "PE"][['Close']].rename(columns={'Close': 'Close_Pe'})
        df_pe = pd.concat([df_pe_open, df_pe_close], axis=1)
        df_pe = df_pe.loc[~df_pe.index.duplicated(keep='first')].sort_index()
        

        # Combine CE and PE data
        df_trade_combined = deepcopy(pd.concat([df_ce, df_pe], axis=1))
        
        # Add index data (Open, High, Low, Close) for the same date range
        index_data = df.loc[En_Date:Max_Ex_Date][['Open', 'High', 'Low', 'Close']]
        
        # Merge index data with options data
        df_trade = pd.concat([df_trade_combined, index_data], axis=1)
        
        # Fill missing values using forward and backward fill
        df_trade.ffill(inplace=True)
        df_trade.bfill(inplace=True)
        df_trade.dropna(inplace=True)

        # Sort by index to ensure chronological order
        df_trade.sort_index(inplace=True)

        print(f"Processed df_trade for row {row.name}:")
        print(df_trade.head())

        # Return the combined DataFrame
        return {row.name: df_trade}
    except Exception as e:
        print(f"Error occurred while collecting data for row {row.name}: {e}")
        return {row.name: pd.DataFrame()}



def get_data_opt(df_ts, data,df):
    start = time.perf_counter()
    result_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Ensure df_ts is a DataFrame and properly iterate over rows
        if isinstance(df_ts, pd.DataFrame):
            results = executor.map(lambda row: data_collector_opt(row, data, df), [row for _, row in df_ts.iterrows()])
            for result in results:
                result_dict.update(result)
        else:
            raise TypeError("df_ts should be a DataFrame.")

    finish = time.perf_counter()
    print(f"Data collection completed in {np.round(finish - start, 2)} seconds")

    return result_dict




from copy import deepcopy                                                             
                                                                                



def calculate_position_size_df(df):
    """
    Calculate position size for each row in the DataFrame and add it as a new column.
    Calculate P&L based on position size and total returns.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the required columns for position sizing and total returns.

    Returns:
    pandas.DataFrame: New DataFrame with the calculated position size and P&L for each row.
    """
    # Create a copy of the inPE DataFrame to avoid modifying the original DataFrame
    new_df = df.copy()

    position_sizes = []
    pnl = []

    for index, row in new_df.iterrows():
        Pe_short_atm_en_priPe = row['Pe_Short_Atm_En_PriPe']
        total_returns = row['Total_Returns']
        
        # Check if the value is not NaN
        if pd.notna(Pe_short_atm_en_priPe):
            position_size = int(20000 / (Pe_short_atm_en_priPe * 15))
            position_size = min(position_size, 3)
            position_sizes.apPend(position_size)
            pnl.apPend(position_size * total_returns * 15)  # Calculate P&L
        else:
            position_sizes.apPend(0)
            pnl.apPend(0)  # ApPend 0 if Pe_short_atm_en_priPe is NaN

    new_df['Position Size'] = position_sizes
    new_df['pnl'] = pnl  # Add P&L column

    max_position_size = new_df['Position Size'].max()  # Find maximum position size
    print("Maximum Position Size:", max_position_size)
    
    return new_df




def day_on_expiry(df):
    # Convert 'Date' column to datetime format if it's not already
    df['En_Date'] = pd.to_datetime(df['En_Date'])
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    
    # Filter rows where 'En_Date' is equal to 'Expiry'
    result_df = df[df['En_Date'].dt.date == df['Expiry'].dt.date]
    
    return result_df



def export_to_csv(df_ts, file_path):
    try:
        df_ts.to_csv(file_path, index=False)
        print("DataFrame exported sucPessfully to", file_path)
    except Exception as e:
        print("Error:", e)



def calculate_equity_curve_max_drawdown_plot(df, returns_column='Total_Returns', date_column='Date'):
    # Set date column as index
    # df.set_index(date_column, inplace=True)

    # Calculate cumulative returns
    cumulative_returns = df[returns_column].cumsum()
    # Calculate drawdown
    cummax = cumulative_returns.cummax()
    drawdown = cumulative_returns - cummax

    # Calculate maximum drawdown
    max_drawdown = drawdown.min()

    # Plot equity curve and drawdown
    plt.figure(figsize=(12, 8))

    # Equity curve plot
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_returns.index, cumulative_returns, label='Equity Curve', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.legend()
    plt.grid(True)

    # Drawdown plot
    plt.subplot(2, 1, 2)
    plt.plot(drawdown.index, drawdown, label='Drawdown', color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Drawdown')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return max_drawdown


def tradesheet_report(df):
    capital = 500000
    # Convert date columns to datetime
    date_columns = ['Date', 'En_Date', 'Pe_En_Date', 'Pe_En_Date', 'Pe_Ex_Date', 'Pe_Ex_Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    
    # Determine win ratio
    win_ratio = (df['pnl'] > 0).mean()
    
    # Calculate maximum losing streak
    max_lose_streak = df['pnl'].lt(0).astyPe(int).groupby(df['pnl'].ge(0).cumsum()).cumsum().max()
    
    # Calculate maximum winning streak
    max_win_streak = df['pnl'].gt(0).astyPe(int).groupby(df['pnl'].le(0).cumsum()).cumsum().max()
    
    # Calculate total number of trades
    total_trades = len(df)
    
    # Calculate loss ratio
    loss_ratio = (df['pnl'] < 0).mean()
    
    # Calculate max profit in a single trade
    max_profit_single_trade = df['pnl'].max()
    
    # Calculate max loss in a single trade
    max_loss_single_trade = df['pnl'].min()
    
    
    # Calculate exPectancy ratio
    exPectancy_ratio = (win_ratio * max_profit_single_trade) / (loss_ratio * abs(max_loss_single_trade))
    
    # Calculate average profit Per trade on profitable days
    avg_profit_Per_profitable_trade = df[df['pnl'] > 0]['pnl'].mean()
    
    # Calculate average loss Per trade on losing days
    avg_loss_Per_loosing_trade = df[df['pnl'] < 0]['pnl'].mean()
    
    # Calculate average P&L Per trade
    avg_pnl_Per_trade = df['pnl'].mean()
    
    # Calculate total points gained
    total_points_gained = df['Total_Returns'].sum()

    
    
    df['Daily_Return'] = df['pnl']
    
    # Calculate daily returns PerPentage on capital
    df['Daily_Return_PerPentage'] = df['Daily_Return'] / capital * 100
    
    # Calculate cumulative returns in absolute value
    df['Cumulative_Return_Absolute'] = df['Daily_Return'].cumsum() + capital 
    
    # Calculate cumulative returns in PerPentage
    df['Cumulative_Return_PerPentage'] =df['Daily_Return_PerPentage'].cumsum()
    
    # Calculate daily drawdown
    df['Daily_Drawdown'] = -1 * (((df['Cumulative_Return_Absolute'].cummax() - df['Cumulative_Return_Absolute']) / capital) * 100)
    
    df['Daily_Drawdown'] = df['Daily_Drawdown'].replaPe(-0, 0)

    df['Total_Trades'] = total_trades
    
    df['Profitable Trades '] = len(df[df['pnl'] > 0])
    
    df['loosing trades'] = len(df[df['pnl'] < 0])
    
    df['Avg_profit_Per_trade'] = avg_profit_Per_profitable_trade
    
    df['Avg_loss_Per_trade'] = avg_loss_Per_loosing_trade
    
    df['Average_PnL_Per_Trade'] = avg_pnl_Per_trade
    
    df['Total_Points_Gained'] = total_points_gained
    
    df['Win_Ratio'] = win_ratio
    
    max_daily_drawdown = 0
    for index, drawdown in enumerate(df['Daily_Drawdown']):
        if drawdown < max_daily_drawdown:
            max_daily_drawdown = drawdown
        df.at[index, 'Max_Daily_Drawdown'] = max_daily_drawdown
    
    # Initialize recovery time column with zeros
    df['recovery_time'] = 0
    
    # Find consecutive negative values and calculate recovery time
    current_drawdown_Period = 0
    
    for index, drawdown in enumerate(df['Daily_Drawdown']):
        if drawdown < 0:
            current_drawdown_Period += 1
        else:
            if current_drawdown_Period > 0:
                df.loc[index - current_drawdown_Period:index, 'recovery_time'] = current_drawdown_Period
            current_drawdown_Period = 0
    
    # If the last value in the drawdown column is negative, set recovery time for remaining Periods
    if current_drawdown_Period > 0:
        df.loc[len(df) - current_drawdown_Period:, 'recovery_time'] = current_drawdown_Period
    
    
    
    # ReplaPe negative values in recovery time with zeros
    df['recovery_time'] = np.where(df['recovery_time'] < 0, 0, df['recovery_time'])
    
    negative_drawdown_count = sum(drawdown < 0 for drawdown in df['Daily_Drawdown'])
    
    # Return calculated metrics
    return {
        'Win_Ratio': win_ratio,
        'Max_Winning_Streak': max_win_streak,
        'Max_Losing_Streak' : max_lose_streak,
        'Total_Trades': total_trades,
        'Loss_Ratio': loss_ratio,
        'Max_Profit_Single_Trade': max_profit_single_trade,
        'Max_Loss_Single_Trade': max_loss_single_trade,
        'Max_Trades_In_Drawdown': negative_drawdown_count,
        'ExPectancy_Ratio': exPectancy_ratio,
        'avg_profit_Per_profitable_trade': avg_profit_Per_profitable_trade,
        'avg_loss_Per_loosing_trade': avg_loss_Per_loosing_trade,
        'Average_PnL_Per_Trade': avg_pnl_Per_trade,
        'Total_Points_Gained': total_points_gained,
        'Max_Drawdown_PerPentage': df['Max_Daily_Drawdown'].min(),
        'Cumulative_Return_Absolute': int((df['Cumulative_Return_Absolute'].iloc[-1])-capital),
        'Cumulative_Return_PerPentage': df['Cumulative_Return_PerPentage'].iloc[-1]
        
    }


def plot_cumulative_return_and_drawdown(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Cumulative Return PerPentage
    ax1.plot(df['Date'], df['Cumulative_Return_PerPentage'], label='Cumulative Return PerPentage', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PerPentage')
    ax1.set_title('Cumulative Return PerPentage')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Daily Drawdown
    ax2.plot(df['Date'], df['Daily_Drawdown'], label='Daily Drawdown', color='red', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('PerPentage')
    ax2.set_title('Daily Drawdown')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()



def remove_zero_total_returns(df):
    df_copy = df.copy()
    df_copy = df_copy[df_copy["Total_Returns"] != 0].reset_index(drop = True)
    return df_copy


def get_new_df(df_ts, result_option_chain_dict):
    # Create a deep copy of df_ts to modify
    df_ts_copy = deepcopy(df_ts)
    rows_to_drop = []

    # Update df_ts_copy with selected strikes from the option chain data
    for idx, row in df_ts_copy.iterrows():
        option_chain = result_option_chain_dict.get(row.name, None)
        if option_chain is not None:
            selected_CE_otm_strike = option_chain['Selected_Call_OTM_Strike'].iloc[0]
            selected_PE_otm_strike = option_chain['Selected_Put_OTM_Strike'].iloc[0]

            selected_CE_delta = option_chain['Selected_Call_Delta'].iloc[0]
            selected_PE_delta = option_chain['Selected_Put_Delta'].iloc[0]

            if selected_CE_otm_strike is None or selected_PE_otm_strike is None:
                rows_to_drop.append(idx)
            else:
                df_ts_copy.at[idx, 'Ce_Short_Atm_Strike'] = selected_CE_otm_strike
                df_ts_copy.at[idx, 'Pe_Short_Atm_Strike'] = selected_PE_otm_strike

                df_ts_copy.at[idx, 'Ce_delta'] = selected_CE_delta
                df_ts_copy.at[idx, 'Pe_delta'] = selected_PE_delta
        else:
            rows_to_drop.append(idx)

    # Drop the rows with None values for selected strikes
    df_ts_copy.drop(rows_to_drop, inplace=True)

    return df_ts_copy




def nearest_multiple(x, n):
    print(x,n)
    remainder = x%n
    print(remainder)
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)






##########################################


import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from copy import deepcopy

# --- Helper Functions ---

# def black_scholes_price(S, K, T, r, sigma, option_type):
#     if sigma <= 0 or T <= 0:
#         return 0
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "CE":
#         return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     elif option_type == "PE":
#         return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# def implied_volatility(option_price, S, K, T, r, option_type):
#     try:
#         return brentq(lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_price, 0.01, 5.0)
#     except (ValueError, RuntimeError):
#         return None

def black_scholes_greeks_delta(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "CE" else -norm.cdf(-d1)
    return {"Delta": delta}



def find_strike_for_delta(option_data, time, expiry, target_delta, option_type, underlying_price, T, r):
    """Find the strike closest to the target delta within ±20 strikes around ATM (ATM rounded to nearest 50)."""
    closest_strike = None
    min_diff = float('inf')

    # Round ATM to nearest 50
    atm_strike = round(underlying_price / roundoff) * roundoff

    # Get all unique strikes and filter within ±20 strikes
    all_strikes = sorted(option_data['StrikePrice'].unique())
    filtered_strikes = [s for s in all_strikes if atm_strike - (roundoff * 20) <= s <= atm_strike + (roundoff * 20)]  # filter 1000 range in advance
    if atm_strike in filtered_strikes:
        atm_index = filtered_strikes.index(atm_strike)
        lower_bound = max(0, atm_index - 20)
        upper_bound = min(len(filtered_strikes), atm_index + 21)
        relevant_strikes = filtered_strikes[lower_bound:upper_bound]
    else:
        relevant_strikes = filtered_strikes  # fallback to all

    # print("ATM Strike:", atm_strike)
    # print("Filtered Strikes:", relevant_strikes)

    for strike in relevant_strikes:
        df_ce_pe = deepcopy(option_data.loc[time])
        df_ce_pe = df_ce_pe[
            (df_ce_pe["StrikePrice"] == strike) &
            (df_ce_pe["ExpiryDate"] == expiry) &
            (df_ce_pe["Type"] == option_type)
        ]

        option_price = df_ce_pe['Open'].values[0] if not df_ce_pe.empty else 0
        # print(f"Strike: {strike}, Option Price: {option_price}")

        iv = implied_volatility(option_price, underlying_price, strike, T, r, option_type)
        # print("IV:", iv)

        if iv is None:
            continue

        greeks = black_scholes_greeks_delta(underlying_price, strike, T, r, iv, option_type)
        # print("Greeks_delta:", greeks)

        if greeks:
            # print("target_delta:", target_delta)
            # print((greeks["Delta"]))
            diff = abs(greeks["Delta"] - target_delta)
            # print(abs(greeks["Delta"]))
            if diff < min_diff:
                # print((greeks["Delta"]))
                min_diff = diff
                # print("Min_diff:", min_diff)
                # print("Closest Strike:", strike)
                closest_strike = strike

    return closest_strike

# --- Main Function ---



# def track_trades_opt(df_ts, result_dict_opt, option_data):
#     df = deepcopy(df_ts.copy())
#     final_df = []

#     risk_free_rate = 0.06
#     days_in_year = 365

#     for r in df.itertuples():
#         Date = str(r.Date)
#         daily_option_data = option_data[option_data['Date'] == Date].sort_index()

#         ce_strike = r.Ce_Short_Atm_Strike
#         pe_strike = r.Pe_Short_Atm_Strike
#         expiry = r.Expiry

#         Ce_delta = r.Ce_delta
#         Pe_delta = r.Pe_delta

#         try:
#             df_trade = deepcopy(result_dict_opt.get(r.Index))
#             if df_trade is None or df_trade.empty:
#                 continue

#             df_trade = df_trade[r.En_Date:r.Max_Ex_Date]
#             df_trade.sort_index(inplace=True)

#             start_price_ce = df_trade.iloc[0].Close_Ce
#             start_price_pe = df_trade.iloc[0].Close_Pe

#             # First row - Normal Trade
#             base_row = df.loc[[r.Index]].copy()
#             base_row.at[r.Index, "Ce_En_Date"] = r.En_Date
#             base_row.at[r.Index, "Pe_En_Date"] = r.En_Date
#             base_row.at[r.Index, "Ce_Short_Atm_Strike"] = ce_strike
#             base_row.at[r.Index, "Pe_Short_Atm_Strike"] = pe_strike
#             base_row.at[r.Index, "Ce_Short_Atm_En_Price"] = start_price_ce
#             base_row.at[r.Index, "Pe_Short_Atm_En_Price"] = start_price_pe
#             base_row.at[r.Index, "Ce_Action"] = "Normal Trade"
#             base_row.at[r.Index, "Pe_Action"] = "Normal Trade"
#             base_row.at[r.Index, "Ce_delta"] = Ce_delta
#             base_row.at[r.Index, "Pe_delta"] = Pe_delta

#             CEFlag = "Open"
#             PEFlag = "Open"

#             for rdft in df_trade.itertuples():
#                 underlying_price = rdft.Close

#                 if CEFlag == "Open" and rdft.Close >= ce_strike:
#                     T = calculate_time_to_expiry(rdft.Index, expiry)
#                     base_row.at[r.Index, "Ce_Ex_Date"] = rdft.Index
#                     base_row.at[r.Index, "Ce_Short_Atm_Ex_Price"] = rdft.Close_Ce
#                     base_row.at[r.Index, "SL_CE"] = "Call side SL hit"
#                     CEFlag = "Close"

#                     # Compute Greeks and Adjust CE
#                     iv_ce = implied_volatility(rdft.Close_Ce, underlying_price, ce_strike, T, risk_free_rate, "CE")
#                     greeks_ce = black_scholes_greeks(underlying_price, ce_strike, T, risk_free_rate, iv_ce, "CE") if iv_ce else {"Delta": 0}
#                     iv_pe = implied_volatility(rdft.Close_Pe, underlying_price, pe_strike, T, risk_free_rate, "PE")
#                     greeks_pe = black_scholes_greeks(underlying_price, pe_strike, T, risk_free_rate, iv_pe, "PE") if iv_pe else {"Delta": 0}

#                     net_delta = greeks_ce["Delta"] + greeks_pe["Delta"]

#                     if abs(net_delta) > 0.05:
#                         adj_strike = find_strike_for_delta(daily_option_data, rdft.Index, expiry, -net_delta, "PE", underlying_price, T, risk_free_rate)
#                         if adj_strike:
#                             df_adj = option_data.loc[rdft.Index:r.Max_Ex_Date]
#                             df_adj = df_adj[(df_adj["StrikePrice"] == adj_strike) &
#                                             (df_adj["ExpiryDate"] == expiry) &
#                                             (df_adj["Type"] == "PE")][["Open"]].sort_index()

#                             if not df_adj.empty:
#                                 entry_price = df_adj.iloc[0]["Open"]
#                                 exit_price = df_adj.iloc[-1]["Open"]
#                                 pe_exit_date = df_adj.index[-1]

#                                 adj_row = df.loc[[r.Index]].copy()
#                                 adj_row.at[r.Index, "Pe_En_Date"] = rdft.Index
#                                 adj_row.at[r.Index, "Pe_Short_Atm_Strike"] = adj_strike
#                                 adj_row.at[r.Index, "Pe_Short_Atm_En_Price"] = entry_price
#                                 adj_row.at[r.Index, "Pe_Ex_Date"] = pe_exit_date
#                                 adj_row.at[r.Index, "Pe_Short_Atm_Ex_Price"] = exit_price
#                                 adj_row.at[r.Index, "Pe_Action"] = "Adjustment Trade"
#                                 adj_row.at[r.Index, "Ce_Action"] = "-"
#                                 adj_row.at[r.Index, "Ce_delta"] = greeks_ce["Delta"]
#                                 adj_row.at[r.Index, "Pe_delta"] = greeks_pe["Delta"]

#                                 final_df.append(adj_row)

#                 if PEFlag == "Open" and rdft.Close <= pe_strike:
#                     T = calculate_time_to_expiry(rdft.Index, expiry)
#                     base_row.at[r.Index, "Pe_Ex_Date"] = rdft.Index
#                     base_row.at[r.Index, "Pe_Short_Atm_Ex_Price"] = rdft.Close_Pe
#                     base_row.at[r.Index, "SL_PE"] = "Put side SL hit"
#                     PEFlag = "Close"

#                     # Compute Greeks and Adjust PE
#                     iv_ce = implied_volatility(rdft.Close_Ce, underlying_price, ce_strike, T, risk_free_rate, "CE")
#                     greeks_ce = black_scholes_greeks(underlying_price, ce_strike, T, risk_free_rate, iv_ce, "CE") if iv_ce else {"Delta": 0}
#                     iv_pe = implied_volatility(rdft.Close_Pe, underlying_price, pe_strike, T, risk_free_rate, "PE")
#                     greeks_pe = black_scholes_greeks(underlying_price, pe_strike, T, risk_free_rate, iv_pe, "PE") if iv_pe else {"Delta": 0}

#                     net_delta = greeks_ce["Delta"] + greeks_pe["Delta"]

#                     if abs(net_delta) > 0.05:
#                         adj_strike = find_strike_for_delta(daily_option_data, rdft.Index, expiry, -net_delta, "CE", underlying_price, T, risk_free_rate)
#                         if adj_strike:
#                             df_adj = option_data.loc[rdft.Index:r.Max_Ex_Date]
#                             df_adj = df_adj[(df_adj["StrikePrice"] == adj_strike) &
#                                             (df_adj["ExpiryDate"] == expiry) &
#                                             (df_adj["Type"] == "CE")][["Open"]].sort_index()

#                             if not df_adj.empty:
#                                 entry_price = df_adj.iloc[0]["Open"]
#                                 exit_price = df_adj.iloc[-1]["Open"]
#                                 ce_exit_date = df_adj.index[-1]
#                                 adj_row = df.loc[[r.Index]].copy()
#                                 adj_row.at[r.Index, "Ce_En_Date"] = rdft.Index
#                                 adj_row.at[r.Index, "Ce_Short_Atm_Strike"] = adj_strike
#                                 adj_row.at[r.Index, "Ce_Short_Atm_En_Price"] = entry_price
#                                 adj_row.at[r.Index, "Ce_Ex_Date"] = ce_exit_date
#                                 adj_row.at[r.Index, "Ce_Short_Atm_Ex_Price"] = exit_price
#                                 adj_row.at[r.Index, "Ce_Action"] = "Adjustment Trade"
#                                 adj_row.at[r.Index, "Pe_Action"] = "-"
#                                 adj_row.at[r.Index, "Ce_delta"] = greeks_ce["Delta"]
#                                 adj_row.at[r.Index, "Pe_delta"] = greeks_pe["Delta"]
#                                 final_df.append(adj_row)

#                 if CEFlag == "Close" and PEFlag == "Close":
#                     break

#             # Exit remaining open legs at last point
#             final_row = df_trade.iloc[-1]
#             if CEFlag == "Open":
#                 base_row.at[r.Index, "Ce_Ex_Date"] = final_row.name
#                 base_row.at[r.Index, "Ce_Short_Atm_Ex_Price"] = final_row.Close_Ce
#                 base_row.at[r.Index, "SL_CE"] = "Exited at max date"
#             if PEFlag == "Open":
#                 base_row.at[r.Index, "Pe_Ex_Date"] = final_row.name
#                 base_row.at[r.Index, "Pe_Short_Atm_Ex_Price"] = final_row.Close_Pe
#                 base_row.at[r.Index, "SL_PE"] = "Exited at max date"

#             final_df.append(base_row)

#         except Exception as e:
#             print("Error:", e)

#     return pd.concat(final_df).reset_index(drop=True)



#####################################



from copy import deepcopy
import pandas as pd

# from copy import deepcopy
# import pandas as pd


# def track_trades_opt(df_ts, result_dict_opt, option_data):
#     df = deepcopy(df_ts.copy())
#     final_df = []

#     risk_free_rate = 0.06
#     days_in_year = 365

#     for r in df.itertuples():
#         try:
#             Date = str(r.Date)
#             print("Date:", Date)
#             daily_option_data = option_data[option_data['Date'] == Date].sort_index()

#             ce_strike = r.Ce_Short_Atm_Strike
#             pe_strike = r.Pe_Short_Atm_Strike
#             expiry = r.Expiry

#             Ce_delta = r.Ce_delta
#             Pe_delta = r.Pe_delta

#             df_trade = deepcopy(result_dict_opt.get(r.Index))
#             if df_trade is None or df_trade.empty:
#                 continue

#             df_trade = df_trade[r.En_Date:r.Max_Ex_Date].sort_index()
#             start_price_ce = df_trade.iloc[0].Open_Ce
#             start_price_pe = df_trade.iloc[0].Open_Pe

#             base_row = df.loc[[r.Index]].copy()
#             base_row.at[r.Index, "Ce_En_Date"] = r.En_Date
#             base_row.at[r.Index, "Pe_En_Date"] = r.En_Date
#             base_row.at[r.Index, "Ce_Short_Atm_Strike"] = ce_strike
#             base_row.at[r.Index, "Pe_Short_Atm_Strike"] = pe_strike
#             base_row.at[r.Index, "Ce_Short_Atm_En_Price"] = start_price_ce
#             base_row.at[r.Index, "Pe_Short_Atm_En_Price"] = start_price_pe
#             base_row.at[r.Index, "Ce_Action"] = "Normal Trade"
#             base_row.at[r.Index, "Pe_Action"] = "Normal Trade"
#             base_row.at[r.Index, "Ce_delta"] = Ce_delta
#             base_row.at[r.Index, "Pe_delta"] = Pe_delta

#             final_df.append(base_row)

#             open_strikes = [
#                 ("CE", ce_strike, Ce_delta, start_price_ce),
#                 ("PE", pe_strike, Pe_delta, start_price_pe)
#             ]

#             adjustment_done = False
#             ce_breakeven = ce_strike + start_price_ce
#             pe_breakeven = pe_strike - start_price_pe

#             for rdft in df_trade.itertuples():
#                 underlying_price = rdft.Open

#                 if not adjustment_done:
#                     if underlying_price >= ce_strike or underlying_price <= pe_strike:
#                         ce_total_delta = 0
#                         pe_total_delta = 0

#                         # updated_open_strikes = []

#                         for t, strike, _, _ in open_strikes:
#                             T = calculate_time_to_expiry(rdft.Index, expiry)
#                             if t == "CE":
#                                 entry_price = rdft.Open_Ce
#                             else:
#                                 entry_price = rdft.Open_Pe

#                             iv = implied_volatility(entry_price, underlying_price, strike, T, risk_free_rate, t)
#                             greeks = black_scholes_greeks(underlying_price, strike, T, risk_free_rate, iv, t) if iv else {"Delta": 0}
#                             delta = greeks.get("Delta", 0)

#                             if t == "CE":
#                                 ce_total_delta += delta
#                             else:
#                                 pe_total_delta += delta

#                             # updated_open_strikes.append((t, strike, delta, entry_price))

#                         net_delta = ce_total_delta + pe_total_delta
#                         side_to_add = "CE" if net_delta < 0 else "PE"
#                         delta_needed = abs(net_delta)

#                         T = calculate_time_to_expiry(rdft.Index, expiry)
#                         adj_strike = find_strike_for_delta(
#                             daily_option_data, rdft.Index, expiry, delta_needed,
#                             side_to_add, underlying_price, T, risk_free_rate
#                         )

#                         if adj_strike:
#                             df_adj = option_data.loc[rdft.Index:r.Max_Ex_Date]
#                             df_adj = df_adj[
#                                 (df_adj["StrikePrice"] == adj_strike) &
#                                 (df_adj["ExpiryDate"] == expiry) &
#                                 (df_adj["Type"] == side_to_add)
#                             ][["Open"]].sort_index()

#                             if not df_adj.empty:
#                                 entry_price = df_adj.iloc[0]["Open"]
#                                 exit_price = df_adj.iloc[-1]["Open"]
#                                 exit_date = df_adj.index[-1]

#                                 col_prefix = "Ce" if side_to_add == "CE" else "Pe"
#                                 adj_row = df.loc[[r.Index]].copy()
#                                 adj_row.at[r.Index, f"{col_prefix}_En_Date"] = rdft.Index
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_Strike"] = adj_strike
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_En_Price"] = entry_price
#                                 adj_row.at[r.Index, f"{col_prefix}_Ex_Date"] = exit_date
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_Ex_Price"] = exit_price
#                                 adj_row.at[r.Index, f"{col_prefix}_Action"] = "Adjustment Trade"
#                                 adj_row.at[r.Index, f"{col_prefix}_delta"] = delta_needed

#                                 final_df.append(adj_row)

#                                 open_strikes.append((side_to_add, adj_strike, delta_needed, entry_price))

#                                 # Update open_strikes and breakevens
#                                 # open_strikes = updated_open_strikes
#                                 print("Open Strikes:", open_strikes)

#                                 # ce_breakeven = max([s + p for t, s, d, p in open_strikes if t == "CE"], default=ce_strike)
#                                 # pe_breakeven = min([s - p for t, s, d, p in open_strikes if t == "PE"], default=pe_strike)
                                
#                                 calls = sorted([item[1:] for item in open_strikes if item[0] == 'CE'], key=lambda x: x[0])
#                                 puts = sorted([item[1:] for item in open_strikes if item[0] == 'PE'], key=lambda x: x[0])

#                                 ce_breakeven = None
#                                 pe_breakeven = None

#                                 if len(calls) == 2 and len(puts) == 1:
#                                     print("Condition: Selling 2 Call Options and 1 Put Option")
#                                     k1, _, c1 = calls[0]
#                                     k2, _, c2 = calls[1]
#                                     kp, _, p = puts[0]
#                                     ce_breakeven = (k1 + k2 + c1 + c2 + p) / 2
#                                     pe_breakeven = kp - c1 - c2 - p
#                                 elif len(calls) == 1 and len(puts) == 2:
#                                     print("Condition: Selling 1 Call Option and 2 Put Options")
#                                     kc, _, c = calls[0]
#                                     kp1, _, p1 = puts[1]  # Higher strike after sorting
#                                     kp2, _, p2 = puts[0]  # Lower strike after sorting
#                                     ce_breakeven = kc + p1 + p2 + c
#                                     pe_breakeven = (kp1 + kp2 - c - p1 - p2) / 2
#                                 else:
#                                     print("Error: open_strikes should contain either (2 Calls and 1 Put) or (1 Call and 2 Puts).")
#                                     return

#                                 print("CE Breakeven:", ce_breakeven)
#                                 print("PE Breakeven:", pe_breakeven)
#                                 adjustment_done = True
#                 else:
#                     # After first adjustment
#                     ce_adj_range = float("inf")
#                     pe_adj_range = float("-inf")

#                     if underlying_price >= ce_strike:
#                         ce_adj_range = ce_strike + 0.5 * (ce_breakeven - ce_strike)
#                         pe_adj_range = ce_strike - 0.66 * (ce_strike - pe_breakeven)
#                     elif underlying_price <= pe_strike:
#                         pe_adj_range = pe_strike - 0.5 * (pe_strike - pe_breakeven)
#                         ce_adj_range = pe_strike + 0.66 * (ce_breakeven - pe_strike)

#                     if underlying_price >= ce_adj_range or underlying_price <= pe_adj_range:
#                         updated_open_strikes = []
#                         for t, strike, _, _ in open_strikes:
#                             if t == "CE":
#                                 entry_price = rdft.Open_Ce
#                             else:
#                                 entry_price = rdft.Open_Pe

#                             iv = implied_volatility(entry_price, underlying_price, strike, T, risk_free_rate, t)
#                             greeks = black_scholes_greeks(underlying_price, strike, T, risk_free_rate, iv, t) if iv else {"Delta": 0}
#                             delta = greeks.get("Delta", 0)

#                             updated_open_strikes.append((t, strike, delta, entry_price))

#                         # 2️⃣ Select max delta leg using recomputed deltas
#                         max_delta_leg = max(updated_open_strikes, key=lambda x: abs(x[2]))
#                         # max_delta_leg = max(open_strikes, key=lambda x: abs(x[2]))
#                         if abs(max_delta_leg[2]) > 0.7:
#                             updated_open_strikes.remove(max_delta_leg)
#                             leg, _, _, _ = max_delta_leg
#                             excess_side = leg
#                         else:
#                             ce_total_delta = sum(d for t, _, d, _ in updated_open_strikes if t == "CE")
#                             pe_total_delta = sum(d for t, _, d, _ in updated_open_strikes if t == "PE")
#                             net_delta = ce_total_delta - pe_total_delta
#                             excess_side = "CE" if net_delta > 0 else "PE"
#                             filtered = [(t, s, d, p) for t, s, d, p in updated_open_strikes if t == excess_side]
#                             to_exit = min(filtered, key=lambda x: abs(x[2]))
#                             updated_open_strikes.remove(to_exit)

#                 #         # Add new strike to balance delta
#                         ce_total_delta = sum(d for t, _, d, _ in updated_open_strikes if t == "CE")
#                         pe_total_delta = sum(d for t, _, d, _ in updated_open_strikes if t == "PE")
#                         net_delta = ce_total_delta - pe_total_delta
#                         side_to_add = "CE" if net_delta < 0 else "PE"
#                         delta_needed = abs(net_delta)

#                         T = calculate_time_to_expiry(rdft.Index, expiry)
#                         adj_strike = find_strike_for_delta(daily_option_data, rdft.Index, expiry, delta_needed,
#                                                            side_to_add, underlying_price, T, risk_free_rate)

#                         if adj_strike:
#                             df_adj = option_data.loc[rdft.Index:r.Max_Ex_Date]
#                             df_adj = df_adj[(df_adj["StrikePrice"] == adj_strike) &
#                                             (df_adj["ExpiryDate"] == expiry) &
#                                             (df_adj["Type"] == side_to_add)][["Open"]].sort_index()

#                             if not df_adj.empty:
#                                 entry_price = df_adj.iloc[0]["Open"]
#                                 exit_price = df_adj.iloc[-1]["Open"]
#                                 exit_date = df_adj.index[-1]

#                                 col_prefix = "Ce" if side_to_add == "CE" else "Pe"
#                                 adj_row = df.loc[[r.Index]].copy()
#                                 adj_row.at[r.Index, f"{col_prefix}_En_Date"] = rdft.Index
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_Strike"] = adj_strike
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_En_Price"] = entry_price
#                                 adj_row.at[r.Index, f"{col_prefix}_Ex_Date"] = exit_date
#                                 adj_row.at[r.Index, f"{col_prefix}_Short_Atm_Ex_Price"] = exit_price
#                                 adj_row.at[r.Index, f"{col_prefix}_Action"] = "Adjustment Trade"

#                                 iv = implied_volatility(entry_price, underlying_price, adj_strike, T, risk_free_rate, side_to_add)
#                                 greeks = black_scholes_greeks(underlying_price, adj_strike, T, risk_free_rate, iv, side_to_add) if iv else {"Delta": 0}
#                                 adj_row.at[r.Index, f"{col_prefix}_delta"] = greeks["Delta"]

#                                 final_df.append(adj_row)
#                                 open_strikes.append((side_to_add, adj_strike, greeks["Delta"], entry_price))

#             final_row = df_trade.iloc[-1]
#             for leg, strike, delta, entry_price in open_strikes:
#                 col_prefix = "Ce" if leg == "CE" else "Pe"
#                 base_row.at[r.Index, f"{col_prefix}_Ex_Date"] = final_row.name
#                 base_row.at[r.Index, f"{col_prefix}_Short_Atm_Ex_Price"] = final_row[f"Open_{col_prefix}"]
#                 base_row.at[r.Index, f"SL_{leg}"] = "Exited at max date"

#             final_df.append(base_row)

#         except Exception as e:
#             print(f"Error in row {r.Index}: {e}")

#     return pd.concat(final_df).reset_index(drop=True)

def track_trades_opt(df_ts, result_dict_opt, option_data,TIME_FRAME,RANGE):
    from copy import deepcopy
    import pandas as pd

    df = deepcopy(df_ts.copy())
    final_trades = []
    risk_free_rate = 0.06

    for r in df.itertuples():
        try:
            Date = str(r.Date)
            # print("yessssssssss")
            daily_option_data = option_data[option_data['Date'] == Date].sort_index()

            ce_strike = r.Ce_Short_Atm_Strike
            pe_strike = r.Pe_Short_Atm_Strike
            expiry = r.Expiry

            df_trade = deepcopy(result_dict_opt.get(r.Index))
            if df_trade is None or df_trade.empty:
                continue

            df_trade = df_trade[r.En_Date:r.Max_Ex_Date].sort_index()
            # 📢 Resample to 5 minutes
            df_trade = df_trade.resample(TIME_FRAME).agg({
                'Open_Ce': 'first',
                'Close_Ce': 'last',
                'Open_Pe': 'first',
                'Close_Pe': 'last',
                'Open': 'first',
                'Close': 'last',
                'High': 'max',
                'Low': 'min'
            }).dropna()

            df_trade = df_trade.sort_index()

            start_price_ce = df_trade.iloc[0].Open_Ce
            start_price_pe = df_trade.iloc[0].Open_Pe

            ce_delta = r.Ce_delta
            pe_delta = r.Pe_delta
            if pe_delta > 0:
                pe_delta = -pe_delta   # ✅ Force PE delta negative

            open_trades = [
                {"type": "CE", "strike": ce_strike, "delta": ce_delta, "entry_date": r.En_Date, "expiry_date": expiry, "entry_price": start_price_ce, "exit_date": None, "exit_price": None, "exit_reason": None},
                {"type": "PE", "strike": pe_strike, "delta": pe_delta, "entry_date": r.En_Date, "expiry_date": expiry, "entry_price": start_price_pe, "exit_date": None, "exit_price": None, "exit_reason": None}
            ]

            ce_breakeven = ce_strike + start_price_ce
            pe_breakeven = pe_strike - start_price_pe
            # adjustment_done = False
            adjustment_triggered = False  # ✅ Track if initial adjustment triggered

            for rdft in df_trade.itertuples():
                if rdft.Index.time() >= pd.to_datetime("15:00:00").time():
                    break

                underlying_price = rdft.Open

                if not adjustment_triggered:
                    if underlying_price >= ce_strike or underlying_price <= pe_strike:
                        # First adjustment based on CE/PE strike breach
                        open_trades, (ce_breakeven, pe_breakeven) = perform_adjustment(open_trades, daily_option_data, rdft, expiry, option_data, r, risk_free_rate, final_trades)
                        adjustment_triggered = True  # ✅ Start tracking range now
                        # Recalculate ranges immediately after adjustment
                        print("ce_breakeven",ce_breakeven)
                        print("pe_breakeven",pe_breakeven)
                        ce_adj_range, pe_adj_range = get_adjustment_ranges(open_trades, ce_breakeven, pe_breakeven, underlying_price,RANGE)
                        print("ce_adj_range",ce_adj_range)
                        print("pe_adj_range",pe_adj_range)
                else:
                    # After first adjustment, keep checking range every minute
                    if underlying_price >= ce_adj_range or underlying_price <= pe_adj_range:
                        # Adjust position again
                        open_trades, (ce_breakeven, pe_breakeven) = perform_adjustment(open_trades, daily_option_data, rdft, expiry, option_data, r, risk_free_rate, final_trades)
                        print("ce_breakeven",ce_breakeven)
                        print("pe_breakeven",pe_breakeven)
                        # After adjustment, immediately recalculate the adjustment range
                        ce_adj_range, pe_adj_range = get_adjustment_ranges(open_trades, ce_breakeven, pe_breakeven, underlying_price,RANGE)
                        print("ce_adj_range",ce_adj_range)
                        print("pe_adj_range",pe_adj_range)
            final_row = df_trade.iloc[-1]
            for trade in open_trades:
                if trade["exit_date"] is None:
                    try:
                        trade["exit_date"] = r.Max_Ex_Date  # or final time in df_trade
                        suffix = "Ce" if trade["type"] == "CE" else "Pe"
                        field_name = "Open"
                        
                        # Look up correct option row for that strike and expiry from option_data
                        df_exit = option_data[
                            (option_data["StrikePrice"] == trade["strike"]) &
                            (option_data["ExpiryDate"] == trade["expiry_date"]) &
                            (option_data["Type"] == trade["type"]) &
                            (option_data.index == trade["exit_date"])
                        ]

                        if not df_exit.empty:
                            trade["exit_price"] = df_exit.iloc[0][field_name]
                        else:
                            print(f"⚠️ Could not find exit price for {trade['type']} strike {trade['strike']} on {trade['exit_date']}")
                            trade["exit_price"] = 0

                        trade["exit_reason"] = "Final Expiry Exit"
                    except Exception as e:
                        print(f"Error while exiting {trade['type']} leg: {e}")
                        trade["exit_price"] = 0
                        trade["exit_reason"] = "Final Expiry Exit"

                    final_trades.append(deepcopy(trade))

        except Exception as e:
            print(f"Error in row {r.Index}: {e}")

    final_df = pd.DataFrame(final_trades)
    if not final_df.empty:
        final_df = final_df[['type', 'strike', 'delta', 'entry_date', 'expiry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason']]
    return final_df



def perform_adjustment(open_trades, daily_option_data, rdft, expiry, option_data, r, risk_free_rate, final_trades):
    T = calculate_time_to_expiry(rdft.Index, expiry)
    updated_trades = []

    print("Performing adjustment for trades:", open_trades)
    for trade in open_trades:
        # entry_price = rdft.Open_Ce if trade["type"] == "CE" else rdft.Open_Pe
        df_ce_pe = deepcopy(daily_option_data.loc[rdft.Index:r.Max_Ex_Date])
        df_ce_pe = df_ce_pe[
            (df_ce_pe["StrikePrice"] == trade["strike"]) &
            (df_ce_pe["ExpiryDate"] == expiry) &
            (df_ce_pe["Type"] == trade["type"] )
        ].sort_index()

        entry_price = df_ce_pe.iloc[0]["Open"] if not df_ce_pe.empty else 0

        print("entry_price",entry_price)
        print("rdft",rdft)
        print("spot",rdft.Open)
        iv = implied_volatility(entry_price, rdft.Open, trade["strike"], T, risk_free_rate, trade["type"])
        greeks = black_scholes_greeks(rdft.Open, trade["strike"], T, risk_free_rate, iv, trade["type"]) if iv else {"Delta": 0}
        trade["delta"] = greeks.get("Delta", 0)
        updated_trades.append(trade)

    print("Updated Trades:", updated_trades)
    net_delta = sum(t.get("delta", 0) for t in updated_trades)
    print("Net Delta:", net_delta)

    if abs(net_delta) < 0.05:
        print("Net Delta is within acceptable range, no adjustment needed.")
        return updated_trades, recalculate_breakevens(updated_trades)

    max_delta_trade = max(updated_trades, key=lambda x: abs(x.get("delta", 0)))

    if (max_delta_trade["type"] == "CE" and max_delta_trade["delta"] >= 0.7) or \
       (max_delta_trade["type"] == "PE" and max_delta_trade["delta"] <= -0.7):
        to_exit = max_delta_trade
        print("Exiting max delta leg:", to_exit)
        
        for trade in updated_trades:
            if trade == to_exit and trade["exit_date"] is None:
                trade["exit_date"] = rdft.Index
                print("Exiting max delta trade on date:", trade["exit_date"])
                trade["exit_reason"] = "Exit Max Delta Leg"
                # suffix = "Ce" if trade["type"] == "CE" else "Pe"
                # field_name = f"Open_{suffix}"
                # if hasattr(rdft, field_name):
                #     trade["exit_price"] = getattr(rdft, field_name)
                #     print("Exiting max delta trade on price:", trade["exit_price"])
                df_exit = option_data[
                    (option_data["StrikePrice"] == trade["strike"]) &
                    (option_data["ExpiryDate"] == trade["expiry_date"]) &
                    (option_data["Type"] == trade["type"]) &
                    (option_data.index == trade["exit_date"])
                ]
                if not df_exit.empty:
                    trade["exit_price"] = df_exit.iloc[0]["Open"]
                else:
                    print(f"⚠️ Could not find exit price for {trade['type']} strike {trade['strike']} on {trade['exit_date']}")
                    trade["exit_price"] = 0
                final_trades.append(deepcopy(trade))

        updated_trades = [t for t in updated_trades if t["exit_date"] is None]
        print("Trades after exiting max delta leg:", updated_trades)

    elif len(updated_trades) >= 3:
        print("More than 2 legs open, adjusting one leg...")

        ce_total = sum(t.get("delta", 0) for t in updated_trades if t["type"] == "CE")
        pe_total = sum(t.get("delta", 0) for t in updated_trades if t["type"] == "PE")

        # Decide side to remove from: the side with more total delta and at least 2 legs
        ce_trades = [t for t in updated_trades if t["type"] == "CE"]
        pe_trades = [t for t in updated_trades if t["type"] == "PE"]

        print("ce_trades",ce_trades)
        print("pe_trades",pe_trades)

        if abs(ce_total) > abs(pe_total) and len(ce_trades) >= 2:
            side_to_remove = "CE"
            # Remove the strike with **highest** delta (more impact)
            to_exit = max(ce_trades, key=lambda x: abs(x.get("delta", 0)))
            print(to_exit)
        elif abs(pe_total) > abs(ce_total) and len(pe_trades) >= 2:
            side_to_remove = "PE"
            to_exit = max(pe_trades, key=lambda x: abs(x.get("delta", 0)))
        else:
            # Fall back to existing logic: remove least delta leg from side with smaller delta
            side_to_remove = "CE" if abs(ce_total) < abs(pe_total) else "PE"
            to_exit = min((t for t in updated_trades if t["type"] == side_to_remove), key=lambda x: abs(x.get("delta", 0)))

        print(f"Exiting from side: {side_to_remove}, Strike: {to_exit['strike']}, Delta: {to_exit['delta']}")

        for trade in updated_trades:
            if trade == to_exit and trade["exit_date"] is None:
                trade["exit_date"] = rdft.Index
                trade["exit_reason"] = "Exit for Delta Adjustment"

                df_exit = option_data[
                    (option_data["StrikePrice"] == trade["strike"]) &
                    (option_data["ExpiryDate"] == trade["expiry_date"]) &
                    (option_data["Type"] == trade["type"]) &
                    (option_data.index == trade["exit_date"])
                ]

                if not df_exit.empty:
                    trade["exit_price"] = df_exit.iloc[0]["Open"]
                else:
                    print(f"⚠️ Could not find exit price for {trade['type']} strike {trade['strike']} on {trade['exit_date']}")
                    trade["exit_price"] = 0

                final_trades.append(deepcopy(trade))

        updated_trades = [t for t in updated_trades if t["exit_date"] is None]

    # ✅ Now after exit, check if we need to add a new strike
    net_delta = sum(t.get("delta", 0) for t in updated_trades)
    print("Net Delta after adjustment:", net_delta)

    if abs(net_delta) >= 0.05:
        side_to_add = "CE" if net_delta < 0 else "PE"
        delta_needed = abs(net_delta)
        print("Delta needed:", delta_needed)

        corrected_delta = delta_needed if side_to_add == "CE" else -delta_needed  # ✅ FIXED Delta Sign

        adj_strike = find_strike_for_delta(daily_option_data, rdft.Index, expiry, corrected_delta, side_to_add, rdft.Open, T, risk_free_rate)

        if adj_strike:
            df_adj = option_data.loc[rdft.Index:r.Max_Ex_Date]
            df_adj = df_adj[(df_adj["StrikePrice"] == adj_strike) & (df_adj["ExpiryDate"] == expiry) & (df_adj["Type"] == side_to_add)][["Open"]].sort_index()

            if not df_adj.empty:
                entry_price = df_adj.iloc[0]["Open"]
                new_trade = {
                    "type": side_to_add,
                    "strike": adj_strike,
                    "delta": corrected_delta,  # ✅ Correct delta
                    "entry_date": rdft.Index,
                    "expiry_date": expiry,
                    "entry_price": entry_price,
                    "exit_date": None,
                    "exit_price": None,
                    "exit_reason": None
                }
                updated_trades.append(new_trade)

    return updated_trades, recalculate_breakevens(updated_trades)







def get_adjustment_ranges(open_trades, ce_breakeven, pe_breakeven, underlying_price, RANGE):
    ce_strikes = [t["strike"] for t in open_trades if t["type"] == "CE"]
    pe_strikes = [t["strike"] for t in open_trades if t["type"] == "PE"]

    ce_strike = max(ce_strikes) if ce_strikes else 0
    pe_strike = min(pe_strikes) if pe_strikes else 0

    ce_adj_range = float('inf')
    pe_adj_range = float('-inf')

    print("RANGE",RANGE)
    print("ce_breakeven",ce_breakeven)
    print("pe_breakeven",pe_breakeven)
    print("underlying_price",underlying_price)
    print("ce_strike",ce_strike)
    print("pe_strike",pe_strike)
    print("open_trades",open_trades)
    # Ensure the RANGE has at least one value (1 pair)
    if RANGE:
        # Extract the single range pair, for example [1, 2]
        range_ce, range_pe = RANGE  # [1, 2] in this case

        if ce_strike and pe_strike:
            # if underlying_price >= ce_strike:
                    # Apply range_ce for CE and range_pe for PE
            ce_adj_range = underlying_price + (range_ce / 100) * (ce_breakeven - underlying_price)
            pe_adj_range = underlying_price - (range_pe / 100) * (underlying_price - pe_breakeven)
            # elif underlying_price <= pe_strike:
            #     pe_adj_range = underlying_price - (range_pe / 100) * (underlying_price - pe_breakeven)
            #     ce_adj_range = underlying_price + (range_ce / 100) * (ce_breakeven - underlying_price)

    return ce_adj_range, pe_adj_range




def recalculate_breakevens(open_trades):
    calls = [t for t in open_trades if t["type"] == "CE"]
    puts = [t for t in open_trades if t["type"] == "PE"]

    if len(calls) == 2 and len(puts) == 1:
        ce_breakeven = (calls[0]["strike"] + calls[1]["strike"] + calls[0]["entry_price"] + calls[1]["entry_price"] + puts[0]["entry_price"]) / 2
        pe_breakeven = puts[0]["strike"] - calls[0]["entry_price"] - calls[1]["entry_price"] - puts[0]["entry_price"]
    elif len(calls) == 1 and len(puts) == 2:
        ce_breakeven = calls[0]["strike"] + puts[0]["entry_price"] + puts[1]["entry_price"] + calls[0]["entry_price"]
        pe_breakeven = (puts[0]["strike"] + puts[1]["strike"] - calls[0]["entry_price"] - puts[0]["entry_price"] - puts[1]["entry_price"]) / 2
    else:
        # ce_breakeven = float("inf")
        # pe_breakeven = float("-inf")
        ce_breakeven = calls[0]["strike"] + calls[0]["entry_price"]
        pe_breakeven = puts[0]["strike"] - puts[0]["entry_price"]


    return ce_breakeven, pe_breakeven


def process_data_for_params(mapped_days ,option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, RANGE, DELTA,TIME):
    """
    This function combines all the steps into a single function to enable multiprocessing with different parameter combinations.

    Parameters:
    - stock: The stock symbol (e.g., 'NIFTY').
    - option_data: The options data required for the analysis.
    - df: The dataframe containing the main trading data.
    - outPE_folder_path: The output folder path to save the results.
    - multiplier: Multiplier value for NIFTY or other stocks (default 50 for NIFTY).
    - risk_free_rate: The risk-free rate used in the calculations (default 0.0696).
    """

    target_list =  [TIME_FRAME, RANGE, DELTA,TIME]


    # df = df

    df = df[(df['Date'] >= start_date) & 
                                    (df['Date'] <= end_date)]
    df = df[(df['Date'] > '2021-06-03') & 
                                    (df['Date'] != '2024-05-18') & (df['Date'] != '2024-05-20') & (df['DaysToExpiry'] == 0) ]
    df = df[
    ~((df['Date'] >= '2024-05-30') & (df['Date'] <= '2024-06-06'))
]
    df = df.sort_index()

    # Step 1: Create daily tradesheet
    df_ts = createtradesheet(option_data, df, TIME)

    # Use multiplier for NIFTY (or other stocks)
    if stock == 'NIFTY':
        multiplier = 50
    else:
        multiplier = 100

    # print("df_ts :", df_ts)

    # Step 2: Get option chain data
    result_option_chain_dict = get_option_chain_data(df_ts, option_data, DELTA)
    print("Option Chain Data Retrieved",result_option_chain_dict)
    # print("Option Chain Data:", result_option_chain_dict)
    # inner_df = list(result_option_chain_dict.values())[0]  # this gives you the actual DataFrame
    # Step 2: Move index into a column (optional: name it clearly)
    # inner_df = inner_df.reset_index() 

    # Step 3: Save it directly to CSV
    # inner_df.to_csv(f'{output_folder_path}result_dict_opt.csv', index=False)

    # print("Saved actual DataFrame with shape:", inner_df.shape)
    # Step 3: Create new DataFrame based on option chain data
    # exit()
    df_ts1 = get_new_df(df_ts, result_option_chain_dict)

    # Step 4: Get data for options trades from entry date to expiry
    result_dict_opt = get_data_opt(df_ts1, option_data, df)
    

    # Step 5: Track trades based on the created option data
    trades = track_trades_opt(df_ts1, result_dict_opt, option_data,TIME_FRAME,RANGE)

    if 'entry_price' not in trades.columns:
        print("entry_price column is missing")

    print("Trades DataFrame columns:", trades.columns)

    # Step 6: Calculate total returns for each trade
    trades['Total_Returns'] = trades['entry_price'] - trades['exit_price']

    # Optionally filter out rows with zero or NaN returns
    # df_ts_01 = df_ts_01[(df_ts_01['Total_Returns'] != 0) & df_ts_01['Total_Returns'].notna() & (df_ts_01['Total_Returns'] != '')]

    # Step 7: Save the result to CSV
    # dfs_path = os.path.join(outPE_folder_path, 'delta_hedging_apeksha.csv')
    # df_ts_01.to_csv(dfs_path, index=False)


    strategy_name = f'{stock}_candle_{TIME_FRAME}_range_{RANGE}_delta_{DELTA}_time_{TIME}'
    sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')


    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4','Day' ,'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
    row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
    filter_df1.loc[row_index, 'Parameters'] = target_list
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date



    if not trades.empty:
        trades.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)

    existing_csv_file = rf"{filter_df_path}/filter_df{counter}.csv"
    if os.path.isfile(existing_csv_file):
        filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
    else:
        filter_df1.to_csv(existing_csv_file, index=False)
        
    return sanitized_strategy_name + '_' + str(start_date) + '_' + str(end_date)


    # Return the dataframe in case you want to use it later in multiprocessing
    # return df_ts_01



def parameter_process(parameter, mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, RANGE, DELTA, TIME = parameter
    resampled_df = resample_data(df,TIME_FRAME)
    resampled = resampled_df.dropna() 
    return process_data_for_params(mapped_days ,option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, RANGE, DELTA, TIME)



#################################### InPEs ######################################################
superset = 'delta_hedging'
stock = 'SENSEX'
option_type = 'ND'



roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' else (100 if stock == 'SENSEX' else None))
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' else (3 if stock == 'FINNIFTY' else None))
LOT_SIZE = 75 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (20 if stock == 'SENSEX' else None))

# crossover = 'Upper' if Type == 'CE' else ('Lower' if Type == 'PE' else None)
root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
# Define all the file paths
if stock == 'NIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/monthly_expiry/"
    option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/Current_Expiry_OI_OHLC/"     #### pickle file
    # option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/current_expiry_paraquette/"                  ####paraquette file
elif stock =='BANKNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/BANKNIFTY_DATA/BANKNIFTY_OHLCV/"
    option_data_path = rf"/home/newberry4/jay_data/Data/BANKNIFTY/monthly_expiry_OI/"
elif stock =='FINNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/FINNIFTY_2/"
    option_data_path = rf"/home/newberry4/jay_data/Data/FINNIFTY/monthly_expiry/"
elif stock =='SENSEX':
    # option_data_path = rf"jay_data/Data/SENSEX/weekly_expiry/"
    # option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OHLC/"
    option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OI_OHLC2/"


if stock == 'NIFTY':        
    expiry_file_path = rf"/home/newberry4/jay_data/NIFTY Market Dates updated 2025.xlsx"
else:       
    expiry_file_path = rf"/home/newberry4/jay_data/SENSEX market dates updated 2025.xlsx"  # expiry_file_path = rf"/home/newberry4/jay_data/nifty_2nd_week_expiry.xlsx"       # for finnifty
# expiry_file_path = "/home/newberry4/jay_data/BANKNIFTY market dates (1).xlsx"   # for banknifty

txt_file_path = rf'{root_path}/new_done.txt'
filter_df_path = rf"{root_path}/Filter_Sheets/"
output_folder_path = rf'{root_path}/Trade_Sheets/'
# outPE_folder_path2 = rf'{root_path}/sheets/'


# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# list of period buckets
# date_ranges = [('2024-02-01', '2024-05-31')]


##trail

# date_ranges = [ 
# ('2024-04-01', '2024-08-31'),     ## trail
#                 ]



if stock == 'NIFTY':
    date_ranges = [('2025-05-01', '2025-08-31'),
        ('2025-02-01', '2025-04-30'),
                    ('2024-10-01', '2025-01-31'),
                ('2024-06-01', '2024-09-30'),
                    ('2024-02-01', '2024-05-31'),
                    ('2023-10-01', '2024-01-31'),
                    ('2023-06-01', '2023-09-30'),
                    ('2023-02-01', '2023-05-31'),
                    ('2022-10-01', '2023-01-31'),
                    ('2022-06-01', '2022-09-30'),
                    ('2022-01-01', '2022-05-31'),
                    ('2021-09-01', '2021-12-31'), 
                    ('2021-06-01', '2021-08-31')
                    ]


#sensex
if stock == 'SENSEX':
    date_ranges = [('2025-05-01', '2025-08-31'),
        ('2025-02-01', '2025-04-30'),
                    ('2024-10-01', '2025-01-31'),
                ('2024-06-01', '2024-09-30'),
                    ('2024-02-01', '2024-05-31'),
                    ('2023-10-01', '2024-01-31'),
                    ('2023-08-01', '2023-09-30')]
               



SIGNAL_PERIOD = 9
dte_list = [0, 1, 2, 3, 4]

delta = [0.3 , 0.35 , 0.4 , 0.45 , 0.5 ]
# delta = [0.3]
ranges = [[50,66] , [70,88] ,[50,75], [50,50] , [66,66]]
# ranges = [[50,50]]

candle_time_frame = ['5T']

Start_time = ['09:30','10:15', '11:45', '13:00']
# Start_time = ['09:30','10:15','13:00']
# Start_time = ['09:30']



parameters = []
risk_free_rate = 0.0696


#############################################################################################

if __name__ == "__main__":

    counter = 0

    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]
    
    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    #mapped_days = mapped_days[(mapped_days['Date'] >= index_start_date) & (mapped_days['Date'] <= index_end_date)]
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]
    mapped_days = mapped_days[
    ~((mapped_days['Date'] >= '2024-05-30') & (mapped_days['Date'] <= '2024-06-06') &
    (mapped_days['DaysToExpiry'] == 0))
]
    weekdays = mapped_days['Date'].to_list()
    
    # Pull Index data
    start_time = time.time()

    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # Print the current time
    print("Start Time : ", current_time)

    # print(start_date_idx, end_date_idx, superset, stock, Type)


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
    df = df[(df['DaysToExpiry'] == 0)] 
    # resampled_df_main = resample_data(df, '5T')

    for start_date, end_date in date_ranges: 
        counter += 1
        print(start_date, end_date, counter)
        # print(superset, stock, Type)

        # outPE_folder_path = f'/home/newberry/EMA Crossover copy/Trade Sheets/{stock}/{Type}/'
        
        # Pull Options data
        start_time = time.time()
        # option_data_path = rf"/home/newberry/Global Files/Options Data/{stock}/"
        option_data_files = next(os.walk(option_data_path))[2]
        option_data = pd.DataFrame()

        for file in option_data_files:
            file1 = compare_month_and_year(start_date, end_date, file, stock)
                
            if not file1:
                continue

            temp_data = pd.read_pickle(option_data_path + file)[['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Low', 'Close','OI','Volume', 'Ticker']  ]
            temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
            temp_data = temp_data.rename_axis('DateTime')
            option_data = pd.concat([option_data, temp_data])

        option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
        option_data['Type'] = option_data['Type'].astype('category')
        option_data = option_data[option_data['Date'].isin(df['Date'])]
        option_data = option_data.sort_index()
        end_time = time.time()
        print('Time taken to pull Options data :', (end_time-start_time))
        print(option_data)

        # exit()

        parameters = []

        if counter==1:
            filter_df = pd.DataFrame()
        elif counter>1:
            if not os.path.exists(f"{filter_df_path}/filter_df{counter-1}.csv"):
                print(f"File filter_df{counter-1}.csv does not exist. Stopping the code.")
                sys.exit()
            else:
                filter_df = pd.read_csv(f"{filter_df_path}/filter_df{counter-1}.csv")  
                filter_df = filter_df.drop_duplicates()   
                    
        if counter!=1:
            parameters = filter_df['Parameters'].to_list()
            parameters = [ast.literal_eval(item.replace("'", "\"")) for item in parameters]
        
        elif counter == 1:
            for TIME_FRAME in candle_time_frame:
                for RANGE in ranges:
                    for DELTA in delta:
                        for TIME in Start_time:
                        # TARGET = 100000 if TARGET == 'na' else TARGET  # Replace 'na' with 100000
                            parameters.append([TIME_FRAME, RANGE, DELTA, TIME])


        # Read the content of the log file to check which parameters have already been processed
        print('Total parameters :', len(parameters))
        file_path = txt_file_path
        with open(file_path, 'r') as file:      
            existing_values = [line.strip() for line in file]      

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if (stock + '_candle_' + str(value[0])  + '_range_' + str(value[1]).replace('.', ',')  + '_delta_' + str(value[2]).replace('.', ',') + '_time_' + str(value[3]).replace(':', ',') + '_'  + start_date + '_' + end_date) not in existing_values]
        print('Parameters to run :', len(parameters))
        for value in  parameters:
            string =  (stock + '_candle_' + str(value[0])  + '_range_' + str(value[1]).replace('.', ',')  + '_delta_' + str(value[2]).replace('.', ',') + '_time_' + str(value[3]).replace('.', ',') + '_'  + start_date + '_' + end_date)
            print(string)
   
        start_time1 = time.time()
        num_processes = 19
        print('No. of processes :', num_processes)

        partial_process = partial(parameter_process,   mapped_days=mapped_days, option_data=option_data, df=df, filter_df=filter_df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path)
        with multiprocessing.Pool(processes = num_processes) as pool:
            
            with tqdm(total = len(parameters), desc = 'Processing', unit = 'Iteration') as pbar:
                def update_progress(combinations):
                    with open(txt_file_path, 'a') as fp:
                        line = str(combinations) + '\n'
                        fp.write(line)
                    pbar.update()
                
                arg_tuples = [tuple(parameter) for parameter in parameters]
                
                for result in pool.imap_unordered(partial_process, arg_tuples):
                    update_progress(result)
        
        end_time1 = time.time()
        elapsed_time = end_time1 - start_time1
        print('Time taken to get Initial Tradesheets:', elapsed_time)
        print('Finished at :', time.time())
        
  
    # vix_df = pd.read_csv("/home/newberry4/jay_test/options_strangle/_INDIAVIX__202410251224.csv")
    # vix_range = [0,19]
    # df_ts = createtradesheet(option_data, df )
    # df_ts = createtradesheet(option_data, df)  # CREATE DAILY TRADESHEET
    # # df_ts.to_csv("/home/newberry4/jay_test/options_strangle/op.csv")

    # if stock == 'NIFTY':
    #     multiplier = 50
    # else:
    #     multiplier = 100

    # risk_free_rate = 0.0696

    # result_option_chain_dict =  get_option_chain_data(df_ts, option_data)
    
    # # df_ts1 = identify_gap1(df_ts)
    # # dfs_path = os.path.join(outPE_folder_path2, 'df_ts11.csv')  # Add file name
    # # df_ts1.to_csv(dfs_path) 

    # df_ts1 = get_new_df(df_ts ,result_option_chain_dict)

    # result_dict_opt = get_data_opt(df_ts1,option_data, df)  # Create Dict Of All Trades Data From Entry Date To Expiry

    # df_ts_01 = track_trades_opt(df_ts1, result_dict_opt, option_data)   

    # print(df_ts_01)                      
    # df_ts_01['Total_Returns'] = df_ts_01['entry_price'] - df_ts_01['exit_price']  
    # # df_ts_01 = df_ts_01[(df_ts_01['Total_Returns'] != 0) & df_ts_01['Total_Returns'].notna() & (df_ts_01['Total_Returns'] != '')]

    # dfs_path = os.path.join(outPE_folder_path, 'delta_hedging_apeksha.csv')    
    # df_ts_01.to_csv(dfs_path, index=False)      
















        # Ensure entry_date is in datetime format
    # import pandas as pd
    # import os
    # from datetime import datetime
    # df_ts_01 = pd.read_csv("/home/newberry4/jay_test/delta_hedging/NIFTY/ND/Trade_Sheets/delta_hedging_apeksha.csv")
    # df_ts_01['entry_date'] = pd.to_datetime(df_ts_01['entry_date'])

    # # Extract just the date part
    # df_ts_01['entry_day'] = df_ts_01['entry_date'].dt.date

    # df_ts_01 = df_ts_01.groupby('entry_day').agg({
    #     'Total_Returns': 'sum',
    #     'entry_day': 'first'}).sort_index(ascending=True)
    # # Group by date and sort by date (ascending)
    # df_ts_01 = df_ts_01.sort_values(by='entry_day')

    # # (Optional) If you want to group and view each day separately:
    # # grouped = df_ts_01.groupby('entry_day')

    # # Save final sorted DataFrame
    # dfs_path = os.path.join(outPE_folder_path, 'delta_hedging_apeksha_date_wise.csv')
    # df_ts_01.to_csv(dfs_path, index=False)

    # plot_cumulative_returns(df_ts_01, file_path='/home/newberry4/jay_test/EMA_Crossover/cumulative_returns.png')









    # df_ts2 = createtradesheet2(option_data, df) 

    # df_ts2 = identify_gap2(df_ts2)
    # dfs_path2 = os.path.join(outPE_folder_path2, 'df_ts22.csv')  # Add file name
    # df_ts2.to_csv(dfs_path2)

    # result_dict_opt2 = get_data_opt(df_ts2, option_data, df)

    # df_ts_02 = track_trades_opt2(df_ts2, result_dict_opt2)
    # # print(df_ts_02)
    # print(df_ts_02['Total_Returns'].cumsum())

    # df_ts_02 = df_ts_02[(df_ts_02['Total_Returns'] != 0) & df_ts_02['Total_Returns'].notna() & (df_ts_02['Total_Returns'] != '')]

    # dfs_path = os.path.join(outPE_folder_path, 'gap_up_gap_down_intraday_extra_selling_lot_1_1.csv') 
    # df_ts_02.to_csv(dfs_path, index=False) 




















    # multiplier1 = 3
    # multiplier2 = 2
    # import pandas as pd

    # # # Load the two files
    # df1 = pd.read_csv('/home/newberry4/jay_test/GAPUPDOWN_INTRADAY/NIFTY/CE/Trade_Sheets/df_ts_01.csv')
    # df2 = pd.read_csv('/home/newberry4/jay_test/GAPUPDOWN_INTRADAY/NIFTY/CE/Trade_Sheets/df_ts_02.csv')

    # # Concatenate the two dataframes along the rows (Y-axis)
    # df1['Total_Returns'] = df1['Total_Returns'] * multiplier1
    # df2['Total_Returns'] = df2['Total_Returns'] * multiplier2
    # concatenated_df = pd.concat([df1, df2], axis=0)


    # # Sort the concatenated dataframe by the Date column
    # # Ensure Date is treated as a datetime object for proper sorting
    # concatenated_df['Date'] = pd.to_datetime(concatenated_df['Date'])
    # concatenated_df = concatenated_df.sort_values(by='Date')

    # # Save the resulting dataframe to a new file
    # outPE_file = "/home/newberry4/jay_test/GAPUPDOWN_INTRADAY/NIFTY/CE/Trade_Sheets/gap_up_gap_down_intraday_lot_3_2.csv"
    # concatenated_df.to_csv(outPE_file, index=False)

    # print(f"Merged and sorted file saved as {outPE_file}")





