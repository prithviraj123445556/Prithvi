import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from datetime import datetime, time
import pandas as pd


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

# # --- Black-Scholes Price ---
# def black_scholes_price(S, K, T, r, sigma, option_type):
#     if sigma <= 0 or T <= 0:
#         return 0

#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     if option_type == "call":
#         return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     elif option_type == "put":
#         return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# # --- Implied Volatility ---
# def implied_volatility(option_price, S, K, T, r, option_type):
#     try:
#         iv = brentq(lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_price, 0.01, 5.0)
#         return iv
#     except (ValueError, RuntimeError):
#         return None


# # --- Black-Scholes Greeks ---
# def black_scholes_greeks(S, K, T, r, sigma, option_type):
#     if sigma <= 0 or T <= 0:
#         return None

#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
#     gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
#     vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
#     theta = None
#     rho = None

#     if option_type == "call":
#         theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
#         rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
#     elif option_type == "put":
#         theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
#         rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

#     return {
#         'Delta': delta,
#         'Gamma': gamma,
#         'Vega': vega,
#         'Theta': theta,
#         'Rho': rho
#     }

# # --- Time to Expiry (Fractional) ---
# def calculate_time_to_expiry(manual_datetime_str, expiry_date_str):
#     now = datetime.strptime(manual_datetime_str, "%Y-%m-%d %H:%M:%S")
#     expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%y").date()

#     market_open = time(9, 15)
#     market_close = time(15, 30)
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

#     print(f"\nManual Time Entered: {now}")
#     print(f"Expiry Date: {expiry_date}")
#     print(f"Days to Expiry (with fraction): {days_left - (fraction_of_day_passed if 0 <= current_minutes_since_open < total_trading_minutes else 0):.6f}")
#     print(f"Time to Expiry in Years (T): {T}")

#     return T

# # --- Inputs ---
# manual_datetime_str = '2024-07-17 09:30:00'
# expiry_date_str = '17-07-24'

# underlying_price = 23300
# call_strike =  23400
# put_strike = 23250
# call_price = 18.5
# put_price = 29.2
# risk_free_rate = 0.0696

# # --- Calculate Time and IVs ---
# T = calculate_time_to_expiry(manual_datetime_str, expiry_date_str)
# #atm_strike = min(available_strikes, key=lambda x: abs(x - underlying_price))

# call_iv = implied_volatility(call_price, underlying_price, call_strike, T, risk_free_rate, "call")
# put_iv = implied_volatility(put_price, underlying_price, put_strike, T, risk_free_rate, "put")

# # --- Calculate Greeks (if IV exists) ---
# call_greeks = black_scholes_greeks(underlying_price, call_strike, T, risk_free_rate, call_iv, "call") if call_iv else None
# put_greeks = black_scholes_greeks(underlying_price, put_strike, T, risk_free_rate, put_iv, "put") if put_iv else None

# # --- Output ---
# print("\n--- Option Snapshot ---")
# print(f"Underlying Price: {underlying_price}")
# print(f"Call Strike: {call_strike}")
# print(f"Call Price: {call_price}")
# print(f"Put Strike: {put_strike}")
# print(f"Put Price: {put_price}")

# print("\n--- Calculated IVs ---")
# print(f"ATM Call IV: {call_iv * 100:.2f}%" if call_iv else "ATM Call IV: None")
# print(f"ATM Put IV: {put_iv * 100:.2f}%" if put_iv else "ATM Put IV: None")

# print("\n--- Call Greeks ---")
# if call_greeks:
#     for greek, value in call_greeks.items():
#         print(f"{greek}: {value:.4f}")
# else:
#     print("Call Greeks: IV unavailable")

# print("\n--- Put Greeks ---")
# if put_greeks:
#     for greek, value in put_greeks.items():
#         print(f"{greek}: {value:.4f}")
# else:
#     print("Put Greeks: IV unavailable")






























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
    print("option_price", option_price)
    print("S", S)
    print("K", K)
    print("T", T)
    print("r", r)
    print("option_type", option_type)
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



def calculate_time_to_expiry(manual_datetime_str, expiry_date_str):
    # print("manual_datetime_str" ,type(manual_datetime_str))
    # print("expiry_date_str" ,type(expiry_date_str))
    # now = datetime.strptime(manual_datetime_str, "%Y-%m-%d %H:%M:%S")
    now = manual_datetime_str
    expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    today = now.date()
    days_left = (expiry_date - today).days
    if days_left <= 0:
        days_left += 1

    total_trading_minutes = (market_close.hour * 60 + market_close.minute) - (market_open.hour * 60 + market_open.minute)
    current_minutes_since_open = (now.hour * 60 + now.minute) - (market_open.hour * 60 + market_open.minute)

    if current_minutes_since_open < 0:
        T = round(days_left / 365, 6)
    elif current_minutes_since_open >= total_trading_minutes:
        T = round(max(0, (days_left - 1) / 365), 6)
    else:
        fraction_of_day_passed = current_minutes_since_open / total_trading_minutes
        T = round((days_left - fraction_of_day_passed) / 365, 6)

    return T





def create_option_chain_data_with_delta(En_Date,atm_strike,expiry,underlying_price, DELTA):
    try:
        # En_Date = pd.to_datetime(row['En_Date'])
        # atm_strike = row['Atm_Strike']
        # expiry = row['Expiry']
        # underlying_price = row['Spot_En_close']
        # manual_time_str = row['Manual_Time']

        T = calculate_time_to_expiry(En_Date, expiry)

        strikes = [atm_strike + i * 50 for i in range(-10, 11)]
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
            print("iv_ce", iv_ce)
            greeks_ce = black_scholes_greeks(underlying_price, strike, T, risk_free_rate, iv_ce, "CE") if iv_ce else None
            print("greeks_ce", greeks_ce)
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
        return df_chain
    except Exception as e:
        print("Error occurred while collecting data for row", e)
        return {}
    


# def get_option_chain_data(df_ts , option_data, DELTA):
#     start = time.perf_counter()
#     result_dict = {}
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         df_ts_list = [row for _, row in df_ts.iterrows()]
#         results = executor.map(lambda row: create_option_chain_data_with_delta(row, DELTA), df_ts_list)
#     for result in results:
#         result_dict.update(result)
#     finish = time.perf_counter()
#     print(f"Data collection completed in {np.round(finish - start, 2)} seconds")
#     return result_dict


superset = 'delta_hedging'
stock = 'NIFTY'
option_type = 'ND'


roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' else (50 if stock == 'FINNIFTY' else None))
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' else (3 if stock == 'FINNIFTY' else None))
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (40 if stock == 'FINNIFTY' else None))

# crossover = 'Upper' if Type == 'CE' else ('Lower' if Type == 'PE' else None)

# Define all the file paths
root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
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
    option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OHLC/"

expiry_file_path = rf"/home/newberry4/jay_data/{stock} market dates 2025 updated.xlsx"


En_Date = datetime.datetime.strptime('2025-03-13 10:40:00', '%Y-%m-%d %H:%M:%S')
expiry = '2024-03-28'

underlying_price = 22347
# call_strike =  23400
# put_strike = 23250
atm_strike = 22350
# call_price = 18.5
# put_price = 29.2
risk_free_rate = 0.0696
DELTA = 0.3









date_ranges = [ 
    ('2024-03-28', '2024-03-28'),      ## trail
#     # ('2025-02-14', '2025-02-27'),
                ]




for start_date, end_date in date_ranges: 
        # counter += 1
        # print(start_date, end_date, counter)
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
        option_data = option_data.sort_index()
        end_time = time.time()
        print('Time taken to pull Options data :', (end_time-start_time))




output_folder_path = rf'/home/newberry4/jay_test/delta_hedging/'
print("option_data ",  option_data)
# result_option_chain_dict = get_option_chain_data(df_ts, option_data, DELTA)
Option_Chain_Data = create_option_chain_data_with_delta(En_Date,atm_strike,expiry,underlying_price, DELTA)
print("Option Chain Data:", Option_Chain_Data)
inner_df = pd.DataFrame(Option_Chain_Data) # this gives you the actual DataFrame
# Step 2: Move index into a column (optional: name it clearly)
inner_df = inner_df.reset_index() 

# Step 3: Save it directly to CSV
inner_df.to_csv(f'{output_folder_path}result_dict_opt_11.csv', index=False)

print("Saved actual DataFrame with shape:", inner_df.shape)