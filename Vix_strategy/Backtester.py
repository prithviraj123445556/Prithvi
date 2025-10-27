# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, timedelta
import ast, json, sys, re
# warnings.filterwarnings("ignore")
sys.path.insert(0, r"/home/newberry4/jay_data/")
from Common_Functions.utils import TSL_PREMIUM, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes_d
from Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, compare_month_and_year



def postgresql_query(input_query, input_tuples = None):
    try:
        connection = psycopg2.connect( 
            host = "192.168.18.18",
            port = 5432,
            database = "postgres",
            user = "postgres",
            password = "New@123",
            # sslmode = "require"
        )
        
        cursor = connection.cursor()
        
        if input_tuples is not None:
            cursor.execute(input_query, input_tuples)
        else:
            cursor.execute(input_query)
        
        data = cursor.fetchall()
    
    except psycopg2.Error as e:
        print('Error connecting to the database:', e)
        return e
    
    else:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        
        return data


def resample_data(data, TIME_FRAME):
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data



def nearest_multiple(x, n):
    remainder = x%n
    # print(remainder)
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)


def get_strike(ATM, minute, daily_option_data, LONG_STRIKE, option_type):
    def get_premium(option_type, temp_option_data):
        subset = temp_option_data[
            (temp_option_data['StrikePrice'] == ATM) &
            (temp_option_data['Type'] == option_type)
        ]
        print("subset is", subset)
        if subset.empty:
            return None
        return subset['Open'].iloc[0]

    def find_nearest_strike(target_premium, option_type, temp_option_data):
        # Filter the data based on option type
        subset = temp_option_data[temp_option_data['Type'] == option_type].copy()

        if subset.empty:
            return None, None

        # Calculate the absolute difference from the target premium
        subset['Premium_Diff'] = (subset['Open'] - target_premium).abs()

        # Sort by 'Premium_Diff' ascending
        subset = subset.sort_values(by='Premium_Diff', ascending=True)
        print("subset", subset)

        # Take the first row after sorting
        nearest_strike = subset.iloc[0]['StrikePrice']
        nearest_premium = subset.iloc[0]['Open']
        print("nearest_strike nearest_premium", nearest_strike, nearest_premium)
        return nearest_strike, nearest_premium

    # ✅ Instead of stopping after 30 mins, continue until last available time
    last_time = daily_option_data.index.max()

    while minute <= last_time:
        temp_option_data = daily_option_data.loc[daily_option_data.index == minute].sort_index()

        if not temp_option_data.empty:
            ATM_premium = get_premium(option_type, temp_option_data)

            # if subset for ATM is empty, keep moving
            if ATM_premium is None:
                print(f"No ATM subset found for {minute}, checking next minute...")
                minute += pd.Timedelta(minutes=1)
                continue

            # we got ATM premium, now compute OTM
            OTM_premium_expected = ATM_premium * LONG_STRIKE
            print('OTM_premium_expected', OTM_premium_expected)

            OTM, OTM_prem = find_nearest_strike(OTM_premium_expected, option_type, temp_option_data)

            if OTM is not None:
                return OTM, OTM_prem
            else:
                print(f"No OTM subset found for {minute}, checking next minute...")

        else:
            print(f"No data at {minute}, checking next minute...")

        minute += pd.Timedelta(minutes=1)

    print("⚠️ No strike found till last available timestamp.")
    return None, None



## old code
# def get_strike(ATM, minute, daily_option_data, LONG_STRIKE ,option_type):
    
#     def get_premium(option_type):
        
#         subset = temp_option_data[(temp_option_data['StrikePrice'] == ATM) & (temp_option_data['Type'] == option_type) ]
#         print("subset is " ,subset)
#         if subset.empty:
#             return None
        
#         return subset['Open'].iloc[0]
    
#     def find_nearest_strike(target_premium, option_type):
#         # Filter the data based on option type and expiry
#         subset = temp_option_data[(temp_option_data['Type'] == option_type) ]
#         subset = subset.reset_index(drop=True)

        
#         # Check if the subset is empty
#         if subset.empty:
#             print("data not found")
#             return None, None

#         # Calculate the absolute difference from the target premium
#         subset['Premium_Diff'] = (subset['Open'] - target_premium).abs()
        
#         # Sort by 'Premium_Diff' ascending
#         subset = subset.sort_values(by='Premium_Diff', ascending=True)
#         print("subset",subset.sort_values(by='Premium_Diff', ascending=True))

#         # Take the first row after sorting
#         nearest_strike = subset.iloc[0]['StrikePrice']
#         nearest_premium = subset.iloc[0]['Open']
        
       
        
#         return nearest_strike, nearest_premium
    
    
#     # time = minute.strftime('%H:%M:%S')

#     # # Convert the filter time to a time object
#     # filter_time_object = pd.to_datetime(time).time()

#     # # Filter the DataFrame for the specific time
#     # temp_option_data = daily_option_data[daily_option_data.index.time == filter_time_object]
#     #temp_option_data = daily_option_data[daily_option_data['Time'] == time]

#     temp_option_data = daily_option_data.loc[daily_option_data.index == minute].sort_index()
#     print("temp_option_data",temp_option_data)
#     ATM_premium = get_premium(option_type)
#     # PE_Av TM_premium = get_premium('PE')
    
#     if (ATM_premium  is None):
#         return None, None
    
#     OTM_premium_expected = ATM_premium * LONG_STRIKE   
#     print('OTM_premium_expected',OTM_premium_expected) 
    
#     OTM ,OTM_prem = find_nearest_strike(OTM_premium_expected, option_type)
    
#     return  OTM , OTM_prem










# Function to get premium
def get_final_premium(premium_data, option_data, RATIO):
    option_data_ce = option_data[option_data['Type'] == 'CE'].drop(columns=['Ticker', 'Type'])
    option_data_pe = option_data[option_data['Type'] == 'PE'].drop(columns=['Ticker', 'Type'])
    
    del option_data

    premium_data['Time'] = pd.to_datetime(premium_data['Time'], format='%H:%M:%S')
    premium_data['Time'] = premium_data['Time'].dt.strftime('%H:%M')
    premium_data['ATM'] = premium_data['ATM'].astype('int32')

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    
    premium_data = premium_data.rename(columns={'Open' : 'CE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'CE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'CE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'PE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    long_lots = RATIO[0]
    short_lots = RATIO[1]

    premium_data['CE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['CE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['CE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['PE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['PE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['PE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['CE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['CE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['CE_OTM_Premium'] * 1.01 - brokerage))
    premium_data['PE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['PE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['PE_OTM_Premium'] * 1.01 - brokerage))

    premium_data['Premium'] = premium_data['CE_ATM_Price'] + premium_data['PE_ATM_Price'] + premium_data['CE_OTM_Price']  + premium_data['PE_OTM_Price']

    premium_data['Date'] = pd.to_datetime(premium_data['Date'], format='%Y-%m-%d')
    premium_data['ExpiryDate'] = pd.to_datetime(premium_data['ExpiryDate'], format='%Y-%m-%d')

    premium_data['DaysToExpiry'] = (premium_data['ExpiryDate'] - premium_data['Date']).dt.days
    premium_data['DaysToExpiry'] = np.where(premium_data['DaysToExpiry']==6, 4, np.where(premium_data['DaysToExpiry']==5, 3, premium_data['DaysToExpiry']))


    return premium_data


    





# Function to pull options data for specified date range 
def pull_options_data_d(start_date, end_date, option_data_path, stock):
    start_time = time.time()
    option_data_files = next(os.walk(option_data_path))[2]
    option_data = pd.DataFrame()

    for file in option_data_files:

        file1 = compare_month_and_year(start_date, end_date, file, stock)
              
        if not file1:
            continue

        temp_data = pd.read_pickle(option_data_path + file)[['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'Ticker']]
        temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
        temp_data = temp_data.rename_axis('DateTime')
        option_data = pd.concat([option_data, temp_data])

    # print('Option data columns :', option_data.columns)
    option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
    option_data['Type'] = option_data['Type'].astype('category')
    
    end_time = time.time()
    print('Time taken to pull Options data :', (end_time-start_time))

    return option_data






# Function to pull index data for specified date range
def pull_index_data(start_date_idx, end_date_idx, stock):
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
    index_data = pd.DataFrame(data, columns=column_names)
    index_data['Date'] = pd.to_datetime(index_data['Ticker'].str[0:8], format='%Y%m%d').astype(str)

    # Merge with mapped_days
    df = index_data.merge(mapped_days, on='Date')
    df.index = pd.to_datetime(df['Ticker'].str[0:13], format='%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()

    # Filter out dates without both 09:30 and 14:30 data
    valid_dates = []
    for date in df['Date'].unique():
        if (pd.Timestamp(f'{date} 09:30') in df.index) and (pd.Timestamp(f'{date} 14:30') in df.index):
            valid_dates.append(date)
        else:
            # Print the date that doesn't have either 09:30 or 14:30 data
            print(f"Missing required times for date: {date}")

    # Keep only rows with valid dates
    df = df[df['Date'].isin(valid_dates)]
    
    return df


def Square_Off_Func_IC2(option_data, next_time_period, expiry_date, LONG_STRIKE_CE, LONG_STRIKE_PE, EXIT, LONG_PT): 
    start_time = next_time_period
    end_time = pd.to_datetime(expiry_date.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')

    # Filter option data for the date, time, strike, and type of the entry position
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    # Sort by index to ensure data is ordered by time
    intraday_data_short_ce = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
    intraday_data_long_ce = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
    intraday_data_short_pe = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()
    intraday_data_long_pe = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()
    # print(intraday_data_long_pe)

    # Initialize variables to track entry and exit prices and times
    SHORT_STRIKE_entry_price_ce = SHORT_STRIKE_entry_price_pe = None
    LONG_STRIKE_entry_price_ce = LONG_STRIKE_entry_price_pe = None
    ce_exit = pe_exit = False  # Flags to track if exit has occurred for CE and PE legs

    # Retrieve entry prices for each leg (using Open price)
    try:
        SHORT_STRIKE_entry_price_ce = intraday_data_short_ce.iloc[0]['Open']
        LONG_STRIKE_entry_price_ce = intraday_data_long_ce.iloc[0]['Open']
    except IndexError:
        print(f"Entry price not found for CE strikes at {next_time_period}")
    try:
        SHORT_STRIKE_entry_price_pe = intraday_data_short_pe.iloc[0]['Open']
        LONG_STRIKE_entry_price_pe = intraday_data_long_pe.iloc[0]['Open']
    except IndexError:
        print(f"Entry price not found for PE strikes at {next_time_period}")

    # Condition to check if any of the required entry prices is missing
    if (SHORT_STRIKE_entry_price_ce is None or LONG_STRIKE_entry_price_ce is None or
        SHORT_STRIKE_entry_price_pe is None or LONG_STRIKE_entry_price_pe is None):
        return [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0], [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0]

    # Calculate stop loss and target values for both CE and PE
    ce_stoploss_value = SHORT_STRIKE_entry_price_ce * SHORT_SL
    ce_target_value = SHORT_STRIKE_entry_price_ce * (SHORT_PT)
    
    pe_stoploss_value = SHORT_STRIKE_entry_price_pe * SHORT_SL
    pe_target_value = SHORT_STRIKE_entry_price_pe * (SHORT_PT)

    ce_results, pe_results = [], []

    # Initialize the universal exit prices for CE and PE
    universal_exit_ce_price = intraday_data_short_ce.iloc[-1]['Open']
    universal_exit_long_ce_price = intraday_data_long_ce.iloc[-1]['Open']

    universal_exit_pe_price = intraday_data_short_pe.iloc[-1]['Open']
    universal_exit_long_pe_price = intraday_data_long_pe.iloc[-1]['Open']


    # Iterate through CE data to check for stop loss or target hit
    for idx, row in intraday_data_short_ce.iterrows():
        current_price = row['Open']  # Use Open price for Short leg CE
        ce_exit_time = idx

        # Check for CE Short leg stop loss hit
        if current_price >= ce_stoploss_value:
            # Exit CE Long at the same time (using Open price)
            long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
            ce_results = [
                SHORT_STRIKE_entry_price_ce, [current_price, universal_exit_ce_price], ce_exit_time, 
                LONG_STRIKE_entry_price_ce, [long_ce_exit_price, universal_exit_long_ce_price], ce_exit_time, 
                'SL hit', 'Exit', 'CE', 
                ce_stoploss_value, ce_target_value
            ]
            ce_exit = True
            break  # Exit both legs

        # Check for CE Short leg target hit
        if current_price <= ce_target_value:
            # Exit CE Long at the same time (using Open price)
            long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
            ce_results = [
                SHORT_STRIKE_entry_price_ce, [current_price,current_price], ce_exit_time, 
                LONG_STRIKE_entry_price_ce, [long_ce_exit_price,long_ce_exit_price], ce_exit_time, 
                'Target hit', 'Exit', 'CE', 
                ce_stoploss_value, ce_target_value
            ]
            ce_exit = True
            break

    # If no exit due to SL or PT, set universal exit for CE
    if not ce_exit:
        ce_results = [
            SHORT_STRIKE_entry_price_ce, [universal_exit_ce_price,universal_exit_ce_price], intraday_data_short_ce.index[-1], 
            LONG_STRIKE_entry_price_ce, [universal_exit_long_ce_price,universal_exit_long_ce_price], intraday_data_short_ce.index[-1], 
            'Universal exit', 'Exit', 'CE', 
            ce_stoploss_value, ce_target_value
        ]

    # Iterate through PE data to check for stop loss or target hit
    for idx, row in intraday_data_short_pe.iterrows():
        current_price = row['Open']  # Use Open price for Short leg PEs
        pe_exit_time = idx

        # Check for PE Short leg stop loss hit
        if current_price >= pe_stoploss_value:
            # Exit PE Long at the same time (using Open price)
            long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
            pe_results = [
                SHORT_STRIKE_entry_price_pe, [current_price, universal_exit_pe_price], pe_exit_time, 
                LONG_STRIKE_entry_price_pe, [long_pe_exit_price, universal_exit_long_pe_price], pe_exit_time, 
                'SL hit', 'Exit', 'PE', 
                pe_stoploss_value, pe_target_value
            ]
            pe_exit = True
            break  # Exit both legs

        # Check for PE Short leg target hit
        if current_price <= pe_target_value:
            # Exit PE Long at the same time (using Open price)
            long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
            pe_results = [
                SHORT_STRIKE_entry_price_pe, [current_price,current_price], pe_exit_time, 
                LONG_STRIKE_entry_price_pe, [long_pe_exit_price,long_pe_exit_price], pe_exit_time, 
                'Target hit', 'Exit', 'PE', 
                pe_stoploss_value, pe_target_value
            ]
            pe_exit = True
            break

    # If no exit due to SL or PT, set universal exit for PE
    if not pe_exit:
        pe_results = [
            SHORT_STRIKE_entry_price_pe, [universal_exit_pe_price,universal_exit_pe_price], intraday_data_short_pe.index[-1], 
            LONG_STRIKE_entry_price_pe, [universal_exit_long_pe_price,universal_exit_long_pe_price], intraday_data_short_pe.index[-1], 
            'Universal exit', 'Exit', 'PE', 
            pe_stoploss_value, pe_target_value
        ]

    return ce_results, pe_results




def trade_sheet_creator(mapped_days, vix_df , option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT):
    
    #column_names = ['Date', 'DateTime', 'Position', 'Time', 'Index Value', 'StrikePrice', 'Action', 'Type', 'Exit', 'ExpiryDate', 'DaysToExpiry', 'Lots', 'Premium']
    column_names = ['Date', 'vix', 'Position', 'Action', 'PE_Long_Time', 'PE_Short_Time', 'CE_Short_Time', 'CE_Long_Time', 'Index Value', 'PE_Long_Strike', 'PE_Short_Strike', 'CE_Short_Strike', 'CE_Long_Strike', 'ExpiryDate', 'DaysToExpiry','Day','PE_Long_Premium', 'PE_Short_Premium', 'CE_Short_Premium', 'CE_Long_Premium', 'PE_Long_Stoploss', 'PE_Short_Stoploss', 'CE_Short_Stoploss', 'CE_Long_Stoploss','PE_Short_target','CE_Short_target','PE_Long_Exit', 'PE_Short_Exit', 'CE_Short_Exit', 'CE_Long_Exit']
    trade_sheet = []   
    
    # list of parameters in each combination
    target_list =  [TIME_FRAME, SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT]
    
    # Check for each combination for each dte if that combination needs to be processed for the current time period
    # We are storing for each combination for each dte if it's profitable or not in filter_df
    
    # if counter!=1:
    #     filter_df_temp = filter_df[filter_df['Parameters'].apply(lambda x: x==str(target_list))].drop(columns=['Status'])
    #     #print(filter_df_temp)
    #     dte_to_process1 = filter_df_temp.columns[filter_df_temp.iloc[0] == 1].tolist()
    #     dte_to_process = [int(column[3:]) for column in dte_to_process1]
    #     #print(dte_to_process)
        
    #     mapped_days_temp = mapped_days[mapped_days['DaysToExpiry'].isin(dte_to_process)]
    # else:
    mapped_days_temp = mapped_days

    mapped_days_temp = mapped_days_temp[(mapped_days_temp['Date'] >= start_date) & 
                                    (mapped_days_temp['Date'] <= end_date)]
    mapped_days_temp = mapped_days_temp[(mapped_days_temp['Date'] > '2021-06-03') & 
                                    (mapped_days_temp['Date'] != '2024-05-18') & (mapped_days_temp['Date'] != '2024-05-20') ]
    # print(mapped_days_temp)
    # mapped_days_temp = mapped_days_temp[mapped_days_temp['DaysToExpiry'].isin(dte_list)]

    for _, row in mapped_days_temp.iterrows():
        date = row['Date']
        expiry_date = row['ExpiryDate']
        Day = row['Day']
        print(rf"processing for date :{date}")
        print(rf"processing for date :{expiry_date}")
        # expiry_date = row['2ndWeekExpiryDate']
        # expiry_date = row['3rdWeekExpiryDate']

        days_to_expiry = row['DaysToExpiry']
        # days_to_expiry = row['secondWeeklyDaysToExpiry']
        # days_to_expiry = row['thirdWeeklyDaysToExpiry']

        start_time = pd.to_datetime(f'{date} {ENTRY}:00')
        # second_entry_time = pd.to_datetime(f'{date} {ENTRY2}:00')
        end_time = pd.to_datetime(f'{date} {EXIT}:00')

        vix_df['Datetime'] = pd.to_datetime(vix_df['Date'] + ' ' + vix_df['Time'] + ':00')

        # Function to get Vix_Open value with fallback to the next available minute if not found
        def get_vix_open_value(df, target_time, date_label):
            vix_open_value = df.loc[df['Datetime'] == target_time, 'Open']
            
            # If no value is found, check 1 minute ahead
            if vix_open_value.empty:
                target_time += pd.Timedelta(minutes=1)
                vix_open_value = df.loc[df['Datetime'] == target_time, 'Open']
                if vix_open_value.empty:
                    print(f"No Vix_Open value found for {date_label} at {target_time - pd.Timedelta(minutes=1)} or 1 minute ahead.")
                    return None
                else:
                    print(f"No Vix_Open value found at {target_time - pd.Timedelta(minutes=1)}, using value from 1 minute ahead.")
            
            return vix_open_value.values[0]

        # Example usage for start_time and second_entry_time
        vix_open_value1 = get_vix_open_value(vix_df, start_time, "start_time")
        # vix_open_value2 = get_vix_open_value(vix_df, second_entry_time, "second_entry_time")

        # Output the results
        
        print("Vix Open Value 1:", vix_open_value1)
        # print("Vix Open Value 2:", vix_open_value2)
        

        daily_data_start_time = start_time - pd.Timedelta('5T')
        daily_data = resampled[(resampled.index >= daily_data_start_time) & (resampled.index <= end_time)]

        start_date_time = pd.to_datetime(date)
        expiry_date = pd.to_datetime(expiry_date)

        mask = (
    (pd.to_datetime(option_data.index).date >= start_date_time.date()) &
    (pd.to_datetime(option_data.index).date <= expiry_date.date()) &
    (option_data['ExpiryDate'] == str(expiry_date.date()))
        )

        daily_option_data = option_data.loc[mask].sort_index()


        mask2 = (
    (pd.to_datetime(option_data.index).date >= start_date_time.date()) &
    (pd.to_datetime(option_data.index).date <= start_date_time.date()) &
    (option_data['ExpiryDate'] == str(expiry_date.date()))
        )


        intraday_daily_option_data  = option_data.loc[mask2].sort_index()
        
        print(daily_option_data)
        def calculate_trade_entry(minute, vix_open_value,stock, daily_option_data, intraday_daily_option_data,days_to_expiry,LONG_STRIKE):
            current_index_open = daily_data.loc[minute, 'Open']
            
            if stock =='NIFTY':  
                ATM = nearest_multiple(current_index_open, 50)
            else:
                ATM = nearest_multiple(current_index_open, 100)
            print(ATM)
            if days_to_expiry == 0:
                days_to_expiry = 0.25

            days_to_expiry_sqrt = np.sqrt(days_to_expiry)
            if days_to_expiry_sqrt is None or vix_open_value is None or ATM is None:
                print(f"Missing data: days_to_expiry_sqrt={days_to_expiry_sqrt}, vix_open_value={vix_open_value}, ATM={ATM}")
                # print(minute)
                return None
            strike_adjustment = (days_to_expiry_sqrt / 16) * vix_open_value * ATM / 100
            if stock =='NIFTY':
                CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 50)
                PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 50)
            else :
                CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 100)
                PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 100)

            # CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 100)
            # PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 100)

            # CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 500)
            # PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 500)
            print("long strike", LONG_STRIKE)

            if LONG_STRIKE > 1:
                CE_Long_Strike = CE_Short_Strike + LONG_STRIKE
                PE_Long_Strike = PE_Short_Strike - LONG_STRIKE

                # --- Check CE premium ---
                ce_subset = intraday_daily_option_data[
                    (intraday_daily_option_data['StrikePrice'] == CE_Long_Strike) &
                    (intraday_daily_option_data['Type'] == 'CE')
                ]
                ce_long_premium = ce_subset['Open'].iloc[0] if not ce_subset.empty else None

                # --- Check PE premium ---
                pe_subset = intraday_daily_option_data[
                    (intraday_daily_option_data['StrikePrice'] == PE_Long_Strike) &
                    (intraday_daily_option_data['Type'] == 'PE')
                ]
                pe_long_premium = pe_subset['Open'].iloc[0] if not pe_subset.empty else None

                # --- If no premium found, use get_strike() ---

                if pe_long_premium is  None  or ce_long_premium is None:
                    LONG_STRIKE = 0.5
                    PE_Long_Strike, pe_long_premium = get_strike(PE_Short_Strike, minute, intraday_daily_option_data, LONG_STRIKE, 'PE')
                    CE_Long_Strike, ce_long_premium = get_strike(CE_Short_Strike, minute, intraday_daily_option_data, LONG_STRIKE, 'CE')

            else:
                print("CE_Short_Strike, PE_Short_Strike", CE_Short_Strike, PE_Short_Strike)
                CE_Long_Strike, ce_long_premium = get_strike(CE_Short_Strike, minute, intraday_daily_option_data, LONG_STRIKE, 'CE')
                print(CE_Long_Strike)
                PE_Long_Strike, pe_long_premium = get_strike(PE_Short_Strike, minute, intraday_daily_option_data, LONG_STRIKE, 'PE')


            print(CE_Long_Strike)
            ce_results, pe_results = Square_Off_Func_IC(daily_option_data,intraday_daily_option_data, minute, expiry_date, CE_Short_Strike, CE_Long_Strike, PE_Short_Strike, PE_Long_Strike, EXIT, SHORT_SL, LONG_PT)

            # Unpack results for CE and PE
            (CE_SHORT_STRIKE_entry_price, CE_SHORT_STRIKE_exit_price, ce_short_exit_time, CE_LONG_STRIKE_entry_price, CE_LONG_STRIKE_exit_price,
            ce_long_exit_time, ce_short_exit_type, ce_long_exit_type, _, ce_short_stoploss, ce_short_target) = ce_results
            (PE_SHORT_STRIKE_entry_price, PE_SHORT_STRIKE_exit_price, pe_short_exit_time, PE_LONG_STRIKE_entry_price, PE_LONG_STRIKE_exit_price,
            pe_long_exit_time, pe_short_exit_type, pe_long_exit_type, _, pe_short_stoploss, pe_short_target) = pe_results

            minute_time = minute.time()
            pe_long_stoploss = 0
            ce_long_stoploss = 0

            # Append entry and exit rows to trade_sheet
            trade_sheet.append(pd.Series([date, vix_open_value, 1, 'Short', minute_time, minute_time, minute_time, minute_time, current_index_open,
                                        PE_Long_Strike, PE_Short_Strike, CE_Short_Strike, CE_Long_Strike, expiry_date, days_to_expiry,Day,
                                        PE_LONG_STRIKE_entry_price, PE_SHORT_STRIKE_entry_price, CE_SHORT_STRIKE_entry_price, CE_LONG_STRIKE_entry_price,
                                        pe_long_stoploss, pe_short_stoploss, ce_short_stoploss, ce_long_stoploss, pe_short_target, ce_short_target,
                                        '', '', '', ''], index=column_names))
            trade_sheet.append(pd.Series([date, '', 0, 'Long', pe_long_exit_time, pe_short_exit_time, ce_short_exit_time, ce_long_exit_time, '',
                                        PE_Long_Strike, PE_Short_Strike, CE_Short_Strike, CE_Long_Strike, expiry_date, days_to_expiry,Day,
                                        PE_LONG_STRIKE_exit_price, PE_SHORT_STRIKE_exit_price, CE_SHORT_STRIKE_exit_price, CE_LONG_STRIKE_exit_price,
                                        pe_long_stoploss, '', '', ce_long_stoploss, '', '', pe_long_exit_type, pe_short_exit_type, ce_short_exit_type,
                                        ce_long_exit_type], index=column_names))

        # First entry at start_time
        calculate_trade_entry(start_time, vix_open_value1, stock, daily_option_data,intraday_daily_option_data , days_to_expiry,LONG_STRIKE)

        # Additional entry at 14:30
        # calculate_trade_entry(second_entry_time, vix_open_value2)
                    
                   
    strategy_name = f'{stock}_candle_{TIME_FRAME}_short_{SHORT_STRIKE}_long_{LONG_STRIKE}_entry_{ENTRY}_exit_{EXIT}_premium_{PREMIUM_TH}_short_sl_{SHORT_SL}_short_pt_{LONG_PT}'
    sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')
    
    try:
        trade_sheet = pd.concat(trade_sheet, axis = 1).T
    except Exception as e:
        print(f"An error occurred: {e}")
        return sanitized_strategy_name + '_' + start_date + '_' + end_date

    
    # trade_sheet = get_final_premium(trade_sheet, option_data, RATIO)
    # trade_sheet['Time'] = pd.to_datetime(trade_sheet['Time'], format='%H:%M')
    # trade_sheet['Time'] = trade_sheet['Time'].dt.strftime('%H:%M:%S')
#     if EXIT_TP == 'Full':
#         trade_sheet['Premium'] = np.where(
#     trade_sheet['Action'] == 'Short', 
#     trade_sheet['CE_Short_Premium'] + trade_sheet['PE_Short_Premium'] - trade_sheet['CE_Long_Premium'] - trade_sheet['PE_Long_Premium'], 
#     -(trade_sheet['CE_Short_Premium'].apply(lambda x: x[0]) + trade_sheet['PE_Short_Premium'].apply(lambda x: x[0]) - trade_sheet['CE_Long_Premium'].apply(lambda x: x[0]) - trade_sheet['PE_Long_Premium'].apply(lambda x: x[0]))
# )   
#     elif EXIT_TP == 'Half' :
#         trade_sheet['Premium'] = trade_sheet.apply(
#     lambda x: calculate_premium(
#         x['Action'],
#         x['CE_Short_Premium'],  # This should be a singular value
#         x['PE_Short_Premium'],  # This should also be a singular value
#         x['CE_Long_Premium'] ,
#         x['PE_Long_Premium'] 
#     ),
#     axis=1
# )

    # trade_sheet['Premium'] = 0     

    # create filter_df to store profitable combo and dte
    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4','Day' ,'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
    row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
    filter_df1.loc[row_index, 'Parameters'] = target_list
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date
    
    # trade_sheet = trade_sheet[trade_sheet['Date'] > '2021-06-03']

    # Go through each dte for the current combo to check if it's profitable
    # for dte in dte_list:
        
    #     trade_sheet_temp = trade_sheet[trade_sheet['DaysToExpiry'] == dte]
        
    if not trade_sheet.empty:
            
        trade_sheet.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            
            # if (start_date=='2023-02-01') & (end_date=='2023-05-31') & ((trade_sheet_temp['Premium'].sum() * LOT_SIZE) > -3500) & (stock=='BANKNIFTY'):
            #     print(start_date, end_date, trade_sheet_temp['Premium'].sum() * LOT_SIZE)
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
            
            # elif trade_sheet_temp['Premium'].sum() > 0:
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
                
            #     # trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            # else:
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 0

    # Store the combo and it's dte which is profitable in filter_df file
    # if filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'].iloc[0] == 1:

    # for day in days:
    #     trade_sheet_temp = trade_sheet[trade_sheet['Day'] == day]
    #     if not trade_sheet_temp.empty:
    #         trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
    #         # if trade_sheet_temp['Premium'].sum() > 0:
    #         filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
    #         filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
                # trade_sheet_temp = trade_sheet_temp[trade_sheet_temp['Date']>'2021-06-03']
                # trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            # else:
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 0
        
    existing_csv_file = rf"{filter_df_path}/filter_df{counter}.csv"
    if os.path.isfile(existing_csv_file):
        filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
    else:
        filter_df1.to_csv(existing_csv_file, index=False)
        
    return sanitized_strategy_name + '_' + str(start_date) + '_' + str(end_date)



def calculate_premium(action, ce_short, pe_short, ce_long, pe_long):
    # Check if all inputs are integers
    if all(isinstance(i, int) for i in [ce_short, pe_short, ce_long, pe_long]):
        return 0  # Directly return 0 if all are integers

    if action == 'Short':  
        # For short actions, expect singular integer values or lists
        return (2 * ce_short + 2 * pe_short) - (2 * ce_long + 2 * pe_long)
    else:
        # For long actions, expect lists for calculations
        return -((sum(ce_short) + sum(pe_short)) - (sum(ce_long) + sum(pe_long)))




def Square_Off_Func_IC(option_data,intraday_daily_option_data, next_time_period, expiry_date, SHORT_STRIKE_CE, LONG_STRIKE_CE, SHORT_STRIKE_PE, LONG_STRIKE_PE, EXIT, SHORT_SL, SHORT_PT): 
    start_time = next_time_period
    print("start_time",start_time)
    end_time = pd.to_datetime(expiry_date.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')

    # Filter option data for the date, time, strike, and type of the entry position
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)].sort_index()
    print("intraday_data",intraday_data)
    intraday_data_day = intraday_daily_option_data[(intraday_daily_option_data.index >= start_time) & (intraday_daily_option_data.index <= end_time)].sort_index()
    
    # Sort by index to ensure data is ordered by time
    intraday_data_short_ce = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
    intraday_data_long_ce = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
    intraday_data_short_pe = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()
    intraday_data_long_pe = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()
    # print("intraday_data_short_pe",intraday_data_short_pe)

    intraday_data_short_ce_start = intraday_data_day[(intraday_data_day['StrikePrice'] == SHORT_STRIKE_CE) & (intraday_data_day['Type'] == 'CE')].sort_index()
    intraday_data_long_ce_start  = intraday_data_day[(intraday_data_day['StrikePrice'] == LONG_STRIKE_CE) & (intraday_data_day['Type'] == 'CE')].sort_index()
    intraday_data_short_pe_start  = intraday_data_day[(intraday_data_day['StrikePrice'] == SHORT_STRIKE_PE) & (intraday_data_day['Type'] == 'PE')].sort_index()
    intraday_data_long_pe_start  = intraday_data_day[(intraday_data_day['StrikePrice'] == LONG_STRIKE_PE) & (intraday_data_day['Type'] == 'PE')].sort_index()

    # Initialize variables to track entry and exit prices and times
    SHORT_STRIKE_entry_price_ce = SHORT_STRIKE_entry_price_pe = None
    LONG_STRIKE_entry_price_ce = LONG_STRIKE_entry_price_pe = None
    ce_exit = pe_exit = False  # Flags to track if exit has occurred for CE and PE legs

    # Retrieve entry prices for each leg (using Open price)
    try:
        SHORT_STRIKE_entry_price_ce = intraday_data_short_ce_start.iloc[0]['Open']
        LONG_STRIKE_entry_price_ce = intraday_data_long_ce_start.iloc[0]['Open']
    except IndexError:
        print(f"Entry price not found for CE strikes at {next_time_period}")

    try:
        SHORT_STRIKE_entry_price_pe = intraday_data_short_pe_start.iloc[0]['Open']
        LONG_STRIKE_entry_price_pe = intraday_data_long_pe_start.iloc[0]['Open']
    except IndexError:
        print(f"Entry price not found for PE strikes at {next_time_period}")


    # Condition to check if any of the required entry prices is missing
    if (SHORT_STRIKE_entry_price_ce is None or LONG_STRIKE_entry_price_ce is None or
        SHORT_STRIKE_entry_price_pe is None or LONG_STRIKE_entry_price_pe is None):
        return [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0], [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0]

    # Calculate stop loss and target values for both CE and PE
    ce_stoploss_value = SHORT_STRIKE_entry_price_ce * SHORT_SL
    ce_target_value = SHORT_STRIKE_entry_price_ce * (SHORT_PT)
    
    pe_stoploss_value = SHORT_STRIKE_entry_price_pe * SHORT_SL
    pe_target_value = SHORT_STRIKE_entry_price_pe * (SHORT_PT)

    ce_results, pe_results = [], []

    # Initialize the universal exit prices for CE and PE
    universal_exit_ce_price = intraday_data_short_ce.iloc[-1]['Open']
    universal_exit_long_ce_price = intraday_data_long_ce.iloc[-1]['Open']

    universal_exit_pe_price = intraday_data_short_pe.iloc[-1]['Open']
    universal_exit_long_pe_price = intraday_data_long_pe.iloc[-1]['Open']


    # Iterate through CE data to check for stop loss or target hit
    for idx, row in intraday_data_short_ce.iterrows():
        current_price = row['Open']  # Use Open price for Short leg CE
        ce_exit_time = idx

        # Check for CE Short leg stop loss hit
        if current_price >= ce_stoploss_value:
            # Exit CE Long at the same time (using Open price)
            long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
            ce_results = [
                SHORT_STRIKE_entry_price_ce, [current_price, universal_exit_ce_price], ce_exit_time, 
                LONG_STRIKE_entry_price_ce, [long_ce_exit_price, universal_exit_long_ce_price], ce_exit_time, 
                'SL hit', 'Exit', 'CE', 
                ce_stoploss_value, ce_target_value
            ]
            ce_exit = True
            break  # Exit both legs

        # Check for CE Short leg target hit
        if current_price <= ce_target_value:
            # Exit CE Long at the same time (using Open price)
            long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
            ce_results = [
                SHORT_STRIKE_entry_price_ce, [current_price,current_price], ce_exit_time, 
                LONG_STRIKE_entry_price_ce, [long_ce_exit_price,long_ce_exit_price], ce_exit_time, 
                'Target hit', 'Exit', 'CE', 
                ce_stoploss_value, ce_target_value
            ]
            ce_exit = True
            break

    # If no exit due to SL or PT, set universal exit for CE
    if not ce_exit:
        ce_results = [
            SHORT_STRIKE_entry_price_ce, [universal_exit_ce_price,universal_exit_ce_price], intraday_data_short_ce.index[-1], 
            LONG_STRIKE_entry_price_ce, [universal_exit_long_ce_price,universal_exit_long_ce_price], intraday_data_short_ce.index[-1], 
            'Universal exit', 'Exit', 'CE', 
            ce_stoploss_value, ce_target_value
        ]

    # Iterate through PE data to check for stop loss or target hit
    for idx, row in intraday_data_short_pe.iterrows():
        current_price = row['Open']  # Use Open price for Short leg PEs
        pe_exit_time = idx

        # Check for PE Short leg stop loss hit
        if current_price >= pe_stoploss_value:
            # Exit PE Long at the same time (using Open price)
            long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
            pe_results = [
                SHORT_STRIKE_entry_price_pe, [current_price, universal_exit_pe_price], pe_exit_time, 
                LONG_STRIKE_entry_price_pe, [long_pe_exit_price, universal_exit_long_pe_price], pe_exit_time, 
                'SL hit', 'Exit', 'PE', 
                pe_stoploss_value, pe_target_value
            ]
            pe_exit = True
            break  # Exit both legs

        # Check for PE Short leg target hit
        if current_price <= pe_target_value:
            # Exit PE Long at the same time (using Open price)
            long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
            pe_results = [
                SHORT_STRIKE_entry_price_pe, [current_price,current_price], pe_exit_time, 
                LONG_STRIKE_entry_price_pe, [long_pe_exit_price,long_pe_exit_price], pe_exit_time, 
                'Target hit', 'Exit', 'PE', 
                pe_stoploss_value, pe_target_value
            ]
            pe_exit = True
            break

    # If no exit due to SL or PT, set universal exit for PE
    if not pe_exit:
        pe_results = [
            SHORT_STRIKE_entry_price_pe, [universal_exit_pe_price,universal_exit_pe_price], intraday_data_short_pe.index[-1], 
            LONG_STRIKE_entry_price_pe, [universal_exit_long_pe_price,universal_exit_long_pe_price], intraday_data_short_pe.index[-1], 
            'Universal exit', 'Exit', 'PE', 
            pe_stoploss_value, pe_target_value
        ]

    return ce_results, pe_results




# def Square_Off_Func_IC(option_data, next_time_period, expiry_date, SHORT_STRIKE_CE, LONG_STRIKE_CE, SHORT_STRIKE_PE, LONG_STRIKE_PE, EXIT, SHORT_SL, SHORT_PT): 
#     start_time = next_time_period
#     end_time = pd.to_datetime(expiry_date.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')

#     # Filter option data for the date, time, strike, and type of the entry position
#     intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

#     # Sort by index to ensure data is ordered by time
#     intraday_data_short_ce = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
#     intraday_data_long_ce = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_CE) & (intraday_data['Type'] == 'CE')].sort_index()
#     intraday_data_short_pe = intraday_data[(intraday_data['StrikePrice'] == SHORT_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()
#     intraday_data_long_pe = intraday_data[(intraday_data['StrikePrice'] == LONG_STRIKE_PE) & (intraday_data['Type'] == 'PE')].sort_index()

#     # Initialize variables to track entry and exit prices and times
#     SHORT_STRIKE_entry_price_ce = SHORT_STRIKE_entry_price_pe = None
#     LONG_STRIKE_entry_price_ce = LONG_STRIKE_entry_price_pe = None
#     ce_exit = pe_exit = False  # Flags to track if exit has occurred for CE and PE legs

#     # Retrieve entry prices for each leg (using Open price)
#     try:
#         SHORT_STRIKE_entry_price_ce = intraday_data_short_ce.iloc[0]['Open']
#         LONG_STRIKE_entry_price_ce = intraday_data_long_ce.iloc[0]['Open']
#     except IndexError:
#         print(f"Entry price not found for CE strikes at {next_time_period}")

#     try:
#         SHORT_STRIKE_entry_price_pe = intraday_data_short_pe.iloc[0]['Open']
#         LONG_STRIKE_entry_price_pe = intraday_data_long_pe.iloc[0]['Open']
#     except IndexError:
#         print(f"Entry price not found for PE strikes at {next_time_period}")

#     # Condition to check if any of the required entry prices is missing
#     if (SHORT_STRIKE_entry_price_ce is None or LONG_STRIKE_entry_price_ce is None or
#         SHORT_STRIKE_entry_price_pe is None or LONG_STRIKE_entry_price_pe is None):
#         return [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0], [0, 0, 0, 0, 0, 0, 'No entry', 'No exit', 'CE/PE', 0, 0]

#     # Calculate stop loss and target values for both CE and PE
#     ce_stoploss_value = SHORT_STRIKE_entry_price_ce * SHORT_SL
#     print(SHORT_STRIKE_entry_price_ce, ce_stoploss_value)
#     ce_target_value = SHORT_STRIKE_entry_price_ce * (1 - SHORT_PT)
    
#     pe_stoploss_value = SHORT_STRIKE_entry_price_pe * SHORT_SL
#     print(SHORT_STRIKE_entry_price_pe, pe_stoploss_value)
#     pe_target_value = SHORT_STRIKE_entry_price_pe * (1 - SHORT_PT)

#     ce_results, pe_results = [], []

#     # Iterate through CE data to check for stop loss or target hit
#     for idx, row in intraday_data_short_ce.iterrows():
#         current_price = row['Open']  # Use Open price for Short leg CE
#         ce_exit_time = idx

#         # Check for CE Short leg stop loss hit
#         if current_price >= ce_stoploss_value:
#             # Exit CE Long at the same time (using Open price)
#             long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
#             ce_results = [
#                 SHORT_STRIKE_entry_price_ce, current_price, ce_exit_time, 
#                 LONG_STRIKE_entry_price_ce, long_ce_exit_price, ce_exit_time, 
#                 'SL hit', 'Exit', 'CE', 
#                 ce_stoploss_value, ce_target_value
#             ]
#             ce_exit = True
#             break  # Exit both legs

#         # Check for CE Short leg target hit
#         if current_price <= ce_target_value:
#             # Exit CE Long at the same time (using Open price)
#             long_ce_exit_price = intraday_data_long_ce.loc[ce_exit_time]['Open'] if ce_exit_time in intraday_data_long_ce.index else intraday_data_long_ce.iloc[-1]['Open']
#             ce_results = [
#                 SHORT_STRIKE_entry_price_ce, current_price, ce_exit_time, 
#                 LONG_STRIKE_entry_price_ce, long_ce_exit_price, ce_exit_time, 
#                 'Target hit', 'Exit', 'CE', 
#                 ce_stoploss_value, ce_target_value
#             ]
#             ce_exit = True
#             break

#     # If no exit due to SL or PT, set universal exit for CE
#     if not ce_exit:
#         last_row = intraday_data_short_ce.iloc[-1]
#         long_ce_exit_price = intraday_data_long_ce.iloc[-1]['Open']  # Exit Long leg CE using Open price
#         ce_results = [
#             SHORT_STRIKE_entry_price_ce, last_row['Open'], last_row.name, 
#             LONG_STRIKE_entry_price_ce, long_ce_exit_price, last_row.name, 
#             'Universal exit', 'Exit', 'CE', 
#             ce_stoploss_value, ce_target_value
#         ]

#     # Iterate through PE data to check for stop loss or target hit
#     for idx, row in intraday_data_short_pe.iterrows():
#         current_price = row['Open']  # Use Open price for Short leg PE
#         pe_exit_time = idx

#         # Check for PE Short leg stop loss hit
#         if current_price >= pe_stoploss_value:
#             # Exit PE Long at the same time (using Open price)
#             long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
#             pe_results = [
#                 SHORT_STRIKE_entry_price_pe, current_price, pe_exit_time, 
#                 LONG_STRIKE_entry_price_pe, long_pe_exit_price, pe_exit_time, 
#                 'SL hit', 'Exit', 'PE', 
#                 pe_stoploss_value, pe_target_value
#             ]
#             pe_exit = True
#             break  # Exit both legs

#         # Check for PE Short leg target hit
#         if current_price <= pe_target_value:
#             # Exit PE Long at the same time (using Open price)
#             long_pe_exit_price = intraday_data_long_pe.loc[pe_exit_time]['Open'] if pe_exit_time in intraday_data_long_pe.index else intraday_data_long_pe.iloc[-1]['Open']
#             pe_results = [
#                 SHORT_STRIKE_entry_price_pe, current_price, pe_exit_time, 
#                 LONG_STRIKE_entry_price_pe, long_pe_exit_price, pe_exit_time, 
#                 'Target hit', 'Exit', 'PE', 
#                 pe_stoploss_value, pe_target_value
#             ]
#             pe_exit = True
#             break

#     # If no exit due to SL or PT, set universal exit for PE
#     if not pe_exit:
#         last_row = intraday_data_short_pe.iloc[-1]
#         long_pe_exit_price = intraday_data_long_pe.iloc[-1]['Open']  # Exit Long leg PE using Open price
#         pe_results = [
#             SHORT_STRIKE_entry_price_pe, last_row['Open'], last_row.name, 
#             LONG_STRIKE_entry_price_pe, long_pe_exit_price, last_row.name, 
#             'Universal exit', 'Exit', 'PE', 
#             pe_stoploss_value, pe_target_value
#         ]

#     return ce_results, pe_results



         
#     if PREMIUM_TP=='MIN':

#         pass
#         # if CE_OTM_entry_price < PE_OTM_entry_price:
#         #     ce_target_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
#         #     pe_target_pt = PE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
        
#         # else:
#         #     ce_target_pt = CE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
#         #     pe_target_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT

#     else:
#         short_stoploss_pt_ce = SHORT_STRIKE_entry_price_ce + SHORT_STRIKE_entry_price_ce * SHORT_SL
#         long_profit_pt_ce = LONG_STRIKE_entry_price_ce + LONG_STRIKE_entry_price_ce * LONG_PT

#         short_stoploss_pt_pe = SHORT_STRIKE_entry_price_pe + SHORT_STRIKE_entry_price_pe * SHORT_SL
#         long_profit_pt_pe = LONG_STRIKE_entry_price_pe + LONG_STRIKE_entry_price_pe * LONG_PT

#     crosses_threshold_short_ce = (intraday_data_short_ce['Open'] > short_stoploss_pt_ce)
#     crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#     crosses_threshold_short_pe = (intraday_data_short_pe['Open'] > short_stoploss_pt_pe)
#     crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#     short_ce_stoploss_leg, short_pe_stoploss_leg, long_ce_stoploss_leg, long_pe_stoploss_leg = 0, 0, 0, 0
#     short_ce_universal_leg, short_pe_universal_leg, long_ce_universal_leg, long_pe_universal_leg = 0, 0, 0, 0

#     if crosses_threshold_short_ce.any() & crosses_threshold_short_pe.any() & (crosses_threshold_short_ce.idxmax() < crosses_threshold_short_pe.idxmax()):
#         # print('Code is here - 1')

#         # # if short and long both leg exit at the same time
#         # if crosses_threshold_short_ce.any() & crosses_threshold_long.any() & (crosses_threshold_short.idxmax() == crosses_threshold_long.idxmax()):

#         #     crosses_threshold_long_index = crosses_threshold_long.idxmax()

#         #     try:
#         #         LONG_STRIKE_exit_price = intraday_data_long.at[crosses_threshold_long_index + pd.to_timedelta('1T'), 'Open']
#         #     except:
#         #         print(crosses_threshold_long_index + pd.to_timedelta('1T'))
#         #         next_index = intraday_data_long.index[intraday_data_long.index < (crosses_threshold_long_index + pd.to_timedelta('1T'))].max()
#         #         LONG_STRIKE_exit_price = intraday_data_long.at[next_index, 'Open']

#         #     long_exit_time_period = crosses_threshold_long_index + pd.to_timedelta('1T')
#         #     long_exit_type = 'Long Stoploss Exit'
#         #     long_stoploss = 1
        
#         #     crosses_threshold_short_index = crosses_threshold_short.idxmax()
#         #     try:
#         #         SHORT_STRIKE_exit_price = intraday_data_short.at[crosses_threshold_short_index + pd.to_timedelta('1T'), 'Open']
#         #     except:
#         #         print(crosses_threshold_short_index + pd.to_timedelta('1T'))
#         #         next_index = intraday_data_short.index[intraday_data_short.index < (crosses_threshold_short_index + pd.to_timedelta('1T'))].max()
#         #         SHORT_STRIKE_exit_price = intraday_data_short.at[next_index, 'Open']

#         #     short_exit_time_period = crosses_threshold_short_index + pd.to_timedelta('1T')
#         #     short_exit_type = 'Short Stoploss Exit'
#         #     short_stoploss = 1

#         # if short ce leg exit at EXIT time 
#         if intraday_data_short_ce[intraday_data_short_ce.index > crosses_threshold_short_ce.idxmax()].empty:
            
#             short_ce_universal_leg = 1
#             short_ce_stoploss_leg = 0

#             # # short ce universal exit
#             short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#             # try:
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#             # except:    
#             #     print(short_exit_time_period_ce)
#             #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
            
#             # short_exit_type_ce = 'Short CE Universal Exit'
#             # short_stoploss_ce = 0
#             long_exit_type_ce = ''

#         else:
        
#             short_ce_universal_leg = 0
#             short_ce_stoploss_leg = 1

#             crosses_threshold_short_index_ce = crosses_threshold_short_ce.idxmax()
#             # try:
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[crosses_threshold_short_index_ce + pd.to_timedelta('1T'), 'Open']
#             # except:
#             #     print(crosses_threshold_short_index_ce + pd.to_timedelta('1T'))
#             #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < (crosses_threshold_short_index_ce + pd.to_timedelta('1T'))].max()
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']

#             short_exit_time_period_ce = crosses_threshold_short_index_ce + pd.to_timedelta('1T')
#             # short_exit_type_ce = 'Short CE Stoploss Exit'
#             # short_stoploss_ce = 1
#             long_exit_type_ce = ''

#         if  long_exit_type_ce != 'Long CE Stoploss Exit':
#             intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= crosses_threshold_short_ce.idxmax()]
#             crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#             if crosses_threshold_long_ce.any():
                
#                 # Long ce universal exit
#                 if intraday_data_long_ce[intraday_data_long_ce.index > crosses_threshold_long_ce.idxmax()].empty:
                    
#                     long_ce_universal_leg = 1
#                     long_ce_stoploss_leg = 0

#                     # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                     # except:    
#                     #     print(long_exit_time_period_ce)
#                     #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                     # long_exit_type_ce = 'Long CE Universal Exit'
#                     # long_stoploss_ce = 0

#                 else:    

#                     long_ce_universal_leg = 0
#                     long_ce_stoploss_leg = 1

#                     # crosses_threshold_long_index_ce = crosses_threshold_long_ce.idxmax()
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[crosses_threshold_long_index_ce + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (crosses_threshold_long_index_ce + pd.to_timedelta('1T'))].max()
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                     # long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#                     # long_exit_type_ce = 'Long CE Stoploss Exit'
#                     # long_stoploss_ce = 1

#             else:

#                 long_ce_universal_leg = 1
#                 long_ce_stoploss_leg = 0


#                 # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open'] 
                
#                 # long_exit_type_ce = 'Long CE Universal Exit'
#                 # long_stoploss_ce = 0

#             # updated pe stoploss 
#             intraday_data_short_pe = intraday_data_short_pe[intraday_data_short_pe.index >= short_exit_time_period_ce]
#             SHORT_STRIKE_new_entry_price_pe = intraday_data_short_pe.iloc[0]['Open']

#             intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= short_exit_time_period_ce]
#             LONG_STRIKE_new_entry_price_pe = intraday_data_long_pe.iloc[0]['Open']

#             short_stoploss_pt_pe = SHORT_STRIKE_new_entry_price_pe + SHORT_STRIKE_new_entry_price_pe * SHORT_SL
#             long_profit_pt_pe = LONG_STRIKE_new_entry_price_pe + LONG_STRIKE_new_entry_price_pe * LONG_PT

#             crosses_threshold_short_pe = (intraday_data_short_pe['Open'] > short_stoploss_pt_pe)
#             crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#             if crosses_threshold_short_pe.any():

#                 if intraday_data_short_pe[intraday_data_short_pe.index > crosses_threshold_short_pe.idxmax()].empty:

#                     short_pe_universal_leg = 1
#                     short_pe_stoploss_leg = 0

#                     # # short pe universal exit
#                     short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                     # try:
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#                     # except:    
#                     #     print(short_exit_time_period_pe)
#                     #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
                    
#                     # short_exit_type_pe = 'Short PE Universal Exit'
#                     # short_stoploss_pe = 0
#                     long_exit_type_pe = ''

#                 else:

#                     short_pe_universal_leg = 0
#                     short_pe_stoploss_leg = 1

#                     # # short pe stoploss exit                    
#                     crosses_threshold_short_index_pe = crosses_threshold_short_pe.idxmax()
#                     # try:
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[crosses_threshold_short_index_pe + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_short_index_pe + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < (crosses_threshold_short_index_pe + pd.to_timedelta('1T'))].max()
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']

#                     short_exit_time_period_pe = crosses_threshold_short_index_pe + pd.to_timedelta('1T')
#                     # short_exit_type_pe = 'Short PE Stoploss Exit'
#                     # short_stoploss_pe = 1
#                     long_exit_type_pe = ''

#                 if  long_exit_type_pe != 'Long PE Stoploss Exit':
#                     intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= crosses_threshold_short_pe.idxmax()]
#                     crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#                     if crosses_threshold_long_pe.any():
                        
#                         # long pe universal exit
#                         if intraday_data_long_pe[intraday_data_long_pe.index > crosses_threshold_long_pe.idxmax()].empty:
                            
#                             long_pe_universal_leg = 1
#                             long_pe_stoploss_leg = 0

#                             # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                             # except:    
#                             #     print(long_exit_time_period_pe)
#                             #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                             # long_exit_type_pe = 'Long PE Universal Exit'
#                             # long_stoploss_pe = 0

#                         else:    

#                             long_pe_universal_leg = 0
#                             long_pe_stoploss_leg = 1                           

#                             # # long pe stoploss exit
#                             # crosses_threshold_long_index_pe = crosses_threshold_long_pe.idxmax()
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[crosses_threshold_long_index_pe + pd.to_timedelta('1T'), 'Open']
#                             # except:
#                             #     print(crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#                             #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < (crosses_threshold_long_index_pe + pd.to_timedelta('1T'))].max()
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                             # long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#                             # long_exit_type_pe = 'Long PE Stoploss Exit'
#                             # long_stoploss_pe = 1 
#                     else:

#                         long_pe_universal_leg = 1
#                         long_pe_stoploss_leg = 0

#                         # # long pe universal exit
#                         # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                         # try:    
#                         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                         # except:
#                         #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                        
#                         # long_exit_type_pe = 'Long PE Universal Exit'
#                         # long_stoploss_pe = 0
#             else:
                
#                 short_pe_universal_leg = 1
#                 short_pe_stoploss_leg = 0

#                 # # short pe universal exit
#                 short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:
#                 #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#                 # except:    
#                 #     print(short_exit_time_period_pe)
#                 #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#                 #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
                
#                 # short_exit_type_pe = 'Short PE Universal Exit'
#                 # short_stoploss_pe = 0
#                 long_exit_type_pe = ''

#                 long_pe_universal_leg = 1
#                 long_pe_stoploss_leg = 0

#                 # # long pe universal exit
#                 # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                
#                 # long_exit_type_pe = 'Long PE Universal Exit'
#                 # long_stoploss_pe = 0                
# ##################################################################################################
#     elif crosses_threshold_short_ce.any() & crosses_threshold_short_pe.any() & (crosses_threshold_short_ce.idxmax() > crosses_threshold_short_pe.idxmax()):
        
#         # print('Code is here - 2')

#         # # if short and long both leg exit at the same time
#         # if crosses_threshold_short_ce.any() & crosses_threshold_long.any() & (crosses_threshold_short.idxmax() == crosses_threshold_long.idxmax()):

#         #     crosses_threshold_long_index = crosses_threshold_long.idxmax()

#         #     try:
#         #         LONG_STRIKE_exit_price = intraday_data_long.at[crosses_threshold_long_index + pd.to_timedelta('1T'), 'Open']
#         #     except:
#         #         print(crosses_threshold_long_index + pd.to_timedelta('1T'))
#         #         next_index = intraday_data_long.index[intraday_data_long.index < (crosses_threshold_long_index + pd.to_timedelta('1T'))].max()
#         #         LONG_STRIKE_exit_price = intraday_data_long.at[next_index, 'Open']

#         #     long_exit_time_period = crosses_threshold_long_index + pd.to_timedelta('1T')
#         #     long_exit_type = 'Long Stoploss Exit'
#         #     long_stoploss = 1
        
#         #     crosses_threshold_short_index = crosses_threshold_short.idxmax()
#         #     try:
#         #         SHORT_STRIKE_exit_price = intraday_data_short.at[crosses_threshold_short_index + pd.to_timedelta('1T'), 'Open']
#         #     except:
#         #         print(crosses_threshold_short_index + pd.to_timedelta('1T'))
#         #         next_index = intraday_data_short.index[intraday_data_short.index < (crosses_threshold_short_index + pd.to_timedelta('1T'))].max()
#         #         SHORT_STRIKE_exit_price = intraday_data_short.at[next_index, 'Open']

#         #     short_exit_time_period = crosses_threshold_short_index + pd.to_timedelta('1T')
#         #     short_exit_type = 'Short Stoploss Exit'
#         #     short_stoploss = 1

#         # if short pe leg exit at EXIT time 
#         if intraday_data_short_pe[intraday_data_short_pe.index > crosses_threshold_short_pe.idxmax()].empty:

#             short_pe_universal_leg = 1
#             short_pe_stoploss_leg = 0

#             # # Short universal exit
#             short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#             # try:
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#             # except:    
#             #     print(short_exit_time_period_pe)
#             #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
            
#             # short_exit_type_pe = 'Short PE Universal Exit'
#             # short_stoploss_pe = 0
#             long_exit_type_pe = ''

#         else:
            
#             short_pe_universal_leg = 0
#             short_pe_stoploss_leg = 1

#             crosses_threshold_short_index_pe = crosses_threshold_short_pe.idxmax()
#             # try:
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[crosses_threshold_short_index_pe + pd.to_timedelta('1T'), 'Open']
#             # except:
#             #     print(crosses_threshold_short_index_pe + pd.to_timedelta('1T'))
#             #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < (crosses_threshold_short_index_pe + pd.to_timedelta('1T'))].max()
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']

#             short_exit_time_period_pe = crosses_threshold_short_index_pe + pd.to_timedelta('1T')
#             # short_exit_type_pe = 'Short PE Stoploss Exit'
#             # short_stoploss_pe = 1
#             long_exit_type_pe = ''

#         if  long_exit_type_pe != 'Long PE Stoploss Exit':

#             intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= crosses_threshold_short_pe.idxmax()]
#             crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#             if crosses_threshold_long_pe.any():

#                 # Long universal exit
#                 if intraday_data_long_pe[intraday_data_long_pe.index > crosses_threshold_long_pe.idxmax()].empty:
                    
#                     long_pe_universal_leg = 1
#                     long_pe_stoploss_leg = 0

#                     # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                     # except:    
#                     #     print(long_exit_time_period_pe)
#                     #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                     # long_exit_type_pe = 'Long PE Universal Exit'
#                     # long_stoploss_pe = 0

#                 else:    

#                     long_pe_universal_leg = 0
#                     long_pe_stoploss_leg = 1

#                     # crosses_threshold_long_index_pe = crosses_threshold_long_pe.idxmax()
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[crosses_threshold_long_index_pe + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < (crosses_threshold_long_index_pe + pd.to_timedelta('1T'))].max()
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                     # long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#                     # long_exit_type_pe = 'Long PE Stoploss Exit'
#                     # long_stoploss_pe = 1

#             else:
#                 long_pe_universal_leg = 1
#                 long_pe_stoploss_leg = 0

#                 # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                
#                 # long_exit_type_pe = 'Long PE Universal Exit'
#                 # long_stoploss_pe = 0

#             intraday_data_short_ce = intraday_data_short_ce[intraday_data_short_ce.index >= short_exit_time_period_pe]
#             SHORT_STRIKE_new_entry_price_ce = intraday_data_short_ce.iloc[0]['Open']

#             intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= short_exit_time_period_pe]
#             LONG_STRIKE_new_entry_price_ce = intraday_data_long_ce.iloc[0]['Open']

#             short_stoploss_pt_ce = SHORT_STRIKE_new_entry_price_ce + SHORT_STRIKE_new_entry_price_ce * SHORT_SL
#             long_profit_pt_ce = LONG_STRIKE_new_entry_price_ce + LONG_STRIKE_new_entry_price_ce * LONG_PT

#             crosses_threshold_short_ce = (intraday_data_short_ce['Open'] > short_stoploss_pt_ce)
#             crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#             if crosses_threshold_short_ce.any():

#                 if intraday_data_short_ce[intraday_data_short_ce.index > crosses_threshold_short_ce.idxmax()].empty:
                    
#                     short_ce_universal_leg = 1
#                     short_ce_stoploss_leg = 0

#                     # # Short universal exit
#                     short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                     # try:
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#                     # except:    
#                     #     print(short_exit_time_period_ce)
#                     #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
                    
#                     # short_exit_type_ce = 'Short CE Universal Exit'
#                     # short_stoploss_ce = 0
#                     long_exit_type_ce = ''

#                 else:
#                     # print('Code is here - 3')
#                     short_ce_universal_leg = 0
#                     short_ce_stoploss_leg = 1                

#                     crosses_threshold_short_index_ce = crosses_threshold_short_ce.idxmax()
#                     # try:
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[crosses_threshold_short_index_ce + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_short_index_ce + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < (crosses_threshold_short_index_ce + pd.to_timedelta('1T'))].max()
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']

#                     short_exit_time_period_ce = crosses_threshold_short_index_ce + pd.to_timedelta('1T')
#                     # short_exit_type_ce = 'Short CE Stoploss Exit'
#                     # short_stoploss_ce = 1
#                     long_exit_type_ce = ''

#                 if  long_exit_type_ce != 'Long CE Stoploss Exit':
#                     intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= crosses_threshold_short_ce.idxmax()]
#                     crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#                     if crosses_threshold_long_ce.any():
#                         # Long universal exit
#                         if intraday_data_long_ce[intraday_data_long_ce.index > crosses_threshold_long_ce.idxmax()].empty:
                            
#                             long_ce_universal_leg = 1
#                             long_ce_stoploss_leg = 0

#                             # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                             # except:    
#                             #     print(long_exit_time_period_ce)
#                             #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                             # long_exit_type_ce = 'Long CE Universal Exit'
#                             # long_stoploss_ce = 0

#                         else:    
#                             long_ce_universal_leg = 0
#                             long_ce_stoploss_leg = 1

#                             # crosses_threshold_long_index_ce = crosses_threshold_long_ce.idxmax()
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[crosses_threshold_long_index_ce + pd.to_timedelta('1T'), 'Open']
#                             # except:
#                             #     print(crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#                             #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (crosses_threshold_long_index_ce + pd.to_timedelta('1T'))].max()
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                             # long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#                             # long_exit_type_ce = 'Long CE Stoploss Exit'
#                             # long_stoploss_ce = 1

#                     else:
#                         long_ce_universal_leg = 1
#                         long_ce_stoploss_leg = 0

#                         # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                         # try:    
#                         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                         # except:
#                         #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open'] 
                        
#                         # long_exit_type_ce = 'Long CE Universal Exit'
#                         # long_stoploss_ce = 0
#             else:
#                 short_ce_universal_leg = 1
#                 short_ce_stoploss_leg = 0

#                 # # Short universal exit
#                 short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:
#                 #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#                 # except:    
#                 #     print(short_exit_time_period_ce)
#                 #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#                 #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
                
#                 # short_exit_type_ce = 'Short CE Universal Exit'
#                 # short_stoploss_ce = 0
#                 # long_exit_type_ce = ''

#                 long_ce_universal_leg = 1
#                 long_ce_stoploss_leg = 0

#                 # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open'] 
                
#                 # long_exit_type_ce = 'Long CE Universal Exit'
#                 # long_stoploss_ce = 0                
# ##########################################################################
#     elif crosses_threshold_short_ce.any() & (not crosses_threshold_short_pe.any()):
#         # print('Code is here - 3')
        
#         if intraday_data_short_ce[intraday_data_short_ce.index > crosses_threshold_short_ce.idxmax()].empty:
            
#             short_ce_universal_leg = 1
#             short_ce_stoploss_leg = 0

#             # # Short universal exit
#             short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#             # try:
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#             # except:    
#             #     print(short_exit_time_period_ce)
#             #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
            
#             # short_exit_type_ce = 'Short CE Universal Exit'
#             # short_stoploss_ce = 0
#             long_exit_type_ce = ''

#         else:
#             short_ce_universal_leg = 0
#             short_ce_stoploss_leg = 1
            
#             crosses_threshold_short_index_ce = crosses_threshold_short_ce.idxmax()
#             # try:
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[crosses_threshold_short_index_ce + pd.to_timedelta('1T'), 'Open']
#             # except:
#             #     print(crosses_threshold_short_index_ce + pd.to_timedelta('1T'))
#             #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < (crosses_threshold_short_index_ce + pd.to_timedelta('1T'))].max()
#             #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']

#             short_exit_time_period_ce = crosses_threshold_short_index_ce + pd.to_timedelta('1T')
#             # short_exit_type_ce = 'Short CE Stoploss Exit'
#             # short_stoploss_ce = 1
#             long_exit_type_ce = ''

#         if  long_exit_type_ce != 'Long CE Stoploss Exit':
#             intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= crosses_threshold_short_ce.idxmax()]
#             crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#             if crosses_threshold_long_ce.any():

#                 # Long universal exit
#                 if intraday_data_long_ce[intraday_data_long_ce.index > crosses_threshold_long_ce.idxmax()].empty:

#                     long_ce_universal_leg = 1
#                     long_ce_stoploss_leg = 0

#                     # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                     # except:    
#                     #     print(long_exit_time_period_ce)
#                     #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                     # long_exit_type_ce = 'Long CE Universal Exit'
#                     # long_stoploss_ce = 0

#                 else:    
#                     long_ce_universal_leg = 0
#                     long_ce_stoploss_leg = 1

#                     # crosses_threshold_long_index_ce = crosses_threshold_long_ce.idxmax()
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[crosses_threshold_long_index_ce + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (crosses_threshold_long_index_ce + pd.to_timedelta('1T'))].max()
#                     #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                     # long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#                     # long_exit_type_ce = 'Long CE Stoploss Exit'
#                     # long_stoploss_ce = 1
#             else:
#                 long_ce_universal_leg = 1
#                 long_ce_stoploss_leg = 0

#                 # long_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price = intraday_data_long_ce.at[long_exit_time_period, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period].max()
#                 #     LONG_STRIKE_exit_price = intraday_data_long_ce.at[next_index, 'Open'] 
                
#                 # long_exit_type_ce = 'Long CE Universal Exit'
#                 # long_stoploss_ce = 0

#             intraday_data_short_pe = intraday_data_short_pe[intraday_data_short_pe.index >= short_exit_time_period_ce]
#             SHORT_STRIKE_new_entry_price_pe = intraday_data_short_pe.iloc[0]['Open']

#             intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= short_exit_time_period_ce]
#             LONG_STRIKE_new_entry_price_pe = intraday_data_long_pe.iloc[0]['Open']

#             short_stoploss_pt_pe = SHORT_STRIKE_new_entry_price_pe + SHORT_STRIKE_new_entry_price_pe * SHORT_SL
#             long_profit_pt_pe = LONG_STRIKE_new_entry_price_pe + LONG_STRIKE_new_entry_price_pe * LONG_PT

#             crosses_threshold_short_pe = (intraday_data_short_pe['Open'] > short_stoploss_pt_pe)
#             crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#             if crosses_threshold_short_pe.any():

#                 if intraday_data_short_pe[intraday_data_short_pe.index > crosses_threshold_short_pe.idxmax()].empty:
                    
#                     short_pe_universal_leg = 1
#                     short_pe_stoploss_leg = 0

#                     # # Short universal exit
#                     short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                     # try:
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#                     # except:    
#                     #     print(short_exit_time_period_pe)
#                     #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
                    
#                     # short_exit_type_pe = 'Short PE Universal Exit'
#                     # short_stoploss_pe = 0
#                     long_exit_type_pe = ''

#                 else:
#                     short_pe_universal_leg = 0
#                     short_pe_stoploss_leg = 1

#                     crosses_threshold_short_index_pe = crosses_threshold_short_pe.idxmax()
#                     # try:
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[crosses_threshold_short_index_pe + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_short_index_pe + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < (crosses_threshold_short_index_pe + pd.to_timedelta('1T'))].max()
#                     #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']

#                     short_exit_time_period_pe = crosses_threshold_short_index_pe + pd.to_timedelta('1T')
#                     # short_exit_type_pe = 'Short PE Stoploss Exit'
#                     # short_stoploss_pe = 1
#                     long_exit_type_pe = ''

#                 if  long_exit_type_pe != 'Long PE Stoploss Exit':
#                     intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= crosses_threshold_short_pe.idxmax()]
#                     crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#                     if crosses_threshold_long_pe.any():
#                         # Long universal exit
#                         if intraday_data_long_pe[intraday_data_long_pe.index > crosses_threshold_long_pe.idxmax()].empty:
                            
#                             long_pe_universal_leg = 1
#                             long_pe_stoploss_leg = 0

#                             # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                             # except:    
#                             #     print(long_exit_time_period_pe)
#                             #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                             # long_exit_type_pe = 'Long PE Universal Exit'
#                             # long_stoploss_pe = 0

#                         else:    
#                             long_pe_universal_leg = 0
#                             long_pe_stoploss_leg = 1

#                             # crosses_threshold_long_index_pe = crosses_threshold_long_pe.idxmax()
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[crosses_threshold_long_index_pe + pd.to_timedelta('1T'), 'Open']
#                             # except:
#                             #     print(crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#                             #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < (crosses_threshold_long_index_pe + pd.to_timedelta('1T'))].max()
#                             #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                             # long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#                             # long_exit_type_pe = 'Long PE Stoploss Exit'
#                             # long_stoploss_pe = 1

#                     else:
#                         long_pe_universal_leg = 1
#                         long_pe_stoploss_leg = 0

#                         # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                         # try:    
#                         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                         # except:
#                         #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                        
#                         # long_exit_type_pe = 'Long PE Universal Exit'
#                         # long_stoploss_pe = 0
#             else:
                
#                 short_pe_universal_leg = 1
#                 short_pe_stoploss_leg = 0                
#                 # # Short universal exit
#                 # short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:
#                 #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#                 # except:    
#                 #     print(short_exit_time_period_pe)
#                 #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#                 #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
                
#                 # short_exit_type_pe = 'Short PE Universal Exit'
#                 # short_stoploss_pe = 0
#                 # long_exit_type_pe = ''

#                 long_pe_universal_leg = 1
#                 long_pe_stoploss_leg = 0

#                 # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                
#                 # long_exit_type_pe = 'Long PE Universal Exit'
#                 # long_stoploss_pe = 0                
# #########################################
#     elif (not crosses_threshold_short_ce.any()) & crosses_threshold_short_pe.any():

#         # print('Code is here - 4')

#         # if short leg exit at EXIT time 
#         if intraday_data_short_pe[intraday_data_short_pe.index > crosses_threshold_short_pe.idxmax()].empty:
            
#             short_pe_universal_leg = 1
#             short_pe_stoploss_leg = 0

#             # # Short universal exit
#             short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#             # try:
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#             # except:    
#             #     print(short_exit_time_period_pe)
#             #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
            
#             # short_exit_type_pe = 'Short PE Universal Exit'
#             # short_stoploss_pe = 0
#             long_exit_type_pe = ''

#         else:
#             short_pe_universal_leg = 0
#             short_pe_stoploss_leg = 1

#             crosses_threshold_short_index_pe = crosses_threshold_short_pe.idxmax()
#             # try:
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[crosses_threshold_short_index_pe + pd.to_timedelta('1T'), 'Open']
#             # except:
#             #     print(crosses_threshold_short_index_pe + pd.to_timedelta('1T'))
#             #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < (crosses_threshold_short_index_pe + pd.to_timedelta('1T'))].max()
#             #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']

#             short_exit_time_period_pe = crosses_threshold_short_index_pe + pd.to_timedelta('1T')
#             # short_exit_type_pe = 'Short PE Stoploss Exit'
#             # short_stoploss_pe = 1
#             long_exit_type_pe = ''

#         if  long_exit_type_pe != 'Long PE Stoploss Exit':
#             intraday_data_long_pe = intraday_data_long_pe[intraday_data_long_pe.index >= crosses_threshold_short_pe.idxmax()]
#             crosses_threshold_long_pe = (intraday_data_long_pe['Open'] > long_profit_pt_pe)

#             if crosses_threshold_long_pe.any():
#                 # print('Code is here - 6')
                
#                 # Long universal exit
#                 if intraday_data_long_pe[intraday_data_long_pe.index > crosses_threshold_long_pe.idxmax()].empty:
                    
#                     long_pe_universal_leg = 1
#                     long_pe_stoploss_leg = 0                    

#                     # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                     # except:    
#                     #     print(long_exit_time_period_pe)
#                     #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                     # long_exit_type_pe = 'Long PE Universal Exit'
#                     # long_stoploss_pe = 0

#                 else:    
#                     long_pe_universal_leg = 0
#                     long_pe_stoploss_leg = 1

#                     # crosses_threshold_long_index_pe = crosses_threshold_long_pe.idxmax()
                    
#                     # try:
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[crosses_threshold_long_index_pe + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < (crosses_threshold_long_index_pe + pd.to_timedelta('1T'))].max()
#                     #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#                     # long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#                     # long_exit_type_pe = 'Long PE Stoploss Exit'
#                     # long_stoploss_pe = 1

#             else:
#                 long_pe_universal_leg = 1
#                 long_pe_stoploss_leg = 0

#                 # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:    
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#                 # except:
#                 #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                 #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
                
#                 # long_exit_type_pe = 'Long PE Universal Exit'
#                 # long_stoploss_pe = 0

#             intraday_data_short_ce = intraday_data_short_ce[intraday_data_short_ce.index >= short_exit_time_period_pe]
#             SHORT_STRIKE_new_entry_price_ce = intraday_data_short_ce.iloc[0]['Open']

#             intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= short_exit_time_period_pe]
#             LONG_STRIKE_new_entry_price_ce = intraday_data_long_ce.iloc[0]['Open']

#             short_stoploss_pt_ce = SHORT_STRIKE_new_entry_price_ce + SHORT_STRIKE_new_entry_price_ce * SHORT_SL
#             long_profit_pt_ce = LONG_STRIKE_new_entry_price_ce + LONG_STRIKE_new_entry_price_ce * LONG_PT

#             crosses_threshold_short_ce = (intraday_data_short_ce['Open'] > short_stoploss_pt_ce)
#             crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#             if crosses_threshold_short_ce.any():

#                 if intraday_data_short_ce[intraday_data_short_ce.index > crosses_threshold_short_ce.idxmax()].empty:
                    
#                     short_ce_universal_leg = 1
#                     short_ce_stoploss_leg = 0

#                     # # Short universal exit
#                     short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                     # try:
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#                     # except:    
#                     #     print(short_exit_time_period_ce)
#                     #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
                    
#                     # short_exit_type_ce = 'Short CE Universal Exit'
#                     # short_stoploss_ce = 0
#                     long_exit_type_ce = ''

#                 else:
#                     short_ce_universal_leg = 0
#                     short_ce_stoploss_leg = 1
                    
#                     crosses_threshold_short_index_ce = crosses_threshold_short_ce.idxmax()
#                     # try:
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[crosses_threshold_short_index_ce + pd.to_timedelta('1T'), 'Open']
#                     # except:
#                     #     print(crosses_threshold_short_index_ce + pd.to_timedelta('1T'))
#                     #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < (crosses_threshold_short_index_ce + pd.to_timedelta('1T'))].max()
#                     #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']

#                     short_exit_time_period_ce = crosses_threshold_short_index_ce + pd.to_timedelta('1T')
#                     # short_exit_type_ce = 'Short CE Stoploss Exit'
#                     # short_stoploss_ce = 1
#                     long_exit_type_ce = ''

#                 if  long_exit_type_ce != 'Long CE Stoploss Exit':
#                     intraday_data_long_ce = intraday_data_long_ce[intraday_data_long_ce.index >= crosses_threshold_short_ce.idxmax()]
#                     crosses_threshold_long_ce = (intraday_data_long_ce['Open'] > long_profit_pt_ce)

#                     if crosses_threshold_long_ce.any():

#                         # Long universal exit
#                         if intraday_data_long_ce[intraday_data_long_ce.index > crosses_threshold_long_ce.idxmax()].empty:
                            
#                             long_ce_universal_leg = 1
#                             long_ce_stoploss_leg = 0

#                             # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                             # except:    
#                             #     print(long_exit_time_period_ce)
#                             #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                             # long_exit_type_ce = 'Long CE Universal Exit'
#                             # long_stoploss_ce = 0

#                         else:    
#                             long_ce_universal_leg = 0
#                             long_ce_stoploss_leg = 1

#                             # crosses_threshold_long_index_ce = crosses_threshold_long_ce.idxmax()
                            
#                             # try:
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[crosses_threshold_long_index_ce + pd.to_timedelta('1T'), 'Open']
#                             # except:
#                             #     print(crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#                             #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (crosses_threshold_long_index_ce + pd.to_timedelta('1T'))].max()
#                             #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                             # long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#                             # long_exit_type_ce = 'Long CE Stoploss Exit'
#                             # long_stoploss_ce = 1

#                     else:
#                         long_ce_universal_leg = 1
#                         long_ce_stoploss_leg = 0

#                         # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                         # try:    
#                         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                         # except:
#                         #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open'] 
                        
#                         # long_exit_type_ce = 'Long CE Universal Exit'
#                         # long_stoploss_ce = 0
#             else:
                
#                 short_ce_universal_leg = 1
#                 short_ce_stoploss_leg = 0

#                 # # Short universal exit
#                 # short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#                 # try:
#                 #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#                 # except:    
#                 #     print(short_exit_time_period_ce)
#                 #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#                 #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
                
#                 # short_exit_type_ce = 'Short CE Universal Exit'
#                 # short_stoploss_ce = 0
#                 # long_exit_type_ce = ''

#                 long_ce_universal_leg = 1
#                 long_ce_stoploss_leg = 0

#                 # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                            
#                 # try:
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#                 # except:    
#                 #     print(long_exit_time_period_ce)
#                 #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#                 #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#                 # long_exit_type_ce = 'Long CE Universal Exit'
#                 # long_stoploss_ce = 0
# #################################################
#     else:
#         short_ce_universal_leg = 1
#         short_ce_stoploss_leg = 0

#         # # short ce exit
#         # short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         # try:
#         #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#         # except:    
#         #     print(short_exit_time_period_ce)
#         #     next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#         #     SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
        
#         # short_exit_type_ce = 'Short CE Universal Exit'
#         # short_stoploss_ce = 0
#         # long_exit_type_ce = ''

#         short_pe_universal_leg = 1
#         short_pe_stoploss_leg = 0

#         # # short pe exit
#         # short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         # try:
#         #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#         # except:    
#         #     print(short_exit_time_period_pe)
#         #     next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#         #     SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
        
#         # short_exit_type_pe = 'Short PE Universal Exit'
#         # short_stoploss_pe = 0
#         # long_exit_type_pe = ''

#         long_ce_universal_leg = 1
#         long_ce_stoploss_leg = 0

#         # # long ce exit
#         # long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         # try:
#         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
        
#         # except:
#         #     next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#         #     LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']
        
#         # long_exit_type_ce = 'Long CE Universal Exit'
#         # long_stoploss_ce = 0

#         long_pe_universal_leg = 1
#         long_pe_stoploss_leg = 0

#         # # long pe exit
#         # long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         # try:    
#         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#         # except:
#         #     next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#         #     LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
        
#         # long_exit_type_pe = 'Long PE Universal Exit'
#         # long_stoploss_pe = 0

#     # CE Exits
#     if short_ce_universal_leg:

#         # short ce universal exit
#         short_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         try:
#             SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[short_exit_time_period_ce, 'Open']

#         except:    
#             print(SHORT_STRIKE_CE, short_exit_time_period_ce)
#             next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < short_exit_time_period_ce].max()
#             SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']
        
#         short_exit_type_ce = 'Short CE Universal Exit'
#         short_stoploss_ce = 0
#         long_exit_type_ce = ''        

#     elif short_ce_stoploss_leg:
#         crosses_threshold_short_index_ce = crosses_threshold_short_ce.idxmax()
#         try:
#             SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[crosses_threshold_short_index_ce + pd.to_timedelta('1T'), 'Open']
#         except:
#             print(SHORT_STRIKE_CE, crosses_threshold_short_index_ce + pd.to_timedelta('1T'))
#             next_index = intraday_data_short_ce.index[intraday_data_short_ce.index < (crosses_threshold_short_index_ce + pd.to_timedelta('1T'))].max()
#             SHORT_STRIKE_exit_price_ce = intraday_data_short_ce.at[next_index, 'Open']

#         short_exit_time_period_ce = crosses_threshold_short_index_ce + pd.to_timedelta('1T')
#         short_exit_type_ce = 'Short CE Stoploss Exit'
#         short_stoploss_ce = 1
#         long_exit_type_ce = ''        

#     if long_ce_universal_leg:

#         long_exit_time_period_ce = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        
#         try:
#             LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#         except:    
#             print(LONG_STRIKE_CE, long_exit_time_period_ce)
#             next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < long_exit_time_period_ce].max()
#             LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#         long_exit_type_ce = 'Long CE Universal Exit'
#         long_stoploss_ce = 0        

#     elif long_ce_stoploss_leg:

#         crosses_threshold_long_index_ce = crosses_threshold_long_ce.idxmax()
        
#         try:
#             LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[crosses_threshold_long_index_ce + pd.to_timedelta('1T'), 'Open']
#         except:
#             print(LONG_STRIKE_CE, crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#             next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (crosses_threshold_long_index_ce + pd.to_timedelta('1T'))].max()
#             LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#         long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#         long_exit_type_ce = 'Long CE Stoploss Exit'
#         long_stoploss_ce = 1        

#     # PE Exits
#     if short_pe_universal_leg:

#         short_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         try:
#             SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[short_exit_time_period_pe, 'Open']

#         except:    
#             print(SHORT_STRIKE_PE, short_exit_time_period_pe)
#             next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < short_exit_time_period_pe].max()
#             SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']
        
#         short_exit_type_pe = 'Short PE Universal Exit'
#         short_stoploss_pe = 0
#         long_exit_type_pe = '' 

#     elif short_pe_stoploss_leg:
#         crosses_threshold_short_index_pe = crosses_threshold_short_pe.idxmax()
#         try:
#             SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[crosses_threshold_short_index_pe + pd.to_timedelta('1T'), 'Open']
#         except:
#             print(SHORT_STRIKE_PE, crosses_threshold_short_index_pe + pd.to_timedelta('1T'))
#             next_index = intraday_data_short_pe.index[intraday_data_short_pe.index < (crosses_threshold_short_index_pe + pd.to_timedelta('1T'))].max()
#             SHORT_STRIKE_exit_price_pe = intraday_data_short_pe.at[next_index, 'Open']

#         short_exit_time_period_pe = crosses_threshold_short_index_pe + pd.to_timedelta('1T')
#         short_exit_type_pe = 'Short PE Stoploss Exit'
#         short_stoploss_pe = 1

#     if long_pe_universal_leg:

#         long_exit_time_period_pe = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
#         try:    
#             LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#         except:
#             print(LONG_STRIKE_PE, long_exit_time_period_pe)
#             next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#             LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open'] 
        
#         long_exit_type_pe = 'Long PE Universal Exit'
#         long_stoploss_pe = 0
       
#     elif long_pe_stoploss_leg:
#         crosses_threshold_long_index_pe = crosses_threshold_long_pe.idxmax()
        
#         try:
#             LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[crosses_threshold_long_index_pe + pd.to_timedelta('1T'), 'Open']
#         except:
#             print(LONG_STRIKE_PE, crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#             next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < (crosses_threshold_long_index_pe + pd.to_timedelta('1T'))].max()
#             LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#         long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#         long_exit_type_pe = 'Long PE Stoploss Exit'
#         long_stoploss_pe = 1        

#     if (short_ce_stoploss_leg & short_pe_stoploss_leg):
#         if long_exit_time_period_ce > max(short_exit_time_period_ce, short_exit_time_period_pe):

#             long_exit_time_period_ce = max(short_exit_time_period_ce, short_exit_time_period_pe)  
#             try:
#                 LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[long_exit_time_period_ce, 'Open']
#             except:
#                 print(LONG_STRIKE_CE, crosses_threshold_long_index_ce + pd.to_timedelta('1T'))
#                 next_index = intraday_data_long_ce.index[intraday_data_long_ce.index < (long_exit_time_period_ce)].max()
#                 LONG_STRIKE_exit_price_ce = intraday_data_long_ce.at[next_index, 'Open']

#             # long_exit_time_period_ce = crosses_threshold_long_index_ce + pd.to_timedelta('1T')
#             long_exit_type_ce = ''
#             long_stoploss_ce = 0
        
#         if long_exit_time_period_pe > max(short_exit_time_period_ce, short_exit_time_period_pe):
            
#             long_exit_time_period_pe = max(short_exit_time_period_ce, short_exit_time_period_pe)  

#             try:
#                 LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[long_exit_time_period_pe, 'Open']
#             except:
#                 print(LONG_STRIKE_PE, crosses_threshold_long_index_pe + pd.to_timedelta('1T'))
#                 next_index = intraday_data_long_pe.index[intraday_data_long_pe.index < long_exit_time_period_pe].max()
#                 LONG_STRIKE_exit_price_pe = intraday_data_long_pe.at[next_index, 'Open']

#             # long_exit_time_period_pe = crosses_threshold_long_index_pe + pd.to_timedelta('1T')
#             long_exit_type_pe = ''
#             long_stoploss_pe = 0                         

#     del intraday_data_short_ce
#     del intraday_data_long_ce
#     del intraday_data_short_pe
#     del intraday_data_long_pe

#     ce_results = [SHORT_STRIKE_entry_price_ce, SHORT_STRIKE_exit_price_ce, short_exit_time_period_ce, LONG_STRIKE_entry_price_ce, LONG_STRIKE_exit_price_ce, long_exit_time_period_ce, short_stoploss_ce, long_stoploss_ce, short_exit_type_ce, long_exit_type_ce]
#     pe_results = [SHORT_STRIKE_entry_price_pe, SHORT_STRIKE_exit_price_pe, short_exit_time_period_pe, LONG_STRIKE_entry_price_pe, LONG_STRIKE_exit_price_pe, long_exit_time_period_pe, short_stoploss_pe, long_stoploss_pe, short_exit_type_pe, long_exit_type_pe]

#     # print(short_exit_type)
#     return ce_results, pe_results








def parameter_process(parameter, mapped_days,vix_df, option_data, df, filter_df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT = parameter
    
    resampled_df = resample_data(df,TIME_FRAME) 
    resampled = resampled_df.dropna() 
    return trade_sheet_creator(mapped_days, vix_df ,option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT)

########################################### INPUTS #####################################################
# Inputs



superset = 'Vix_strategy'
stock = 'SENSEX'
option_type = 'ND'
dte_list = [1,2,3,6,7]


roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' else (50 if stock == 'FINNIFTY' else (100 if stock == 'SENSEX' else None)))
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' else (3 if stock == 'FINNIFTY' else (3 if stock == 'SENSEX' else None)))
LOT_SIZE = 75 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (25 if stock == 'FINNIFTY' else (20 if stock == 'SENSEX' else None)))


# Define all the file paths
root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{option_type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"


# expiry_file_path = rf"/home/newberry4/jay_data/nifty_2nd_week_expiry.xlsx"
# expiry_file_path = rf"/home/newberry4/jay_data/nifty_3rd_week_expiry.xlsx"
txt_file_path = rf'{root_path}/new_done.txt'
output_folder_path = rf'{root_path}/Trade_Sheets/'


if stock == 'NIFTY':        
    expiry_file_path = rf"/home/newberry4/jay_data/NIFTY Market Dates updated 2025.xlsx"
else:       
    expiry_file_path = rf"/home/newberry4/jay_data/SENSEX market dates updated 2025 new.xlsx"  # expiry_file_path = rf"/home/newberry4/jay_data/nifty_2nd_week_expiry.xlsx"       # for finnifty
# expiry_file_path = "/home/newberry4/jay_data/BANKNIFTY market dates (1).xlsx"   # for banknifty




if stock == 'NIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/monthly_expiry/"
    # option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/Current_Expiry_OI_OHLC/"     #### pickle file
    # option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/current_expiry_paraquette/"                  ####paraquette file
    option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/ohlc_with_all_strikes/"
elif stock =='BANKNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/BANKNIFTY_DATA/BANKNIFTY_OHLCV/"
    option_data_path = rf"/home/newberry4/jay_data/Data/BANKNIFTY/monthly_expiry_OI/"
elif stock =='FINNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/FINNIFTY_2/"
    option_data_path = rf"/home/newberry4/jay_data/Data/FINNIFTY/monthly_expiry/"
elif stock =='SENSEX':
    # option_data_path = rf"jay_data/Data/SENSEX/weekly_expiry/"
    # option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OHLC/"
    # option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/weekly_expiry_OI_OHLC2/"
    option_data_path = rf"/home/newberry4/jay_data/Data/SENSEX/ohlc_with_all_strikes2/"



# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None


# list of period buckets   
# date_ranges = [('2024-06-01', '2024-06-10')]

if stock == 'NIFTY':
    date_ranges = [
                    ('2024-12-01', '2025-06-30'),
                    ('2024-06-01', '2024-11-30'),
                    ('2023-12-01', '2024-05-31'),
                    ('2023-06-01', '2023-11-30'),
                    ('2022-12-01', '2023-05-31'),
                    ('2022-06-01', '2022-11-30'),
                    ('2021-12-01', '2022-05-31'),  
                    ('2021-06-01', '2021-11-30')
                    ]
    
# elif stock == 'SENSEX':
#     date_ranges = [
#                     ('2024-12-01', '2025-06-30'),
#                     ('2024-06-01', '2024-11-30'),
#                     ('2023-12-01', '2024-05-31'),
#                     ('2023-08-01', '2023-11-30'),
#                     ]

elif stock == 'SENSEX':
#    date_ranges = [('2023-09-25', '2023-09-25')]

   date_ranges = [
    ('2025-07-01', '2025-08-31'),   
    ('2025-05-01', '2025-06-30'),

    ('2025-04-01', '2025-04-30'),
    ('2025-02-01', '2025-03-31'),

    ('2025-01-01', '2025-01-31'),
    ('2024-11-01', '2024-12-31'),

    ('2024-10-01', '2024-10-31'),
    ('2024-08-01', '2024-09-30'),

    ('2024-07-01', '2024-07-31'),
    # ('2024-05-01', '2024-06-30'),

    # ('2024-04-01', '2024-04-30'),
    # ('2024-02-01', '2024-03-31'),

    # ('2024-01-01', '2024-01-31'),
    # ('2023-11-01', '2023-12-31'),

    # ('2023-10-01', '2023-10-31'),
    # ('2023-08-01', '2023-09-30'),
    ]


# Testing Combinations
candle_time_frame = ['5T']
entries = ['09:30','14:30']
# entries = ['09:30']
exits = ['15:00']
short_strikes = [0]
if stock == 'SENSEX' :
    # long_strikes = [1000] 
    long_strikes = [0.33,0.5,1000 ,500]   ## 1000 for sensex and 500 for nifty , less than 1 are percentages and more than 1 are abs values
elif stock =='NIFTY':
    long_strikes = [0.33 , 0.5 , 500]
short_stoploss_per = [2, 2.5 ,3]
# short_stoploss_per = [2.5]
long_profit_per = [0.3,0.2,0.1]
# long_profit_per = [0.3]
premium_threshold = ['1st_week']



if stock=='NIFTY':
    allowed_short_sl = {
        '09:30': [2, 3 ,2.5],
        '14:30': [2, 3 , 2.5],
        # '09:45': [0.2, 0.3, 0.4, 0.5, 0.6],
        # '10:15': [0.6, 0.7, 0.8, 0.9],
        # '10:45': [0.3, 0.4, 0.6, 0.7],
        # '11:30': [0.4, 0.5, 0.6, 0.7, 0.8],
        # '12:00': [0.6, 0.7, 0.8, 0.9],
        # '12:30': [0.6, 0.7, 0.8, 0.9],
        # '13:00': [0.2, 0.3, 0.4, 0.5, 0.6],
        # '13:30': [0.4, 0.5, 0.6, 0.8, 0.9],
        # '14:00': [0.2, 0.3, 0.4, 0.7, 0.8],       
    }

if stock=='SENSEX':
    allowed_short_sl = {
        '09:30': [2, 3 ,2.5],
        '14:30': [2, 3 , 2.5],
    }



if stock=='FINNIFTY':
    allowed_short_sl = {
        '09:30': [0.3, 0.4, 0.5, 0.6],
        '09:45': [0.2, 0.3, 0.4, 0.5, 0.6],
        '10:15': [0.3, 0.5, 0.6, 0.7],
        '10:45': [0.3, 0.4, 0.6, 0.7],
        '11:30': [0.3, 0.4, 0.7, 0.8],
        '12:00': [0.6, 0.7, 0.8, 0.9],
        '12:30': [0.6, 0.7, 0.8, 0.9],
        '13:00': [0.2, 0.3, 0.4, 0.5, 0.6],
        '13:30': [0.3, 0.4, 0.5, 0.6, 0.7],
        '14:00': [0.3, 0.4, 0.8, 0.9],
    }
    
if stock=='BANKNIFTY':
    allowed_short_sl = {
        '09:30': ['NA', 0.3, 0.4, 0.5],
        '09:45': ['NA', 0.3, 0.4, 0.5, 0.6],
        '10:15': ['NA', 0.5, 0.6, 0.7, 0.9],
        '10:45': ['NA', 0.3, 0.4, 0.5, 0.6],
        '11:30': ['NA', 0.3, 0.4, 0.5, 0.6],
        '12:00': ['NA', 0.4, 0.5, 0.6, 0.7],
        '12:30': ['NA', 0.3, 0.4, 0.6, 0.7],
        '13:00': ['NA', 0.3, 0.4, 0.9],
        '13:30': ['NA', 0.3, 0.5, 0.6],
        '14:00': ['NA', 0.3, 0.4, 0.6, 0.7],
    }

parameters = []

# if stock == 'BANKNIFTY' or stock == 'SENSEX':
#     short_strikes = [x * 2 for x in short_strikes]
#     long_strikes = [x * 2 for x in long_strikes]

excel_results = []

# if stock == 'FINNIFTY':
#     pass
if __name__ == "__main__":
    
    counter = 0
    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]
    # end_date_idx = '2024-10-25'

    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[
    (mapped_days['Date'] >= start_date_idx) & 
    (mapped_days['Date'] <= end_date_idx) 
]
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry' : 'DaysToExpiry'})
    mapped_days = mapped_days[
    (mapped_days['Date'] > '2021-06-03') &
    (mapped_days['Date'] != '2024-05-18') &
    (mapped_days['Date'] != '2024-05-20') ]
    mapped_days = mapped_days[
    ~((mapped_days['Date'] >= '2024-05-30') & (mapped_days['Date'] <= '2024-06-06'))
]
    # mapped_days = mapped_days[mapped_days['Date'] != pd.Timestamp('2024-05-18')]
    weekdays = mapped_days['Date'].to_list()

    # Pull Index Data
    df = pull_index_data(start_date_idx, end_date_idx, stock)
    # df = df[df['Date'] != pd.Timestamp('2024-05-18')]
    # resampled_df_main = resample_data(df, '5T')
    vix_df = pd.read_csv("/home/newberry4/jay_test/Vix_strategy/_INDIAVIX__202510151515.csv")
    # option_data = pull_options_data_d(start_date_idx, end_date_idx, option_data_path, stock)
    # filtered_data = option_data.loc[
    # (option_data['Date'] == '2021-06-11') & (option_data['ExpiryDate'] == '2021-06-17')
    # ]

    # start_date1 = datetime.strptime('2021-06-07', '%Y-%m-%d').date()
    # end_date1 = datetime.strptime('2021-06-17', '%Y-%m-%d').date()

# Filter data within the date range and ExpiryDate condition
    # daily_option_data = option_data[
    #     (option_data.index.date >= start_date1) &
    #     (option_data.index.date <= end_date1) &
    #     (option_data['ExpiryDate'] == '2021-06-17')
    # ]

# Print or further process daily_option_data as needed

    # Sort the filtered data by index
    # sorted_data = daily_option_data.sort_index()

    # Print the sorted data
    # print(sorted_data)

    for start_date, end_date in date_ranges: 
        
        counter += 1
        print(start_date, end_date, counter)

        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
        # Extend end_date by 60 days
        extended_end_date = end_date_dt + timedelta(days=15)
        
        # Convert extended_end_date back to string if needed
        extended_end_date_str = extended_end_date.strftime("%Y-%m-%d")
        
        # Pull Options data with the extended end_date
        option_data = pull_options_data_d(start_date, extended_end_date_str, option_data_path, stock)
         

        mapped_dates_in_range = mapped_days[
            (mapped_days['Date'] >= start_date) & (mapped_days['Date'] <= end_date)
        ]['Date'].unique()

        option_dates = option_data['Date'].unique()

        missing_dates = [d for d in mapped_dates_in_range if d not in option_dates]


        row = {
            "Start_Date": start_date,
            "End_Date": end_date,
            "Mapped_Dates": ", ".join(str(d) for d in mapped_dates_in_range),
            "Option_Data_Dates": ", ".join(str(d) for d in option_dates),
            "Missing_Dates": ", ".join(str(d) for d in missing_dates) if missing_dates else "None"
        }

        excel_results.append(row)

        results_df = pd.DataFrame(excel_results)

        results_df.to_csv(rf"{root_path}/expiry_date_check.csv", index=False)

        print("✅ Expiry date check saved to expiry_date_check.csv")

        # Pull Options data
        # option_data = pull_options_data_d(start_date, end_date, option_data_path, stock)


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
        
        elif counter==1:
            for TIME_FRAME in candle_time_frame:
                for SHORT_STRIKE in short_strikes:
                    for LONG_STRIKE in long_strikes:
                        for ENTRY in entries:
                            if ENTRY in allowed_short_sl: 
                                for EXIT in exits:
                                    if ENTRY < EXIT :
                                        for PREMIUM_TH in premium_threshold:
                                            for SHORT_SL in short_stoploss_per:
                                                if SHORT_SL in allowed_short_sl[ENTRY] :
                                                    for LONG_PT in long_profit_per: 
                                                        if (SHORT_SL=='NA') & (LONG_PT=='NA'):
                                                            parameters.append([TIME_FRAME,  SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT])
                                                        elif (SHORT_SL=='NA') | (LONG_PT=='NA'):
                                                            continue
                                                        elif SHORT_SL > LONG_PT:   
                                                            parameters.append([TIME_FRAME, SHORT_STRIKE, LONG_STRIKE, ENTRY, EXIT, PREMIUM_TH, SHORT_SL, LONG_PT])


        # Read the content of the log file to check which parameters have already been processed
        print('Total parameters :', len(parameters))
        file_path = txt_file_path 
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]

        print("existing_values",existing_values)

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if (stock + '_candle_' + str(value[0])  + '_short_' + str(value[1]).replace('.', ',')  + '_long_' + str(value[2]).replace('.', ',') + '_entry_' + str(value[3]).replace(':', ',') +  '_exit_' + str(value[4]).replace(':', ',') + '_premium_' + str(value[5]) + '_short_sl_' + str(value[6]).replace('.', ',') + '_short_pt_' + str(value[7]).replace('.', ',') + '_' + start_date + '_' + end_date) not in existing_values]
        # print("parameters",parameters)
        for value in parameters:
            param_str = (
                stock + '_candle_' + str(value[0])  + '_short_' + str(value[1])  + '_long_' + str(value[2]) + '_entry_' + str(value[3]).replace(':', ',') +  '_exit_' + str(value[4]).replace(':', ',') + '_premium_' + str(value[5]) + '_short_sl_' + str(value[6]).replace('.', ',') + '_short_pt_' + str(value[7]).replace('.', ',') + '_' + start_date + '_' + end_date)
            print(param_str)
        print('Parameters to run :', len(parameters))
   
        start_time = time.time()
        num_processes = 19
        print('No. of processes :', num_processes)

        partial_process = partial(parameter_process,   mapped_days=mapped_days,vix_df =vix_df, option_data=option_data, df=df, filter_df=filter_df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path)
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
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Time taken to get Initial Tradesheets:', elapsed_time)
print('Finished at :', time.time())

















































