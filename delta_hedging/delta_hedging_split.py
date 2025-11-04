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
from datetime import datetime, timedelta
import ast, json, sys, re

from helpers import find_hedge_strike, is_expiry, pull_options_data_d, pull_index_data, resample_data , find_hedge_strike2

# TODO: rerun deleting from 30th may to 6th june


def calculate_profit(premium, strike, entry_time, df, TARGET, option_type, expiry_date, entry_date, max_exit_datetime):
    # 🛑 Handle case where premium is None
    if premium is None:
        print(f"⚠️ Warning: No premium data found for {option_type}, Strike: {strike}, Expiry: {expiry_date}, Date: {entry_date}")
        return None, None

    # ✅ Combine entry_date and entry_time into a full timestamp
    # start_datetime = pd.to_datetime(f"{entry_date} {entry_time}")

    # ✅ Filter dataframe based on time range, strike and type
    df = df[(df.index >= entry_time) & (df.index <= max_exit_datetime)]
    df = df[(df['StrikePrice'] == strike) & (df['Type'] == option_type)]

    # ✅ If DataFrame is empty after filtering
    if df.empty:
        print(f"⚠️ No data available for {option_type}, Strike: {strike}, between {entry_time} and {max_exit_datetime}")
        return None, None

    # ✅ Sort by datetime to ensure proper order
    df = df.sort_index()

    # ✅ Compute the target exit price
    theoretical_exit_premium = premium * (1 - TARGET)

    # ✅ Iterate through the DataFrame to find first instance where 'Open' breaches target
    for ts, row in df.iterrows():
        if row['Open'] < theoretical_exit_premium:
            exit_premium = row['Open']
            exit_time = ts
            print(f"🎯 Target hit: Exit Premium = {exit_premium}, Exit Time = {exit_time}")
            return exit_premium, exit_time

    # ❌ If target not hit, return the last candle's close and time
    exit_premium = df['Open'].iloc[-1]
    exit_time = df.index[-1]
    print(f"❌ Target not hit: Final Exit Premium = {exit_premium}, Exit Time = {exit_time}")
    return exit_premium, exit_time



# index_data = pull_index_data(start_date, end_date, stock,mapped_days)
# option_data = pull_options_data_d(start_date, end_date, option_data_path, stock)        



def process_profit_point(mapped_days ,option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, HEDGE, TARGET):
    
    target_list =  [TIME_FRAME, HEDGE, TARGET]

    mapped_days_temp = mapped_days

    mapped_days_temp = mapped_days_temp[(mapped_days_temp['Date'] >= start_date) & 
                                    (mapped_days_temp['Date'] <= end_date)]
    mapped_days_temp = mapped_days_temp[(mapped_days_temp['Date'] > '2021-06-03') & 
                                    (mapped_days_temp['Date'] != '2024-05-18') & (mapped_days_temp['Date'] != '2024-05-20') ]
    # print(resampled)
    trades = []
    for day in mapped_days_temp['Date']: 
        print("day", day)               
        # df = df[df['Date'] == day]
        # resampled = resampled[resampled['Date'] == pd.to_datetime(day).date()]\
        start_time = pd.to_datetime(f'{day} 09:15:00')
        end_time = pd.to_datetime(f'{day} 15:30:00')
        
        daily_data = resampled[(resampled.index >= start_time) & (resampled.index <= end_time)].sort_index()
        # print(f"resampled_df for day{day} is ", daily_data)
        daily_option_data = option_data[option_data['Date'] == day].sort_index()     
        if daily_option_data.empty:
            print(f"Option data empty for: {day}")
            continue
        # resampled_df_main = resample_data(daily_index_data, '15min')
        if TIME_FRAME == '15T':
            current_time = pd.to_datetime(f"{day} 09:30:00")
        elif TIME_FRAME == '10T':
            current_time = pd.to_datetime(f"{day} 09:25:00")
        first_time = current_time

        # print("day", first_time)
        # print("first_time", first_time)
        expiry_date = daily_option_data['ExpiryDate'].values[0]
        atm_price = daily_data.loc[daily_data['Time'] == current_time.time(), 'Open'].values[0]
        atm_strike = round(atm_price / 50) * 50 
        atm_price = round(atm_price/50) * 50 
        entry_time = '09:30' if TIME_FRAME == '15T' else '09:25'
        exit_time = '15:25:00'  # Fixed exit time
        max_exit_datetime = pd.to_datetime(f"{expiry_date} {exit_time}")

        # Filter data between entry and exit times, then take the first available value
        filtered_data_call = daily_option_data[
            (daily_option_data['StrikePrice'] == atm_strike) & 
            (daily_option_data['Type'] == 'CE') & 
            (daily_option_data['Time'] >= entry_time) & 
            (daily_option_data['Time'] <= exit_time)
        ]['Open'].sort_index() # Take the first available value

        if not filtered_data_call.empty:
            atm_call_premium = filtered_data_call.iloc[0]  # Safe access
        else:
            print(f"Warning: No option data for {entry_time}, {atm_strike} CE")
            atm_call_premium = None 

        filtered_data_put = daily_option_data[
            (daily_option_data['StrikePrice'] == atm_strike) & 
            (daily_option_data['Type'] == 'PE') & 
            (daily_option_data['Time'] >= entry_time) & 
            (daily_option_data['Time'] <= exit_time)
        ]['Open'].sort_index()

        if not filtered_data_put.empty:
            atm_put_premium = filtered_data_put.iloc[0]  # Safe access
            # print(atm_put_premium)
            
        else:
            print(f"Warning: No option data for {entry_time}, {atm_strike} PE") 
            atm_put_premium = None 

        atm_put_exit_premium, put_exit_time = calculate_profit(atm_put_premium, atm_strike, first_time, option_data, TARGET, 'PE', expiry_date, day,max_exit_datetime)

        atm_call_exit_premium, call_exit_time = calculate_profit(atm_call_premium, atm_strike, first_time, option_data, TARGET, 'CE', expiry_date, day , max_exit_datetime)

        if pd.Timestamp(day).dayofweek != 4:    ## not friday
            check_time = '09:25:00' if TIME_FRAME == '10T' else '09:30:00'  # Adjust time based on TIME_FRAME
            while first_time.time() == pd.to_datetime(check_time).time():
                next_time = first_time
                new_price = daily_data.loc[daily_data['Time'] == next_time.time(), 'Open'].values[0]   
                diff = abs(new_price - current_price)
                delta = int((diff // 50) * 50)
                if delta in [50]:
                    quantity = -10 * (delta // 50)               
                    if new_price > current_price: 
                        # put_strike = round((new_price - 25) / 50) * 50    
                        put_strike = current_price + delta  
                        filtered_data_put = daily_option_data[
                            (daily_option_data['StrikePrice'] == put_strike) & 
                            (daily_option_data['Type'] == 'PE') & 
                            (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                            (daily_option_data['Time'] <= exit_time)
                        ]['Open'].sort_index()

                        if not filtered_data_put.empty:
                            put_premium = filtered_data_put.iloc[0]  # Safe access
                        else:
                            print(f"Warning: No option data for {entry_time}, {atm_strike} PE") 
                            put_premium = None 

                        # put_exit_premium = option_data[(option_data['StrikePrice'] == put_strike) & (option_data['Type'] == 'PE') & (option_data['ExpiryDate'] == expiry_date) & (option_data['Date'] == expiry_date)]['Open'].iloc[-1]
                        put_exit_premium, put_exit_time = calculate_profit(put_premium, put_strike, next_time, option_data,TARGET, 'PE', expiry_date, day , max_exit_datetime)
  
                        trades.append({'Date':day, 'Entry Time': next_time,'Type': 'Sell', 'Strike': put_strike, 'Option': 'PE', 'Entry Premium': put_premium, 'Quantity': quantity,'Exit Premium': put_exit_premium, 'Exit Time': put_exit_time, 'Index': new_price})
                        current_price = put_strike

                    elif new_price < current_price :
                        # call_strike = round((new_price + 25) / 50) * 50   
                        call_strike = current_price - delta      
                        filtered_data_call = daily_option_data[
                            (daily_option_data['StrikePrice'] == call_strike) & 
                            (daily_option_data['Type'] == 'CE') & 
                            (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                            (daily_option_data['Time'] <= exit_time)
                        ]['Open'].sort_index()

                        if not filtered_data_call.empty:
                            call_premium = filtered_data_call.iloc[0]  # Safe access
                        else:
                            print(f"Warning: No option data for {entry_time}, {atm_strike} CE")
                            call_premium = None 
                        # call_exit_premium = option_data[(option_data['StrikePrice'] == call_strike) & (option_data['Type'] == 'CE') & (option_data['ExpiryDate'] == expiry_date) & (option_data['Date'] == expiry_date)]['Open'].iloc[-1]  
                        call_exit_premium, call_exit_time = calculate_profit(call_premium, call_strike, next_time, option_data, TARGET, 'CE', expiry_date, day,max_exit_datetime)
    
                        trades.append({'Date':day, 'Entry Time': next_time, 'Type': 'Sell', 'Strike': call_strike, 'Option': 'CE', 'Entry Premium': call_premium, 'Quantity': quantity,'Exit Premium': call_exit_premium, 'Exit Time': call_exit_time,'Index': new_price })
                        current_price = call_strike   
                    increment = 15 if TIME_FRAME == '15T' else 10
                    first_time = next_time + pd.Timedelta(minutes=increment)

                elif delta >= 100:
                    quantity = -10
                    if new_price > current_price:
                        for step in range(50, delta + 1, 50):
                            put_strike = current_price + step
                            filtered_data_put = daily_option_data[
                                (daily_option_data['StrikePrice'] == put_strike) & 
                                (daily_option_data['Type'] == 'PE') & 
                                (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                                (daily_option_data['Time'] <= exit_time)
                            ]['Open'].sort_index()

                            if not filtered_data_put.empty:
                                put_premium = filtered_data_put.iloc[0]  # Safe access
                            else:
                                print(f"Warning: No option data for {entry_time}, {atm_strike} PE") 
                                put_premium = None 

                            # put_exit_premium = option_data[(option_data['StrikePrice'] == put_strike) & (option_data['Type'] == 'PE') & (option_data['ExpiryDate'] == expiry_date) & (option_data['Date'] == expiry_date)]['Open'].iloc[-1]
                            put_exit_premium, put_exit_time = calculate_profit(put_premium, put_strike, next_time, option_data,TARGET, 'PE', expiry_date, day , max_exit_datetime)
    
                            trades.append({'Date':day, 'Entry Time': next_time,'Type': 'Sell', 'Strike': put_strike, 'Option': 'PE', 'Entry Premium': put_premium, 'Quantity': quantity,'Exit Premium': put_exit_premium, 'Exit Time': put_exit_time, 'Index': new_price})
                            
                            put_strike -= 50  # Move to the next lower strike

                        current_price = put_strike + 50

                    elif new_price < current_price:
                        for step in range(50, delta + 1, 50):
                            call_strike = current_price - step
                            filtered_data_call = daily_option_data[
                                (daily_option_data['StrikePrice'] == call_strike) & 
                                (daily_option_data['Type'] == 'CE') & 
                                (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                                (daily_option_data['Time'] <= exit_time)
                            ]['Open'].sort_index()

                            if not filtered_data_call.empty:
                                call_premium = filtered_data_call.iloc[0]  # Safe access
                            else:
                                print(f"Warning: No option data for {entry_time}, {atm_strike} CE")
                                call_premium = None 
                            call_exit_premium, call_exit_time = calculate_profit(call_premium, call_strike, next_time, option_data, TARGET, 'CE', expiry_date, day,max_exit_datetime)
                            
                            trades.append({'Date':day, 'Entry Time': next_time, 'Type': 'Sell', 'Strike': call_strike, 'Option': 'CE', 'Entry Premium': call_premium, 'Quantity': quantity,'Exit Premium': call_exit_premium, 'Exit Time': call_exit_time,'Index': new_price })
                            
                            call_strike += 50 

                        current_price = call_strike - 50     

                    else:            
                        new_price = current_price     

                    increment = 15 if TIME_FRAME == '15T' else 10
                    first_time = next_time + pd.Timedelta(minutes=increment)

                else:
                    increment = 15 if TIME_FRAME == '15T' else 10
                    first_time = next_time + pd.Timedelta(minutes=increment)

                    
        elif pd.Timestamp(day).dayofweek != 4 and (day - pd.Timedelta(days=3)) not in mapped_days['Date']:
            trades.append({'Date':day, 'Entry Time': current_time,'Type': 'Sell', 'Strike': atm_strike, 'Option': 'CE', 'Entry Premium': atm_call_premium, 'Quantity': -10, 'Exit Premium': atm_call_exit_premium, 'Exit Time': call_exit_time, 'Index': atm_price})
            trades.append({'Date':day, 'Entry Time': current_time,'Type': 'Sell', 'Strike': atm_strike, 'Option': 'PE', 'Entry Premium': atm_put_premium,  'Quantity': -10,'Exit Premium': atm_put_exit_premium, 'Exit Time' : put_exit_time, 'Index': atm_price})
            current_price = atm_price
        else:  
            trades.append({'Date':day, 'Entry Time': current_time,'Type': 'Sell', 'Strike': atm_strike, 'Option': 'CE', 'Entry Premium': atm_call_premium, 'Quantity': -10, 'Exit Premium': atm_call_exit_premium, 'Exit Time' : call_exit_time, 'Index': atm_price})
            trades.append({'Date':day, 'Entry Time': current_time,'Type': 'Sell', 'Strike': atm_strike, 'Option': 'PE', 'Entry Premium': atm_put_premium,  'Quantity': -10,'Exit Premium': atm_put_exit_premium, 'Exit Time' : put_exit_time, 'Index': atm_price})
            current_price = atm_price


        while current_time.time() < pd.to_datetime('15:15:00').time():
            print("non expiry day is", day)
            next_time = current_time + pd.Timedelta(minutes=15 if TIME_FRAME == '15T' else 10)
            
            for _ in range(60):  # check up to 60 minutes ahead
                match = daily_data[daily_data['Time'] == next_time.time()]
                if not match.empty:
                    new_price = match['Open'].values[0]
                    break
                next_time += pd.Timedelta(minutes=1)
            else:
                new_price = None 

            print("current_price", current_price)
            print("new_price", new_price)
            diff = abs(new_price - current_price)
            delta = int((diff // 50) * 50)

            if delta in [50]:
                quantity = -10 * (delta // 50)            
                if new_price > current_price: 
                    put_strike = current_price + delta 
                    filtered_data_put = daily_option_data[
                        (daily_option_data['StrikePrice'] == put_strike) & 
                        (daily_option_data['Type'] == 'PE') & 
                        (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                        (daily_option_data['Time'] <= exit_time)
                    ]['Open'].sort_index()
                                
                    if not filtered_data_put.empty:
                        put_premium = filtered_data_put.iloc[0]
                    else:
                        print(f"Warning: No option data for {entry_time}, {atm_strike} PE") 
                        put_premium = None 

                    put_exit_premium, put_exit_time = calculate_profit(
                        put_premium, put_strike, next_time, option_data, TARGET, 'PE', expiry_date, day, max_exit_datetime
                    )
                    trades.append({
                        'Date': day, 'Entry Time': next_time, 'Type': 'Sell',
                        'Strike': put_strike, 'Option': 'PE', 'Entry Premium': put_premium,
                        'Quantity': quantity, 'Exit Premium': put_exit_premium,
                        'Exit Time': put_exit_time, 'Index': new_price
                    })
                    current_price = put_strike

                elif new_price < current_price:
                    call_strike = current_price - delta 
                    print("call strike", call_strike)
                    filtered_data_call = daily_option_data[
                        (daily_option_data['StrikePrice'] == call_strike) & 
                        (daily_option_data['Type'] == 'CE') & 
                        (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                        (daily_option_data['Time'] <= exit_time)
                    ]['Open'].sort_index()

                    if not filtered_data_call.empty:
                        call_premium = filtered_data_call.iloc[0]
                    else:        
                        print(f"Warning: No option data for {entry_time}, {atm_strike} CE")
                        call_premium = None 

                    call_exit_premium, call_exit_time = calculate_profit(
                        call_premium, call_strike, next_time, option_data, TARGET, 'CE', expiry_date, day, max_exit_datetime
                    )
                    trades.append({
                        'Date': day, 'Entry Time': next_time, 'Type': 'Sell',
                        'Strike': call_strike, 'Option': 'CE', 'Entry Premium': call_premium,
                        'Quantity': quantity, 'Exit Premium': call_exit_premium,
                        'Exit Time': call_exit_time, 'Index': new_price
                    })
                    current_price = call_strike

                else:
                    new_price = current_price                 
                current_time = next_time   

            elif delta >= 100:
                quantity = -10
                if new_price > current_price:
                    for step in range(50, delta + 1, 50):
                        put_strike = current_price + step
                        filtered_data_put = daily_option_data[
                            (daily_option_data['StrikePrice'] == put_strike) & 
                            (daily_option_data['Type'] == 'PE') & 
                            (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                            (daily_option_data['Time'] <= exit_time)
                        ]['Open'].sort_index()
                        
                        if not filtered_data_put.empty:
                            put_premium = filtered_data_put.iloc[0]
                        else:
                            print(f"Warning: No option data for {entry_time}, {atm_strike} PE") 
                            put_premium = None 

                        put_exit_premium, put_exit_time = calculate_profit(
                            put_premium, put_strike, next_time, option_data, TARGET, 'PE', expiry_date, day, max_exit_datetime
                        )
                        trades.append({
                            'Date': day, 'Entry Time': next_time, 'Type': 'Sell',
                            'Strike': put_strike, 'Option': 'PE', 'Entry Premium': put_premium,
                            'Quantity': quantity, 'Exit Premium': put_exit_premium,
                            'Exit Time': put_exit_time, 'Index': new_price
                        })
                    current_price = current_price + delta

                elif new_price < current_price:
                    for step in range(50, delta + 1, 50):
                        call_strike = current_price - step
                        filtered_data_call = daily_option_data[
                            (daily_option_data['StrikePrice'] == call_strike) & 
                            (daily_option_data['Type'] == 'CE') & 
                            (daily_option_data['Time'] >= next_time.strftime('%H:%M')) & 
                            (daily_option_data['Time'] <= exit_time)
                        ]['Open'].sort_index()

                        if not filtered_data_call.empty:
                            call_premium = filtered_data_call.iloc[0]
                        else:
                            print(f"Warning: No option data for {entry_time}, {atm_strike} CE")
                            call_premium = None 

                        call_exit_premium, call_exit_time = calculate_profit(
                            call_premium, call_strike, next_time, option_data, TARGET, 'CE', expiry_date, day, max_exit_datetime
                        )
                        trades.append({
                            'Date': day, 'Entry Time': next_time, 'Type': 'Sell',
                            'Strike': call_strike, 'Option': 'CE', 'Entry Premium': call_premium,
                            'Quantity': quantity, 'Exit Premium': call_exit_premium,
                            'Exit Time': call_exit_time, 'Index': new_price
                        })
                    current_price = current_price - delta

                else:
                    new_price = current_price                  
                current_time = next_time        
            else:
                current_time = next_time
                    
        # trades_n = [trade for trade in trades if trade['Date'] == day]
        # if len(trades_n) !=0:        
        #     hedge_premium = (sum(trade['Entry Premium'] for trade in trades_n))
        #     ce_trades_quantity = sum(trade['Quantity'] for trade in trades_n if ((trade['Option'] == 'CE') & (trade['Exit Time'] != '15:20:00')))
        #     pe_trades_quantity = sum(trade['Quantity'] for trade in trades_n if ((trade['Option'] == 'PE') & (trade['Exit Time'] != '15:20:00')))
        #     hedge_premium_per_trade = hedge_premium/ len(trades_n)
        #     call_hedge_strike = find_hedge_strike(daily_option_data, 'CE', 0.25 * hedge_premium_per_trade, '15:15')
        #     call_hedge_premium = daily_option_data[(daily_option_data['StrikePrice'] == call_hedge_strike) & (daily_option_data['Type'] == 'CE') & (daily_option_data['Time'] == '15:15')]['Open'].iloc[0]
        #     call_hedge_exit_premium = option_data[(option_data['StrikePrice'] == call_hedge_strike) & (option_data['Type'] == 'CE') & (option_data['ExpiryDate'] == expiry_date) & (option_data['Date'] == expiry_date)]['Open'].iloc[-1]
                
        #     put_hedge_strike = find_hedge_strike(daily_option_data, 'PE', 0.25 * hedge_premium_per_trade, '15:15')
        #     put_hedge_premium = daily_option_data[(daily_option_data['StrikePrice'] == put_hedge_strike) & (daily_option_data['Type'] == 'PE') & (daily_option_data['Time'] == '15:15')]['Open'].iloc[0]
        #     put_hedge_exit_premium = option_data[(option_data['StrikePrice'] == put_hedge_strike) & (option_data['Type'] == 'PE') & (option_data['ExpiryDate'] == expiry_date) & (option_data['Date'] == expiry_date)]['Open'].iloc[-1]
        #     trades.append({'Date':day, 'Entry Time': current_time.time(),'Type': 'Buy', 'Strike': put_hedge_strike, 'Option': 'PE', 'Entry Premium': put_hedge_premium, 'Quantity': -pe_trades_quantity, 'Exit Premium': put_hedge_exit_premium, 'Exit Time': '15:20:00', 'Index': put_hedge_strike})
        #     trades.append({'Date':day, 'Entry Time': current_time.time(),'Type': 'Buy', 'Strike': call_hedge_strike, 'Option': 'CE', 'Entry Premium': call_hedge_premium, 'Quantity': -ce_trades_quantity, 'Exit Premium': call_hedge_exit_premium, 'Exit Time': '15:20:00', 'Index': call_hedge_strike})
        # else:
        #     continue           
    # trades_df = pd.DataFrame(trades)
    trades = pd.DataFrame(trades)
    # print(trades)

    # trades = pd.DataFrame(trades)  
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])
    trades['Date'] = pd.to_datetime(trades['Date'])
    trades['Entry Premium'] = trades['Entry Premium'].astype(float)

    # Get unique trading dates
    unique_dates = trades['Date'].unique()

    # Store hedge trades
    hedge_trades = []

    # Process each trading day separatelys
    for trade_date in unique_dates:
        # print(trade_date)
        threshold_time = trade_date + pd.Timedelta(hours=15, minutes=15)
        
        # Filter trades that are not squared off before 15:15
        option_data['Date'] = pd.to_datetime(option_data['Date']).dt.date
        daily_trades = trades[(trades['Date'] == trade_date) & (trades['Exit Time'] > threshold_time)]
        # daily_trades = trades[(trades['Date'] == trade_date)]
        daily_option_data = option_data[option_data['Date'] == trade_date.date()] # Lock daily option data

        if daily_trades.empty:
            continue  # Skip if no trades exist for the day

        hedge_premium = daily_trades['Entry Premium'].sum()
        hedge_premium_per_trade = hedge_premium / len(daily_trades)

        # Process each trade that hasn't exited before 15:15
        for index, trade in daily_trades.iterrows():
            option_type = trade['Option']  # CE or PE
            strike_price = trade['Strike']
            # hedge_premium = trade['Entry Premium']
            # hedge_premium_per_trade = hedge_premium
            # print(hedge_premium_per_trade)
            # threshold_time =  trade['Entry Time'] 
            # threshold_time = pd.to_datetime(trade['Entry Time'])

            # Find hedge strike at 15:15
            hedge_strike = find_hedge_strike2(daily_option_data, option_type, HEDGE * hedge_premium_per_trade, threshold_time)
            # print(hedge_strike)
            if hedge_strike:
                # Fetch hedge entry premium at 15:15
                hedge_premium = daily_option_data[
                    (daily_option_data['StrikePrice'] == hedge_strike) & 
                    (daily_option_data['Type'] == option_type) & 
                    (daily_option_data.index == threshold_time)
                ]['Open'].iloc[0]

                # Fetch hedge exit premium at trade's exit time
                filtered_data = option_data[
                    (option_data['StrikePrice'] == hedge_strike) & 
                    (option_data['Type'] == option_type) & 
                    (option_data.index >= threshold_time) & 
                    (option_data.index <= trade['Exit Time'])
                ]

                if not filtered_data.empty:
                    hedge_exit_premium = filtered_data['Open'].iloc[-1]  # Take the last available value
                else:
                    hedge_exit_premium = None  # Handle cases where no data is available
                    print(f"⚠️ No hedge exit premium found between {threshold_time} and {trade['Exit Time']}")

                print(f"Hedge Exit Premium: {hedge_exit_premium}")

                hedge_trades.append({        
                    'Trade Index': index,  # To align with main trades
                    'Hedge Date': trade_date,
                    'Hedge Entry Time': threshold_time,
                    'Hedge Type': 'Buy',
                    'Hedge Strike': hedge_strike,
                    'Hedge Option': 'CE' if option_type == 'CE' else 'PE',
                    'Hedge Entry Premium': hedge_premium,
                    'Hedge Quantity': -trade['Quantity'],  # Hedge with same lot size as sell trade
                    'Hedge Exit Premium': hedge_exit_premium,
                    'Hedge Exit Time': trade['Exit Time']
                })

    # Convert hedge trades to DataFrame
    hedge_trades_df = pd.DataFrame(hedge_trades)

    # Merge hedge trades with the main trades based on the index
    trades_df = trades.reset_index().merge(hedge_trades_df, left_on='index', right_on='Trade Index', how='left')

    # Drop unnecessary index columns
    trades_df.drop(columns=['Trade Index'], inplace=True)

    # Save the modified tradesheet
    # trades_df.to_csv(f"{output_folder_path}/Split_{day}_DH2_trades_{int(TARGET*100)}.csv", index=False)
    strategy_name = f'{stock}_candle_{TIME_FRAME}_hedge_{HEDGE}_target_{TARGET}'
    sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')


    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4','Day' ,'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
    row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
    filter_df1.loc[row_index, 'Parameters'] = target_list
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date



    if not trades_df.empty:
        trades_df.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)

    existing_csv_file = rf"{filter_df_path}/filter_df{counter}.csv"
    if os.path.isfile(existing_csv_file):
        filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
    else:
        filter_df1.to_csv(existing_csv_file, index=False)
        
    return sanitized_strategy_name + '_' + str(start_date) + '_' + str(end_date)

# profit_points = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]



def parameter_process(parameter, mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, HEDGE, TARGET = parameter
    
    resampled_df = resample_data(df,TIME_FRAME) 
    resampled = resampled_df.dropna() 
    return process_profit_point(mapped_days ,option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, HEDGE, TARGET)


superset = 'delta_hedging'
stock = 'NIFTY'
option_type = 'ND'

dte_list = [1,2,3,6,7]

roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' else (50 if stock == 'FINNIFTY' else None))
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' else (3 if stock == 'FINNIFTY' else None))
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else (40 if stock == 'FINNIFTY' else None))

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


output_folder_path = rf'{root_path}/Trade_Sheets/split/'
filter_df_path = rf"{root_path}/Filter_Sheets/"
# mapped_days = pd.read_excel(expiry_file_path)
# mapped_days = mapped_days[(mapped_days['Date'] >= start_date) & (mapped_days['Date'] <= end_date)]
txt_file_path = rf'{root_path}/new_done.txt'

# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None


# ('2024-01-05', '2024-01-27')

date_ranges = [ 
    # ('2025-01-03', '2025-02-27')
    ('2025-02-14', '2025-02-27'),
    ('2024-09-27', '2025-02-13'),
    ('2024-05-31', '2024-09-26'),
                ('2024-01-19', '2024-05-30'),
                ('2023-09-29', '2024-01-18'),
                ('2023-05-26', '2023-09-28'),
                ('2023-01-20', '2023-05-25'),
                ('2022-09-30', '2023-01-19'),
                ('2022-05-27', '2022-09-29'),
                ('2021-12-31', '2022-05-26'),
                ('2021-10-01', '2021-12-30'), 
                ('2021-06-04', '2021-09-30')
                ]


profit_points = ['na', 0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
# profit_points = [0.45]
hedge_percentages = [0.25, 0.3 , 0.2]
# hedge_percentages = [0.25]
candle_time_frame = ['15T' , '10T']
# candle_time_frame = ['15T']

parameters = []

# if stock == 'BANKNIFTY':
#     short_strikes = [x * 2 for x in short_strikes]
#     long_strikes = [x * 2 for x in long_strikes]



# if __name__ == '__main__':
#     processes = []
#     for TARGET in profit_points:
#         p = multiprocessing.Process(target=process_profit_point, args=(TARGET,))
#         processes.append(p)
#         p.start()

#     for p in processes:
#         p.join()

#     print("All processes completed.")






if __name__ == "__main__":
    
    counter = 0
    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]
    # end_date_idx = '2024-10-25'

    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[
    (mapped_days['Date'] >= start_date_idx) & 
    (mapped_days['Date'] <= end_date_idx) & 
    (mapped_days['Date'] != '2024-05-18') & 
    (mapped_days['Date'] != '2024-05-20')
]   
    mapped_days = mapped_days[
    ~((mapped_days['Date'] >= '2024-05-30') & (mapped_days['Date'] <= '2024-06-06'))
]
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry' : 'DaysToExpiry'})
    # mapped_days = mapped_days[mapped_days['Date'] != pd.Timestamp('2024-05-18')]
    weekdays = mapped_days['Date'].to_list()
    # Pull Index Data
    
    df = pull_index_data(start_date_idx, end_date_idx, stock,mapped_days)
    # option_data = pull_options_data_d(start_date, end_date, option_data_path, stock)        
    # df = pull_index_data(start_date_idx, end_date_idx, stock)
    # df = df[df['Date'] != pd.Timestamp('2024-05-18')]
    # resampled_df_main = resample_data(df, '5T')
    vix_df = pd.read_csv("/home/newberry4/jay_test/Vix_strategy/_INDIAVIX__202410251224.csv")
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
        extended_end_date = end_date_dt + timedelta(days=60)
        
        # Convert extended_end_date back to string if needed
        extended_end_date_str = extended_end_date.strftime("%Y-%m-%d")
        
        # Pull Options data with the extended end_date
        option_data = pull_options_data_d(start_date, extended_end_date_str, option_data_path, stock)
         
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
        
        elif counter == 1:
            for TIME_FRAME in candle_time_frame:
                for HEDGE in hedge_percentages:
                    for TARGET in profit_points:
                        TARGET = 100000 if TARGET == 'na' else TARGET  # Replace 'na' with 100000
                        parameters.append([TIME_FRAME, HEDGE, TARGET])


        # Read the content of the log file to check which parameters have already been processed
        print('Total parameters :', len(parameters))
        file_path = txt_file_path
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if (stock + '_candle_' + str(value[0])  + '_hedge_' + str(value[1]).replace('.', ',')  + '_target_' + str(value[2]).replace('.', ',') + '_' + start_date + '_' + end_date) not in existing_values]
        print('Parameters to run :', len(parameters))
   
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