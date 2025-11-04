import time
import pandas as pd
import numpy as np
import psycopg2 as sql
from datetime import datetime, timedelta
import os
from functools import partial

def compare_month_and_year(date1, date2, file, stock):
    date1 = datetime.strptime(date1, "%Y-%m-%d").replace(day=1)
    date2 = datetime.strptime(date2, "%Y-%m-%d").replace(day=1)
    date3 = datetime.strptime(file.replace(stock + '_', '')[0:4] + file.replace(stock + '_', '')[4:6] + '01', "%Y%m%d").date().strftime("%Y-%m-%d")
    date3 = datetime.strptime(date3, '%Y-%m-%d')    
    if (date1 <= date3) & (date3 <= date2):
        return file
    else:
        return None
    
def modify_ticker(ticker):
    date_part = ticker[:8] 
    instrument_part = ticker[13:]
    return date_part + instrument_part

def load_options_data(start_date, end_date, stock, expiry_file_path, option_file_path):

        dte_column = "DaysToExpiry"  
        mapped_days = pd.read_excel(expiry_file_path)    
        mapped_days = mapped_days[(mapped_days['Date'] >= start_date) & (mapped_days['Date'] <= end_date)]
        weekdays = mapped_days['Date'].to_list()
        dte_list = [0]    # change for dte1
        mapped_days_new = mapped_days[mapped_days[dte_column].isin(dte_list)]
        option_data_files = next(os.walk(option_file_path))[2]
        option_data_list = []        
        for file in option_data_files:
            if compare_month_and_year(start_date, end_date, file, stock):
                temp_data = pd.read_pickle(option_file_path + file)[['ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Ticker']]
                temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format='%Y%m%d%H:%M')
                temp_data = temp_data.rename_axis('DateTime')
                option_data_list.append(temp_data)
        
        option_data = pd.concat(option_data_list)
        option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
        option_data['Type'] = option_data['Type'].astype('category')
        option_data = option_data.sort_index()
        option_data = option_data.reset_index().rename(columns={'index': 'DateTime'})
        mapped_days_new['Date'] = mapped_days_new['Date'].astype(str)
        option_data['Date'] = (option_data['DateTime'].astype(str)).str[:10]        
        option_data = option_data[option_data['Date'].isin(mapped_days_new['Date'])]
        option_data['DateTime'] = pd.to_datetime(option_data['DateTime'])
        option_data['Ticker'] = option_data['Ticker'].astype(str) 
                 
        return option_data


def process_file_expiry(file_path, strategy):
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
        if 'expiry_date'  in df.columns:
            df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        else :
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
        df = df[df['entry_date'].dt.date == df['expiry_date'].dt.date]      ###### use for apeksha
        return df


def process_file(file_path, strategy):
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
        if 'expiry_date'  in df.columns:
            df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        else :
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
        return df


def read_and_process_folder(folder_path):
        all_data = pd.DataFrame()
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                processed_df = process_file(file_path)
                all_data = pd.concat([all_data, processed_df], ignore_index=True)        
        return all_data


def params(trade_sheet, option_data): 
    trade_sheet = trade_sheet[['Date', 'ExpiryDate', 'CE_Time', 'PE_Time', 'CE_OTM', 'PE_OTM', 'DaysToExpiry', 'CE_OTM_Premium', 'PE_OTM_Premium', 'CE_OTM_New_Premium', 'PE_OTM_New_Premium','Position', 'Lots']]
    #option_data = option_data.drop(columns=['Low', "High"])
    option_data = option_data.sort_values(by=['DateTime']).reset_index(drop=True)
    return trade_sheet, option_data

def postgresql_query(input_query, input_tuples = None):
    try:
        connection = sql.connect(
            host = "algo-backtest-data-do-user-14334263-0.b.db.ondigitalocean.com",
            port = 25060,
            database = "defaultdb",
            user = "doadmin",
            password = "AVNS_kOwuGIv2gd1DmiPl9Cx",
            sslmode = "require"
        )

        cursor = connection.cursor()
        if input_tuples is not None:
            cursor.execute(input_query, input_tuples)
        else:
            cursor.execute(input_query)
        data = cursor.fetchall()

    except sql.Error as e:
        print('Error connecting to the database:', e)
        return e

    else:
        if cursor:
            cursor.close()
        if connection:
            connection.close()       
        return data


'''def get_daily_min_max_pnl(calculated_pnl_df, args):    # returns daily min/max PnL in another df
    result_df = calculated_pnl_df.groupby(calculated_pnl_df[defs.col_ohlc_timestamp].dt.date)['pnl'].agg(['min', 'max']).reset_index()
    result_df.columns = ['Trade_date', 'Daily_min_pnl', 'Daily_max_pnl']
    # file_path_input = input('Enter a file path to store daily min and max PnL data: ')
    if args.pnl_filepath is not None:
        result_df.to_csv(args.pnl_filepath, index=False)
        
        
        
    def resample_data(data):
    data.set_index('DateTime', inplace=True)
    resampled_data = data.groupby(pd.Grouper(freq='1T', origin=data.index[0])).first()
    resampled_data.reset_index(inplace=True)
    return resampled_data'''

'''import pandas as pd

df = pd.read_csv('/home/newberry3/Ruchika/final_merged_pnl.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date
trades = df.copy(deep = True)
grouped = df.groupby('date')

total_pnl = 0
last_trade_date = None
daily_pnls = []
max_crossing_count = 0

#analytics for portfolio : no profit point/loss point

#1L profit + 10kinc 
# 25k loss : 5kinc 
# if both, then whichever is hit first

results = []
for date, group in grouped:
    max_pnl = group['PnL Combined'].max()
    min_pnl = group['PnL Combined'].min()
    threshold_max = 100000
    threshold_min = -25000
    max_crossing_row = group[group['PnL Combined'] >= threshold_max].iloc[0] if not group[group['PnL Combined'] >= threshold_max].empty else None
    min_crossing_row = group[group['PnL Combined'] <= threshold_min].iloc[0] if not group[group['PnL Combined'] <= threshold_min].empty else None
    
    results.append({
        'Date': date,
        #'max_crossing': max_crossing_row['DateTime'] if max_crossing_row is not None else None,
        'Daily PNl': max_crossing_row['PnL Combined'] if max_crossing_row is not None else group['PnL Combined'].iloc[-1],
        #'min_crossing': min_crossing_row['DateTime'] if min_crossing_row is not None else None,
        #'min_crossing_pnl': min_crossing_row['PnL Combined'] if min_crossing_row is not None else None,
        #'max_pnl': max_pnl,
        #'min_pnl': min_pnl
    })

    if max_crossing_row is not None:
        trades_filtered = trades[trades['DateTime'] <= max_crossing_row['DateTime']]
        max_crossing_count += 1
        last_trade_date = max_crossing_row['DateTime']
    else:
        trades_filtered = trades
        last_trade_date = trades['DateTime'].iloc[-1]
    daily_pnl = trades_filtered['PnL Combined'].iloc[-1]
    daily_pnls.append(daily_pnl)
    total_pnl += daily_pnl

final_metrics_df = pd.DataFrame(results)
final_metrics_df['No of times PnL crossed over'] = max_crossing_count
print(final_metrics_df)'''