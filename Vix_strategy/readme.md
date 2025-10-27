Steps:
1)backtester
2)lots_changer
3)dte or days
4)analytics

RULES:


strike_adjustment = (days_to_expiry_sqrt / 16) * vix_open_value * ATM / 100
if stock =='NIFTY':
    CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 50)
    PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 50)
else :
    CE_Short_Strike = nearest_multiple(ATM + strike_adjustment, 100)
    PE_Short_Strike = nearest_multiple(ATM - strike_adjustment, 100)


# Testing Combinations
candle_time_frame = ['5T']
entries = ['09:30','14:30']    
exits = ['15:00']
short_strikes = [0]
long_strikes = [0.33 , 0.5 , 1000 , 500]    ## 1000 for sensex and 500 for nifty , less than 1 are percentages and more than 1 are abs values . long strikes % are % taken close to premium of short strike . 
short_stoploss_per = [2, 2.5 ,3]             2.5 is 150%
long_profit_per = [0.3,0.2,0.1]       0.3 is 70% and vice versa
premium_threshold = ['1st_week']  


we have in exit price , [exit value , eod value]

in lots_changer applying diff lots combination

in dte or days segregating trades based on day or dte

in analytics computing analytics for each dte 