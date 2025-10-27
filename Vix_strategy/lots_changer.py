import pandas as pd
import numpy as np
import os


import ast

def parse_value(value):
    # If the value is a string and looks like a list, convert it to a list
    if isinstance(value, str):
        try:
            # Try to parse the string into a list using ast.literal_eval
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If parsing fails, return an empty list or handle it based on your requirements
            return []
    # Ensure the value is a numeric list, not a string
    if isinstance(value, list):
        # Convert all elements in the list to numeric (float)
        value = [pd.to_numeric(v, errors='coerce') for v in value]
    return value



stock = 'SENSEX'
strategy = 'Vix_strategy'               
# Directory containing tradesheets     
tradesheet_dir = f'/home/newberry4/jay_test/{strategy}/{stock}/ND/Trade_Sheets/'
output_dir = f'/home/newberry4/jay_test/{strategy}/{stock}/ND/new_tradesheets/'

os.makedirs(output_dir, exist_ok=True)


# Define function to calculate premium based on action, premiums, and lots/exit type
def calculate_premium(action, ce_short, pe_short, ce_long, pe_long, exit_type):
    # Check if all inputs are integers; if so, return 0 directly
    # if all((isinstance(i, (int, float)) and i == 0) or (isinstance(i, list) and all(val == 0 for val in i)) for i in [ce_short, pe_short, ce_long, pe_long]):
    #     return 0
    print(ce_short,pe_short, ce_long , pe_long)
    ce_short = parse_value(ce_short)
    pe_short = parse_value(pe_short)
    ce_long = parse_value(ce_long)
    pe_long = parse_value(pe_long)

    # Ensure that the variables are lists if not already
    # ce_short = ce_short if isinstance(ce_short, list) else [ce_short]
    # pe_short = pe_short if isinstance(pe_short, list) else [pe_short]
    # ce_long = ce_long if isinstance(ce_long, list) else [ce_long]
    # pe_long = pe_long if isinstance(pe_long, list) else [pe_long]

    if action == 'Short':
        if exit_type == 'full':
            # For "Short" with 3 lots, 2/3rd exit
            return (ce_short + pe_short) - (ce_long+ pe_long)
        if exit_type == '2_3rd':
            # For "Short" with 3 lots, 2/3rd exit
            return (3 * ce_short + 3 * pe_short) - (3 * ce_long + 3 * pe_long)
        elif exit_type == '1_3rd':
            # For "Short" with 3 lots, 1/3rd exit
            return (3 * ce_short + 3 * pe_short) - (3 * ce_long + 3 * pe_long)
        elif exit_type == 'Half':
            # For "Short" with 2 lots, Half exit
            return (2 * ce_short + 2 * pe_short) - (2 * ce_long + 2 * pe_long)
        elif exit_type == '1_4th':
            # For "Short" with 4 lots, 1/4th exit
            return (4 * ce_short + 4 * pe_short) - (4 * ce_long + 4 * pe_long)
        elif exit_type == '3_4th':
            # For "Short" with 4 lots, 3/4th exit
            return (4 * ce_short + 4 * pe_short) - (4 * ce_long + 4 * pe_long)

    else:  # For "Long" action
        if exit_type == 'full':
            # For "Short" with 3 lots, 2/3rd exit
            return -(ce_short[0] + pe_short[0] - (ce_long[0] + pe_long[0]))
        if exit_type == '2_3rd':
            # For "Long" with 3 lots, 2/3rd exit
            return -(2 * ce_short[0] + ce_short[1] + 2 * pe_short[0] + pe_short[1] - (2 * ce_long[0] + ce_long[1] + 2 * pe_long[0] + pe_long[1]))
        elif exit_type == '1_3rd':
            # For "Long" with 3 lots, 1/3rd exit
            return -(ce_short[0] + 2 * ce_short[1] + pe_short[0] + 2 * pe_short[1] - (ce_long[0] + 2 * ce_long[1] + pe_long[0] + 2 * pe_long[1]))
        elif exit_type == 'Half':
            # For "Long" with 2 lots, Half exit
            return -((sum(ce_short) + sum(pe_short)) - (sum(ce_long) + sum(pe_long)))
        elif exit_type == '1_4th':
            # For "Long" with 4 lots, 1/4th exit
            return -(ce_short[0] + 3 * ce_short[1] + pe_short[0] + 3 * pe_short[1] - (ce_long[0] + 3 * ce_long[1] + pe_long[0] + 3 * pe_long[1]))
        elif exit_type == '3_4th':
            # For "Long" with 4 lots, 3/4th exit
            return -(3 * ce_short[0] + ce_short[1] + 3 * pe_short[0] + pe_short[1] - (3 * ce_long[0] + ce_long[1] + 3 * pe_long[0] + pe_long[1]))



def is_zero_list(value):
    # Check if the value is a list and if all elements are zero
    return isinstance(value, list) and all(i == 0 for i in value)

# Define function to process tradesheet for each lot size and exit type
def process_tradesheet(file_path, lot_num, exit_type):
    # Load the tradesheet CSV file
    trade_sheet = pd.read_csv(file_path)

    trade_sheet = trade_sheet[ 
        ~((trade_sheet['CE_Short_Premium'].isna()) | 
           (trade_sheet['PE_Short_Premium'].isna()) | 
           (trade_sheet['CE_Long_Premium'].isna()) | 
           (trade_sheet['PE_Long_Premium'].isna()) | 
           (trade_sheet['CE_Short_Premium'] == '0') | 
           (trade_sheet['PE_Short_Premium'] == '0') | 
           (trade_sheet['CE_Long_Premium'] == '0') | 
           (trade_sheet['PE_Long_Premium'] == '0') |
           # Check if any of the premium columns contain a list of zeros
           trade_sheet['CE_Short_Premium'].apply(is_zero_list) |
           trade_sheet['PE_Short_Premium'].apply(is_zero_list) |
           trade_sheet['CE_Long_Premium'].apply(is_zero_list) |
           trade_sheet['PE_Long_Premium'].apply(is_zero_list))
    ]
    
    # Apply calculations based on Action, lot number, and exit type
    if lot_num == 1 and exit_type == 'full':
        # For lot 2 with exit type 'Half', use the calculate_premium function
        trade_sheet['Premium'] = trade_sheet.apply(
            lambda x: calculate_premium(
                x['Action'],
                x['CE_Short_Premium'], 
                x['PE_Short_Premium'], 
                x['CE_Long_Premium'], 
                x['PE_Long_Premium'],
                exit_type
            ),
            axis=1
        )
        
    
    elif lot_num == 2 and exit_type == 'Half':
        # For lot 2 with exit type 'Half', use the calculate_premium function
        trade_sheet['Premium'] = trade_sheet.apply(
            lambda x: calculate_premium(
                x['Action'],
                x['CE_Short_Premium'], 
                x['PE_Short_Premium'], 
                x['CE_Long_Premium'], 
                x['PE_Long_Premium'],
                exit_type
            ),
            axis=1
        )
        
    elif lot_num == 3:
        # For lot 3 with exit types '1/3rd' and '2/3rd', use the calculate_premium function with specified exit type
        trade_sheet['Premium'] = trade_sheet.apply(
            lambda x: calculate_premium(
                x['Action'],
                x['CE_Short_Premium'], 
                x['PE_Short_Premium'], 
                x['CE_Long_Premium'], 
                x['PE_Long_Premium'],
                exit_type
            ),
            axis=1
        )

    elif lot_num == 4:
        # For lot 4 with exit types '1/4th' and '3/4th', use the calculate_premium function with specified exit type
        trade_sheet['Premium'] = trade_sheet.apply(
            lambda x: calculate_premium(
                x['Action'],
                x['CE_Short_Premium'], 
                x['PE_Short_Premium'], 
                x['CE_Long_Premium'], 
                x['PE_Long_Premium'],
                exit_type
            ),
            axis=1
        )
    
    # Update filename to indicate lot size and exit strategy
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    new_filename = f"{base_filename}_lots_{lot_num}_exit_{exit_type}.csv"
    output_path = os.path.join(output_dir, new_filename)
    
    # Save the modified tradesheet
    trade_sheet.to_csv(output_path, index=False)
    print(f"Processed {output_path}")

# Process each tradesheet in the folder
for filename in os.listdir(tradesheet_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(tradesheet_dir, filename)
        
        # Apply transformations for each lot and exit type
        for lot_num in range(1, 5):  # Adjusted range for clarity
            if lot_num == 1:
                exit_type = 'full'
                process_tradesheet(file_path, lot_num, exit_type)
            elif lot_num == 2:
                exit_type = 'Half'
                process_tradesheet(file_path, lot_num, exit_type)
            elif lot_num == 3:
                for exit_type in ['1_3rd', '2_3rd']:
                    process_tradesheet(file_path, lot_num, exit_type)
            elif lot_num == 4:
                for exit_type in ['1_4th', '3_4th']:
                    process_tradesheet(file_path, lot_num, exit_type)    