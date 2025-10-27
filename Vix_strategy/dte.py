import os
import pandas as pd

def process_tradesheets(input_folder, output_folder):
    # List all files in the input folder
    trade_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Days to expiry values
    dte_values = [1, 2, 3, 4, 0.25]

    # Create a folder for each DTE value if it doesn't exist
    for dte in dte_values:
        dte_folder = os.path.join(output_folder, str(dte))
        os.makedirs(dte_folder, exist_ok=True)

    # Create folder for original full files
    original_folder = os.path.join(output_folder, 'Original')
    os.makedirs(original_folder, exist_ok=True)

    # Process each tradesheet in the input folder
    for trade_file in trade_files:
        file_path = os.path.join(input_folder, trade_file)

        # Load the tradesheet
        trade_sheet = pd.read_csv(file_path)

        # Remove extra spaces (if any) from DaysToExpiry
        if trade_sheet['DaysToExpiry'].dtype == object:
            trade_sheet['DaysToExpiry'] = trade_sheet['DaysToExpiry'].str.strip()

        # Split the tradesheet into separate files for each DTE value
        for dte in dte_values:
            day_data = trade_sheet[trade_sheet['DaysToExpiry'] == dte]

            if not day_data.empty:
                output_filename = f"{os.path.splitext(trade_file)[0]}_DTE{dte}.csv"
                output_path = os.path.join(output_folder, str(dte), output_filename)
                day_data.to_csv(output_path, index=False)

        # Also store the original file
        output_filename_original = f"{os.path.splitext(trade_file)[0]}_original.csv"
        original_output_path = os.path.join(original_folder, output_filename_original)
        trade_sheet.to_csv(original_output_path, index=False)


# Example usage:

stock = 'SENSEX'
strategy = 'Vix_strategy'
input_folder = f'/home/newberry4/jay_test/{strategy}/{stock}/ND/new_tradesheets'  # Folder containing your original tradesheet CSV files
output_folder = f'/home/newberry4/jay_test/{strategy}/{stock}/ND/dte'  # Folder where you want to store the new files

os.makedirs(output_folder, exist_ok=True)
process_tradesheets(input_folder, output_folder)
