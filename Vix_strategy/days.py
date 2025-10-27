import os
import pandas as pd

def process_tradesheets(input_folder, output_folder):
    # List all files in the input folder
    trade_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Create a folder for each day if it doesn't exist
    for day in days_of_week:
        day_folder = os.path.join(output_folder, day)
        os.makedirs(day_folder, exist_ok=True)

    # Process each tradesheet in the input folder
    for trade_file in trade_files:
        file_path = os.path.join(input_folder, trade_file)

        # Load the tradesheet
        trade_sheet = pd.read_csv(file_path)

        # Add the day of the week to the tradesheet name
        trade_sheet['Day'] = trade_sheet['Day'].str.strip()  # Ensures there's no extra whitespace

        # Split the tradesheet into separate files for each day
        for day in days_of_week:
            # Filter tradesheet for the specific day
            day_data = trade_sheet[trade_sheet['Day'] == day]

            if not day_data.empty:
                # Modify the filename to append the day
                output_filename = f"{os.path.splitext(trade_file)[0]}_{day}.csv"
                output_path = os.path.join(output_folder, day, output_filename)
                day_data.to_csv(output_path, index=False)

        # Also store the original file with the day added to its name
        output_filename_original = f"{os.path.splitext(trade_file)[0]}_original.csv"
        original_output_path = os.path.join(output_folder, 'Original', output_filename_original)
        trade_sheet.to_csv(original_output_path, index=False)

# Example usage:
input_folder = '/home/newberry4/jay_test/Vix_strategy/NIFTY/ND/new_tradesheets'  # Folder containing your original tradesheet CSV files
output_folder = '/home/newberry4/jay_test/Vix_strategy/NIFTY/ND/days'  # Folder where you want to store the new files

process_tradesheets(input_folder, output_folder)
