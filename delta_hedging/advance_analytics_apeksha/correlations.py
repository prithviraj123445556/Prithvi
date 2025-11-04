import pandas as pd

# Parameters
stock = 'SENSEX'
strategy = 'SENSEX_apeksha_new'

# Load the final DataFrame (from the code you provided) 
final_df = pd.read_csv(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_pnl_final_PREMIUM.csv')

# Calculate the correlation matrix among the strategy columns (excluding the 'Date' column)
correlation_matrix = final_df.drop(columns=['Date']).corr()

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_correlation_matrix.csv')

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Now calculate the loss correlation matrix by filtering out rows where both strategies have positive PnLs
loss_df = final_df.copy()

# Remove rows where both strategies have positive daily PnL
loss_df = loss_df[(loss_df.drop(columns=['Date']) <= 0).any(axis=1)]  # Keep rows where any strategy has a negative PnL

# Calculate the correlation matrix for these filtered rows (with any negative PnL)
loss_correlation_matrix = loss_df.drop(columns=['Date']).corr()

# Save the loss correlation matrix to a new CSV file
loss_correlation_matrix.to_csv(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_loss_correlation_matrix.csv')

# Print the loss correlation matrix
print("\nLoss Correlation Matrix:")
print(loss_correlation_matrix)

# Optionally, save the correlation matrices to an Excel file
with pd.ExcelWriter(f'/home/newberry4/jay_test/delta_hedging/{strategy}/{stock}_correlation_matrices.xlsx') as writer:
    correlation_matrix.to_excel(writer, sheet_name='Correlation')
    loss_correlation_matrix.to_excel(writer, sheet_name='Loss_Correlation')

print("Correlation matrices saved to Excel.")
