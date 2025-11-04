# import pandas as pd
# import itertools
# import pandas as pd
# import os
# from helpers import analytics

# type = "non_split"
# results_df = pd.read_excel("/home/newberry4/jay_test/delta_hedging/NIFTY_non_split/ND/analytics_top10_non_split.xlsx")
# unique_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# unique_target_points = sorted(results_df["Target_Point"].unique())


# pnl_lookup = results_df.set_index(["Weekday", "Target_Point"])["Total_PnL"].to_dict()
# all_combinations = list(itertools.product(unique_target_points, repeat=len(unique_weekdays)))

# final_results = []

# for combo in all_combinations:
#     total_pnl = 0
#     combo_dict = {weekday: target for weekday, target in zip(unique_weekdays, combo)}

#     for weekday, target in combo_dict.items():
#         total_pnl += pnl_lookup.get((weekday, target), 0) 
#     final_results.append({"Combination": combo_dict, "Total_PnL": total_pnl})


# combo_df = pd.DataFrame(final_results).sort_values(by="Total_PnL", ascending=False)
# # combo_df1 = analytics(combo_df, 'Total_PnL')
# combo_df.to_csv(f"/home/newberry4/jay_test/delta_hedging/NIFTY/ND/weekday_combos_pnl_{type}.csv", index=False)
# # combo_df1.to_csv("weekday_combos_analytics.csv", index= False)



# import pandas as pd
# import itertools

# # Load your DataFrame
# results_df = pd.read_excel("/home/newberry4/jay_test/delta_hedging/NIFTY_non_split/ND/analytics_top10_non_split.xlsx")

# # Extract strategies per weekday
# weekday_strategies = {}
# for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
#     weekday_strategies[day] = results_df[results_df['Strategy_Name'].str.contains(f"WD_{day}")]['Strategy_Name'].tolist()

# # Make sure each has 10 strategies
# assert all(len(strats) == 10 for strats in weekday_strategies.values()), "Each weekday must have exactly 10 strategies"

# # Generate all possible 5-day combinations (one per day)
# all_combos = list(itertools.product(
#     weekday_strategies['Monday'],
#     weekday_strategies['Tuesday'],
#     weekday_strategies['Wednesday'],
#     weekday_strategies['Thursday'],
#     weekday_strategies['Friday']
# ))

# print(f"Total combinations: {len(all_combos)}")  # Should be 100,000

# # Create a lookup for Strategy_Name -> Total_PnL
# pnl_lookup = results_df.set_index("Strategy_Name")["Total_PnL"].to_dict()

# # Evaluate Total_PnL for each combo
# combo_pnl_list = []
# for combo in all_combos:
#     total_pnl = sum(pnl_lookup.get(strategy, 0) for strategy in combo)
#     combo_pnl_list.append({
#         "Monday": combo[0],
#         "Tuesday": combo[1],
#         "Wednesday": combo[2],
#         "Thursday": combo[3],
#         "Friday": combo[4],
#         "Total_PnL": total_pnl
#     })

# # Convert to DataFrame
# combo_df = pd.DataFrame(combo_pnl_list)

# # Sort by Total_PnL descending
# combo_df = combo_df.sort_values(by="Total_PnL", ascending=False)

# # Save to CSV
# combo_df.to_csv("/home/newberry4/jay_test/delta_hedging/NIFTY_non_split/ND/weekday_strategy_combos_top100k.csv", index=False)

























#####   new code  #####


# import pandas as pd
# import itertools
# import os
# from helpers import analytics

# # ----------- FILES -----------
# analytics_file = "/home/newberry4/jay_test/delta_hedging/NIFTY/ND/analytics_top10_non_split.xlsx"
# tradesheet_folder = "/home/newberry4/jay_test/delta_hedging/NIFTY/ND/dailypnl_non_split"
# output_csv = "/home/newberry4/jay_test/delta_hedging/NIFTY_non_split/ND/weekday_strategy_combos_top30.csv"

# # ----------- LOAD STRATEGIES AND PNL -----------
# analytics_df = pd.read_excel(analytics_file)
# weekday_strategies = {
#     day: analytics_df[analytics_df['Strategy_Name'].str.contains(f"WD_{day}")]['Strategy_Name'].tolist()
#     for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
# }
# print(analytics_df)

# # ----------- STEP 1: BUILD ALL COMBOS & TOTAL PNL (FAST) -----------
# combo_pnl_list = []

# all_combos = list(itertools.product(
#     weekday_strategies['Monday'],
#     weekday_strategies['Tuesday'],
#     weekday_strategies['Wednesday'],
#     weekday_strategies['Thursday'],
#     weekday_strategies['Friday']
# ))

# for combo in all_combos:
#     total_pnl = 0
#     skip = False

#     for strat in combo:
#         row = analytics_df[analytics_df["Strategy_Name"] == strat]
#         if row.empty:
#             skip = True
#             break
#         total_pnl += row["Total_PnL"].values[0]

#     if not skip:
#         combo_pnl_list.append({
#             "Monday": combo[0],
#             "Tuesday": combo[1],
#             "Wednesday": combo[2],
#             "Thursday": combo[3],
#             "Friday": combo[4],
#             "Total_PnL": total_pnl
#         })

# # ----------- STEP 2: SELECT TOP 30 COMBOS -----------
# combo_df = pd.DataFrame(combo_pnl_list)
# top_30_combos = combo_df.sort_values(by="Total_PnL", ascending=False).head(30)
# print(f"Top 30 combinations ", top_30_combos)

# # ----------- STEP 3: FOR TOP 30, DO ANALYTICS WITH ACTUAL TRADE DATA -----------
# final_results = []

# for idx, row in top_30_combos.iterrows():
#     all_trades = []

#     for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
#         strat = row[day]
#         base_name = strat.split("_WD_")[0]
#         weekday = strat.split("_WD_")[1]

#         file_path = os.path.join(tradesheet_folder, f"{base_name}.xlsx")
#         if not os.path.exists(file_path):
#             print(f"Missing file: {file_path}")
#             continue

#         df = pd.read_excel(file_path)
#         df["Date"] = pd.to_datetime(df["Date"])
#         df["Day"] = df["Date"].dt.day_name()
#         df_filtered = df[df["Day"] == weekday]

#         all_trades.append(df_filtered)

#     if not all_trades:
#         continue

#     combined_df = pd.concat(all_trades).sort_values("Date")
#     metrics = analytics(combined_df, "Pnl")

#     final_results.append({
#         "Monday": row["Monday"],
#         "Tuesday": row["Tuesday"],
#         "Wednesday": row["Wednesday"],
#         "Thursday": row["Thursday"],
#         "Friday": row["Friday"],
#         **metrics
#     })

# # ----------- SAVE FINAL RESULT -----------
# final_df = pd.DataFrame(final_results)
# final_df = final_df.sort_values(by="Total_PnL", ascending=False)
# final_df.to_csv(output_csv, index=False)
# print(f"✅ Done! Saved top 30 analytics to {output_csv}")

















#### final code ####

import pandas as pd
import itertools
import os
import multiprocessing as mp
from helpers import analytics2
from datetime import datetime, timedelta


# ----------- CONFIG -----------
stock = 'NIFTY'
split_type = 'split'

analytics_file = f"/home/newberry4/jay_test/delta_hedging/{stock}/ND/analytics_top10_{split_type}.xlsx"
tradesheet_folder = f"/home/newberry4/jay_test/delta_hedging/{stock}/ND/dailypnl/{split_type}/"
output_csv = f"/home/newberry4/jay_test/delta_hedging/{stock}/ND/weekday_strategy_combos_full_{split_type}.csv"

# ----------- LOAD STRATEGY PNL -----------
results_df = pd.read_excel(analytics_file)
print("results_df", results_df)
weekday_strategies = {
    day: results_df[results_df['Strategy_Name'].str.contains(f"WD_{day}")]['Strategy_Name'].tolist()
    for day in ['Monday', 'Tuesday', 'Wednesdsay', 'Thursday', 'Friday']
}
assert all(len(strats) == 10 for strats in weekday_strategies.values())

# ----------- EXTRACT BASE STRATEGY NAMES -----------
def extract_base_and_day(strategy_name):
    parts = strategy_name.split("_WD_")
    return parts[0], parts[1] if len(parts) > 1 else ""

unique_strategies = {extract_base_and_day(strat)[0] for strats in weekday_strategies.values() for strat in strats}

# ----------- LOAD TRADE SHEETS BY STRATEGY & SPLIT BY DAY -----------
strategy_trades_by_day = {}
for base in unique_strategies:
    path = os.path.join(tradesheet_folder, f"{base}.xlsx")
    if os.path.exists(path):
        df = pd.read_excel(path)
        print("the df is ",df)
        start_exclude = datetime.strptime("2024-05-31", "%Y-%m-%d").date()
        end_exclude = datetime.strptime("2024-06-06", "%Y-%m-%d").date()

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            df = df[~((df['Date'] >= start_exclude) & (df['Date'] <= end_exclude))]

        elif 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            df = df[~((df['entry_date'].dt.date >= start_exclude) &
                                    (df['entry_date'].dt.date <= end_exclude))]
        df['Date'] = pd.to_datetime(df['Date'])
        # df['Day'] = df['Date'].dt.day_name()
        strategy_trades_by_day[base] = dict(tuple(df.groupby("Day")))  # ✅ Fix here
    else:
        print(f"Missing tradesheet: {path}")
        strategy_trades_by_day[base] = {}

# ----------- SHARED ACCESS FOR MULTIPROCESSING -----------
shared_trades = None

def init_worker(trades_data):
    global shared_trades
    shared_trades = trades_data

def process_combo(combo):
    def extract_base_and_day(strategy_name):
        parts = strategy_name.split("_WD_")
        return parts[0], parts[1] if len(parts) > 1 else ""

    combined_trades = []
    for strat_full in combo:
        base, day = extract_base_and_day(strat_full)
        df = shared_trades.get(base, {}).get(day, pd.DataFrame())
        if not df.empty:
            combined_trades.append(df)

    if not combined_trades:
        print(f"🚫 No data for combo: {combo}")
        return None

    combined_df = pd.concat(combined_trades).sort_values("Date")
    print("combined_df", combined_df)
    combined_df = combined_df[combined_df['Date'] > pd.Timestamp("2022-05-01")]

    expiry_pnl_df = (
        combined_df.groupby("ExpiryDate")["Pnl"]
        .sum()
        .reset_index()
        .sort_values("ExpiryDate")
    )

    metrics = analytics2(combined_df, expiry_pnl_df, "Pnl")

    # ✅ Convert 1-row DataFrame to dict
    if isinstance(metrics, pd.DataFrame) and len(metrics) == 1:
        metrics = metrics.iloc[0].to_dict()
    else:
        print(f"❗ Invalid metrics shape or type for combo: {combo} → {type(metrics)} = {metrics}")
        return None

    return {
        "Monday": combo[0],
        "Tuesday": combo[1],
        "Wednesday": combo[2],
        "Thursday": combo[3],
        "Friday": combo[4],
        **metrics
    }


# ----------- PROCESS ALL COMBOS -----------
print("🔄 Processing all weekday strategy combinations in parallel...")

all_combos = list(itertools.product(
    weekday_strategies['Monday'],
    weekday_strategies['Tuesday'],
    weekday_strategies['Wednesday'],
    weekday_strategies['Thursday'],
    weekday_strategies['Friday']
))



print("🔄 Analyzing all combos in parallel...")
with mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(strategy_trades_by_day,)) as pool:
    final_combos = list(pool.map(process_combo, all_combos))


print("✅ Finished multiprocessing. Sample of final_combos:")
for i, item in enumerate(final_combos[:3]):  # Just first 3 for brevity
    print(f"[{i}] {item}")


final_combos = [x for x in final_combos if x is not None and isinstance(x.get("totalpnl", None), (int, float))]

# Now safe to create dataframe
final_df = pd.DataFrame(final_combos)
print("🧾 Columns in final_df:", final_df.columns.tolist())

# Try both column names if needed
if "totalpnl" in final_df.columns:
    final_df = final_df.sort_values("totalpnl", ascending=False)
elif "Total_PnL" in final_df.columns:
    final_df = final_df.sort_values("Total_PnL", ascending=False)
else:
    print("❌ Neither 'totalpnl' nor 'Total_PnL' found in final_df.")


final_df.to_csv(output_csv, index=False)
print(f"✅ Done! Saved top 30 analytics to {output_csv}")


