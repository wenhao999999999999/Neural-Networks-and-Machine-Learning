from utils.tools import aggregate_hourly_to_daily

df = aggregate_hourly_to_daily('data/ELE/power_load.csv')
print(df.head())