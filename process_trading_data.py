import pandas as pd
 
# Function to calculate EMA
def calculate_ema(prices, window_size, smoothing_factor):
    ema = [prices.iloc[0]]  # Initialize EMA with the first price
    for price in prices.iloc[1:]:
        ema.append((price * (smoothing_factor / (1 + window_size))) + ema[-1] * (1 - (smoothing_factor / (1 + window_size))))
    return pd.Series(ema, index=prices.index)
 
# File paths for the dataset, should be replaced accordingly where the dataset is placed before testing
file_paths = ['Downloads/6382482/debs2022-gc-trading-day-08-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-09-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-10-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-11-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-12-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-13-11-21.csv', 'Downloads/6382482/debs2022-gc-trading-day-14-11-21.csv']  # Replace with actual file paths
 
 
# Define column names
columns = [
    'ID', 'SecType', 'Date', 'Time', 'Ask', 'Ask volume', 'Bid', 'Bid volume', 'Ask time', 
    "Day's high ask", 'Close', 'Currency', "Day's high ask time", "Day's high", 'ISIN', 
    'Auction price', "Day's low ask", "Day's low", "Day's low ask time", 'Open', 
    'Nominal value', 'Last', 'Last volume', 'Trading time', 'Total volume', 'Mid price', 
    'Trading date', 'Profit', 'Current price', 'Related indices', 'Day high bid time', 
    'Day low bid time', 'Open Time', 'Last trade time', 'Close Time', 'Day high Time', 
    'Day low Time', 'Bid time', 'Auction Time'
]
 
chunk_size = 20**8
processed_data = []
 
# Processing files
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    for chunk in pd.read_csv(
        file_path,
        chunksize=chunk_size,
        skiprows=12,  # Skip metadata rows
        names=columns,
        on_bad_lines='skip'
    ):
        print(f"Initial chunk size: {chunk.shape}")
        # Filter relevant columns
        filtered_chunk = chunk[['ID', 'Date', 'Time', 'Last']].dropna(subset=['Date', 'Time', 'Last'])
 
        # Log unique values for debugging
        print(f"Unique Date values: {filtered_chunk['Date'].unique()}")
        print(f"Unique Time values: {filtered_chunk['Time'].unique()}")
        print(f"Unique Last values: {filtered_chunk['Last'].unique()}")
 
        # Clean Date and Time columns
        filtered_chunk['Date'] = filtered_chunk['Date'].str.strip()
        filtered_chunk['Time'] = filtered_chunk['Time'].str.strip()
 
        # Convert Date and Time to datetime
        filtered_chunk['datetime'] = pd.to_datetime(
            filtered_chunk['Date'] + ' ' + filtered_chunk['Time'],
            errors='coerce',
            format='%d-%m-%Y %H:%M:%S.%f'
        )
        # Drop rows with invalid datetime
        filtered_chunk = filtered_chunk.drop(columns=['Date', 'Time']).dropna(subset=['datetime', 'Last'])
        print(f"Filtered chunk size after datetime parsing: {filtered_chunk.shape}")
        processed_data.append(filtered_chunk)
 
# Combine processed chunks
if processed_data:
    data = pd.concat(processed_data, ignore_index=True)
else:
    print("No valid data found.")
    data = pd.DataFrame(columns=['ID', 'datetime', 'Last'])
 
# Ensure datetime is the index
data.set_index('datetime', inplace=True)
 
# Resample and calculate EMAs
if not data.empty:
    resampled_data = []
    for symbol, group in data.groupby('ID'):
        group = group.resample('5min').last()  # Resample for 5-minute intervals
        group['ID'] = symbol  # Reintroduce ID as a column
        resampled_data.append(group)
    # Combine the resampled data
    data = pd.concat(resampled_data)
    # Reset index
    data.reset_index(inplace=True)
    # Fill missing values in 'Last' with forward fill
    data['Last'] = data['Last'].fillna(method='ffill')
    # Calculate short and long EMAs
    smoothing_factor = 2
    data['EMA38'] = data.groupby('ID')['Last'].transform(lambda x: calculate_ema(x, 38, smoothing_factor))
    data['EMA100'] = data.groupby('ID')['Last'].transform(lambda x: calculate_ema(x, 100, smoothing_factor))
    # Detect breakout patterns
    data['bullish_breakout'] = (data['EMA38'] > data['EMA100']) & (data['EMA38'].shift(1) <= data['EMA100'].shift(1))
    data['bearish_breakout'] = (data['EMA38'] < data['EMA100']) & (data['EMA38'].shift(1) >= data['EMA100'].shift(1))
    # Add trade advisory
    data['advisory'] = 'Hold'
    data.loc[data['bullish_breakout'], 'advisory'] = 'Buy'
    data.loc[data['bearish_breakout'], 'advisory'] = 'Sell'
    print("Processed data:")
    print(data.head())
else:
    print("No data available after processing.")
 
# Save results
data.to_csv('processed_data_with_ema.csv', index=False)
