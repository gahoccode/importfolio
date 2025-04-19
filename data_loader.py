import pandas as pd
from vnstock import Quote

class DataLoader:
    def __init__(self, symbols=None, start_date=None, end_date=None, interval="1D"):
        if not symbols:
            symbols = ["REE", "FMC", "TLG"]
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def fetch(self):
        all_historical_data = {}
        for symbol in self.symbols:
            try:
                quote = Quote(symbol=symbol)
                historical_data = quote.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    to_df=True
                )
                if not historical_data.empty:
                    all_historical_data[symbol] = historical_data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        return all_historical_data

    @staticmethod
    def combine_data(data_dict):
        combined_data = pd.DataFrame()
        for symbol, data in data_dict.items():
            if not data.empty:
                temp_df = data.copy()
                for col in temp_df.columns:
                    if col != 'time':
                        temp_df.rename(columns={col: f'{symbol}_{col}'}, inplace=True)
                if combined_data.empty:
                    combined_data = temp_df
                else:
                    combined_data = pd.merge(combined_data, temp_df, on='time', how='outer')
        if not combined_data.empty:
            combined_data = combined_data.sort_values('time')
        return combined_data

    @staticmethod
    def combine_close_prices(data_dict):
        combined_prices = pd.DataFrame()
        for symbol, data in data_dict.items():
            if not data.empty:
                temp_df = data[['time', 'close']].copy()
                # Use just the symbol as the column name instead of symbol_close
                temp_df.rename(columns={'close': symbol}, inplace=True)
                if combined_prices.empty:
                    combined_prices = temp_df
                else:
                    combined_prices = pd.merge(combined_prices, temp_df, on='time', how='outer')
        if not combined_prices.empty:
            # Sort by date
            combined_prices = combined_prices.sort_values('time')
            # Set the time column as the index
            combined_prices.set_index('time', inplace=True)
            # Convert index to datetime if not already
            combined_prices.index = pd.to_datetime(combined_prices.index)
            # Rename the index to 'date' for clarity
            combined_prices.index.name = 'date'
        return combined_prices
