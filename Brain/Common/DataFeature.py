import pandas as pd
import collections
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

class DataFeature():
    """
    Data preprocessing class for calculating logarithmic rate of change and performing Min-Max normalization.
    """

    def __init__(self, formal: bool = False) -> None:
        self.formal = formal
        self.PricesObject = collections.namedtuple('Prices', field_names=[
            "log_open", "log_high", "log_low", "log_close",
            "open", "high", "low", "close", "volume",
            "quote_av", "trades", "tb_base_av", "tb_quote_av"
        ])

    def get_train_net_work_data_by_pd(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Retrieve data required for neural network training, suitable for formal trading scenarios.
        """
        out_dict = {}
        self.df = df
        out_dict.update({symbol: self.load_relative()})
        return out_dict

    def get_train_net_work_data_by_path(self, symbols: list) -> dict:
        """
        Retrieve data required for neural network training, suitable when reading data from files.
        """
        out_dict = {}
        for symbol in symbols:
            df = pd.read_csv(f'Brain/simulation/data/{symbol}-F-30-Min.csv')
            df.set_index('Datetime', inplace=True)
            self.df = df
            out_dict.update({symbol: self.load_relative()})
        return out_dict

    def calculate_log_return(self, values, epsilon=1e-10):
        """
        Calculate the logarithmic rate of change with zero handling.

        Returns:
            A numpy array representing the logarithmic rate of change.
        """
        shift_data = np.roll(values, shift=1, axis=0)
        # Handle the first element to avoid division by zero or log errors
        shift_data[0] = shift_data[1] if len(shift_data) > 1 else shift_data[0]

        # Replace zeros in `values` and `shift_data` with a small epsilon value
        values = np.where(values == 0, epsilon, values)
        shift_data = np.where(shift_data == 0, epsilon, shift_data)

        # Calculate the logarithmic rate of change
        log_return = np.log(np.divide(values, shift_data))

        return log_return
    
    def load_relative(self):
        """
        Load data, calculate the logarithmic rate of change, and perform Min-Max normalization.
        """
        np_data = np.array(self.df.values, dtype=np.float32)
        
        # Extract raw data
        open = np_data[:, 0]
        high = np_data[:, 1]
        low = np_data[:, 2]
        close = np_data[:, 3]
        volume = np_data[:, 4]
        quote_av = np_data[:, 5]
        trades = np_data[:, 6]
        tb_base_av = np_data[:, 7]
        tb_quote_av = np_data[:, 8]

        # Calculate the logarithmic rate of change
        log_return_open = self.calculate_log_return(open)
        print(open.shape)
        log_return_high = self.calculate_log_return(high)
        log_return_low = self.calculate_log_return(low)
        log_return_close = self.calculate_log_return(close)
        log_return_volume = self.calculate_log_return(volume)
        log_return_quote_av = self.calculate_log_return(quote_av)
        log_return_trades = self.calculate_log_return(trades)
        log_return_tb_base_av = self.calculate_log_return(tb_base_av)
        log_return_tb_quote_av = self.calculate_log_return(tb_quote_av)

        # Combine all logarithmic rate of change data for normalization
        log_return_data = np.vstack((
            log_return_open,
            log_return_high,
            log_return_low,
            log_return_close,
            log_return_volume,
            log_return_quote_av,
            log_return_trades,
            log_return_tb_base_av,
            log_return_tb_quote_av
        )).T  # Transpose to match the sample count


        # Handle potential NaN or Inf values
        log_return_data = np.nan_to_num(log_return_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply Min-Max normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(log_return_data)
        # Split the normalized data back into individual variables
        log_return_open = normalized_data[:, 0]
        log_return_high = normalized_data[:, 1]
        log_return_low = normalized_data[:, 2]
        log_return_close = normalized_data[:, 3]
        log_return_volume = normalized_data[:, 4]
        log_return_quote_av = normalized_data[:, 5]
        log_return_trades = normalized_data[:, 6]
        log_return_tb_base_av = normalized_data[:, 7]
        log_return_tb_quote_av = normalized_data[:, 8]

        # Construct the namedtuple for the return value
        return self.PricesObject(
            open = open,
            high = high,
            low = low,
            close = close,
            log_open=log_return_open,
            log_high=log_return_high,
            log_low=log_return_low,
            log_close=log_return_close,
            volume=log_return_volume,
            quote_av=log_return_quote_av,
            trades=log_return_trades,
            tb_base_av=log_return_tb_base_av,
            tb_quote_av=log_return_tb_quote_av
        )
