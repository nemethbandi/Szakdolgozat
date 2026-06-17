import yfinance as yf
import pandas as pd

def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    data = yf.download(symbol, start=start_date, end=end_date)
   
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data


def save_data_to_csv(data: pd.DataFrame, filename: str) -> None:
    
    data.to_csv(filename)