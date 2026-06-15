import yfinance as yf
import pandas as pd

def load_data(symbol, start_date, end_date):
    
    data = yf.download(symbol, start=start_date, end=end_date)
    data.columns = data.columns.levels[0]
    return data


def save_data_to_csv(data, filename):
    data.to_csv(filename)