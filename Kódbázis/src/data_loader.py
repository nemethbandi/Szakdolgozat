import yfinance as yf
import pandas as pd

# Ez a modul az árfolyamadatok letöltésére és mentésére szolgál. Ehhez a yahoo finance API-t használja.

#load_data függvény letölti az árfolyamadatokat a megadott szimbólumhoz és időintervallumhoz, 
# majd visszaadja az adatokat egy pandas DataFrame-ben.

#save_data_to_csv függvény elmenti a pandas DataFrame-ben lévő adatokat egy CSV fájlba a megadott fájlnévvel.

#készítette: Németh András

def load_data(symbol, start_date, end_date):
    
    data = yf.download(symbol, start=start_date, end=end_date)
    data.columns = data.columns.levels[0]
    return data


def save_data_to_csv(data, filename):
    data.to_csv(filename)