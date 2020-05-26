import requests
import pandas as pd
from ...models import StockMetaData, Ticker


def get_data(symbol, api_key='H579PUKW0SVGRIK9'):
    print('fetching data ...')
    r = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='
                     + symbol + '&interval=5min'+'&apikey=' + api_key)

    data = r.json()

    return data


def manage_metadata(metadata):
    obj, created = StockMetaData.objects.update_or_create(
        Symbol=metadata['Meta Data']['2. Symbol'],
        defaults={
            "Symbol": metadata['Meta Data']['2. Symbol'],
            "Interval": metadata['Meta Data']['4. Interval'],
            "Last_Refreshed": metadata['Meta Data']['3. Last Refreshed']
            })
    for element in metadata['Time Series (5min)']:

        Ticker.objects.update_or_create(
            date=element,
            defaults={
                "date": element,
                "open": metadata['Time Series (5min)'][element]['1. open'],
                "high": metadata['Time Series (5min)'][element]['2. high'],
                "low": metadata['Time Series (5min)'][element]['3. low'],
                "close": metadata['Time Series (5min)'][element]['4. close'],
                "volume": metadata['Time Series (5min)'][element]['5. volume'],
            }, stock=obj)


def convert_to_df(data):
    """Convert the result JSON in pandas dataframe"""

    df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')

    df = df.reset_index()

    # Rename columns

    df = df.rename(index=str, columns={"index": "Date", "1. open": "Open", "2. high": "High", "3. low": "Low",
                                       "4. close": "Close", "5. volume": "Volume"})

    # Change to datetime

    df['Date'] = pd.to_datetime(df['Date'])

    # Sort data according to date

    df = df.sort_values(by=['Date'])

    # Change the datatype

    df.Open = df.Open.astype(float)
    df.Close = df.Close.astype(float)
    df.High = df.High.astype(float)
    df.Low = df.Low.astype(float)
    df.Volume = df.Volume.astype(float)

    # Checks
    df.head()
    df.info()
    # df.to_csv('data.csv')

    return df
