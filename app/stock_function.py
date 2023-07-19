from loguru import logger
import traceback
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import yfinance as yf
import math


def init_stock(tickers: str, start: str, end: str):
    stock_info = yf.download(tickers, start=start, end=end)
    return stock_info

def get_local_maxima(original_price_data: pd.DataFrame, smoothed_price_data: pd.DataFrame, interval: int=5) -> pd.DataFrame:
        """
        Return table of local maximum price
        columns of table:
        - date, price: raw price, type: 'peak'

        Parameter
        -----
        original_price_data: raw price time serise (DataFrame with 1 col)
        smoothed_price_data: smoothed price time serise (DataFrame with 1 col)
        interval: window to locate peak/bottom price on raw price time serise by local extrema of smoothed price time sereise
        """
        peak_indexes = argrelextrema(smoothed_price_data.to_numpy(), np.greater)[0]
        peak_dates = []
        peak_close = []
        for index in peak_indexes: #smoothed peak index
            lower_boundary = index - interval
            if lower_boundary < 0:
                lower_boundary = 0
            upper_boundary = index + interval + 1
            if upper_boundary > len(original_price_data) - 1:
                upper_boundary = len(original_price_data)
            stock_data_in_interval = original_price_data.iloc[list(range(lower_boundary, upper_boundary))]
            peak_dates.append(stock_data_in_interval.idxmax())
            peak_close.append(stock_data_in_interval.max())
        peaks = pd.DataFrame({"price": peak_close, "type": 'peak'}, index=peak_dates)
        peaks.index.name = "date"
        peaks = peaks[~peaks.index.duplicated()]
        return peaks

def get_local_minima(original_close_data: pd.DataFrame, smoothed_price_data: pd.DataFrame, interval: int=5) -> pd.DataFrame:
    """
    Return table of local minium price
    columns of table:
    - date, price: raw price, type:'bottom'

    Parameter
    -----
    original_price_data: raw price time serise
    smoothed_price_data: smoothed price time serise
    interval: window to locate peak/bottom price on raw price time serise by local extrema of smoothed price time sereise
    """

    bottom_indexs = argrelextrema(smoothed_price_data.to_numpy(), np.less)[0]   
    bottom_dates = []
    bottom_close = []
    for index in bottom_indexs:
        lower_boundary = index - interval
        if lower_boundary < 0:
            lower_boundary = 0
        upper_boundary = index + interval + 1
        if upper_boundary > len(original_close_data) - 1:
            upper_boundary = len(original_close_data)
        stock_data_in_interval = original_close_data.iloc[list(range(lower_boundary, upper_boundary))]
        bottom_dates.append(stock_data_in_interval.idxmin())
        bottom_close.append(stock_data_in_interval.min())
    bottoms = pd.DataFrame({"price": bottom_close, "type": 'bottom'}, index=bottom_dates)
    bottoms = bottoms[~bottoms.index.duplicated()]
    bottoms.index.name = "date"
    return bottoms

def get_local_extrema(original_close_data: pd.DataFrame, smoothed_price_data: pd.DataFrame, interval: int=5) -> pd.DataFrame:
    """
    Return table of local min/max price and percentage change relative to previous local extrema
    columns of table:
    - date, price: raw price, type: peak/bottom, percentage change

    Parameter
    -----
    original_price_data: raw price time serise
    smoothed_price_data: smoothed price time serise
    interval: window to locate peak/bottom price on raw price time serise by local extrema of smoothed price time sereise

    """
    print("this is from stock")
    peaks = get_local_maxima(original_close_data, smoothed_price_data, interval)
    bottoms = get_local_minima(original_close_data, smoothed_price_data, interval)
    local_extrema = pd.concat([peaks, bottoms]).sort_index()
    
    # calculate percentage change
    percentage_change_lst =[np.nan]
    for i in range(1, len(local_extrema)):
        #print(local_extrema['price'][i])
        percentage_change = (local_extrema['price'][i]-local_extrema['price'][i-1])/local_extrema['price'][i-1]
        #print(percentage_change)
        percentage_change_lst.append(percentage_change)

    # pd.DataFrame({'percetage': percentage_change_lst})
    local_extrema['percentage change'] = percentage_change_lst


    return local_extrema


def add_column_ma(stock_data: pd.DataFrame, period: int=9, mode='ma', major_col_name='Close'):
    """
    add a column of moving average (MA) to stock_data
    
    Parameter
    -----
    - stock_data: DataFrame with column named['Close'] which is closing price of each day
    - period: time period (day)
    - mode options: moving average:'ma', exponential moving average:'ema', displaced moving average:'dma'
    """

    if(mode =='ma'):
        stock_data[f'ma{period}'] = stock_data['Close'].rolling(period).mean()
        stock_data[f'ma{period}'].dropna(inplace=True)
        
    elif mode =='dma':
        ma = stock_data['Close'].rolling(period).mean()
        ma.dropna(inplace=True)
        stock_data[f"dma{period}"] = ma.shift(math.ceil(period/2)*(-1))

    elif(mode=='ema'):
        stock_data[f'ema{period}'] = stock_data['Close'].ewm(span=period, adjust=False).mean()

    return stock_data

def smoothen(self, original_data: pd.Series, N: int=10) -> pd.DataFrame:
    """
    Return: 1-col-DataFrame of smoothen data (length differ with original data)
    Argument
    ------
    - original_data: time serise of stock price
    """
    # Smaller N -> More accurate
    # Larger N -> More smooth
    # Ref: https://books.google.com.hk/books?id=m2T9CQAAQBAJ&pg=PA189&lpg=PA189&dq=numpy+blackman+and+convolve&source=bl&ots=5lqrOE_YHL&sig=ACfU3U3onrK4g3uAo3a9FLT_3yMcQXGfKQ&hl=en&sa=X&ved=2ahUKEwjE8p-l-rbyAhVI05QKHfJnAL0Q6AF6BAgQEAM#v=onepage&q=numpy%20blackman%20and%20convolve&f=false
    window = np.blackman(N)
    smoothed_data = np.convolve(window / window.sum(), original_data, mode="same")
    smoothed_data = pd.DataFrame(smoothed_data, index=original_data.index, columns=["Data"])

    return smoothed_data
