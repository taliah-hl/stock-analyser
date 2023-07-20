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


def add_column_ma(stock_data: pd.DataFrame, period: int=9, mode: str='ma', price_col_name: str='Close'):
    """
    add a column of moving average (MA) to stock_data
    
    Parameter
    -----
    - stock_data: DataFrame with column named['Close'] which is closing price of each day
    - period: time period (day)
    - mode options: moving average:'ma', exponential moving average:'ema', displaced moving average:'dma'
    - price_col_name: name of column of original stock price
    """

    if(mode =='ma'):
        stock_data[f'ma{period}'] = stock_data[f'{price_col_name}'].rolling(period).mean()
        stock_data[f'ma{period}'].dropna(inplace=True)
        
    elif mode =='dma':
        ma = stock_data[f'{price_col_name}'].rolling(period).mean()
        ma.dropna(inplace=True)
        stock_data[f"dma{period}"] = ma.shift(math.ceil(period/2)*(-1))

    elif(mode=='ema'):
        stock_data[f'ema{period}'] = stock_data[f'{price_col_name}'].ewm(span=period, adjust=False).mean()

    return stock_data

def smoothen(original_data: pd.Series, N: int=10) -> pd.DataFrame:
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

def plotX(
        org_price: pd.DataFrame, 
        extrema: pd.DataFrame, 
        plt_title: str='Extrema',
        curves: list=[], 
        op_col_name: str='Close',
        extrema_col_name: str='price', 
        percent_col_name: str='percentage change',
        
       ):
    
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(org_price[f'{op_col_name}'], label='close price', color='lightseagreen')
    plt.plot(extrema[f'{extrema_col_name}'], "x", color='red')

    for item in curves:    
        plt.plot(item, 
                 label=item.name if isinstance(item.name, str) else '',
                 alpha=0.9)
    for date, extrema, percent in zip(extrema.index, extrema[f'{extrema_col_name}'], extrema[f'{percent_col_name}']):
        plt.annotate("{:.2f}".format(extrema)
                 + ", {:.2%}".format(percent), (date, extrema), fontsize=6)
    plt.legend()
    plt.grid(which='major', color='lavender')
    plt.grid(which='minor', color='lavender')
    plt.title(plt_title)
    plt.show()


def runner(tickers: str, start: str, end: str, ma_mode: str, ma_T: int, smooth: bool=False):
    """
    run this function to download stock, plot extrema with smoothed DMA9

    """
    stock_info = yf.download(tickers, start=start, end=end)
    stock_data = pd.DataFrame(stock_info["Close"])
    stock_data = add_column_ma(stock_data, ma_T, ma_mode) # can amend
    if smooth:
        stock_data[f"{ma_mode}{ma_T}"] = smoothen(stock_data[f"{ma_mode}{ma_T}"])
    local_extrema = get_local_extrema(stock_data['Close'], stock_data[f"{ma_mode}{ma_T}"])
    plotX(stock_data, local_extrema,f"{tickers} {ma_mode}{ma_T}", [stock_data[f"{ma_mode}{ma_T}"]])


if __name__ == "__main__":
    runner()
