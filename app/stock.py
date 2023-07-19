from loguru import logger
import traceback
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import yfinance as yf


class Stock:
    def __init__(self, tickers: str, start: str, end: str):
        # Load stock info
        if tickers != '':
            self.stock_info = yf.download(tickers, start=start, end=end)
        self.smooth_data_N = 10
        self.find_extrema_interval = 5

    def get_stock_close_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.stock_info["Close"])

    def smoothen(self, close_data: pd.Series, N: int=10) -> pd.DataFrame:
        # Smaller N -> More accurate
        # Larger N -> More smooth
        # Ref: https://books.google.com.hk/books?id=m2T9CQAAQBAJ&pg=PA189&lpg=PA189&dq=numpy+blackman+and+convolve&source=bl&ots=5lqrOE_YHL&sig=ACfU3U3onrK4g3uAo3a9FLT_3yMcQXGfKQ&hl=en&sa=X&ved=2ahUKEwjE8p-l-rbyAhVI05QKHfJnAL0Q6AF6BAgQEAM#v=onepage&q=numpy%20blackman%20and%20convolve&f=false
        window = np.blackman(N)
        smoothed_data = np.convolve(window / window.sum(), close_data, mode="same")
        smoothed_data = pd.DataFrame(smoothed_data, index=close_data.index, columns=["Data"])

        return smoothed_data

    def get_local_maxima(self, original_price_data: pd.Series, smoothed_price_data: pd.Series, interval: int=5) -> pd.DataFrame:
        """
        Return table of local maximum price
        columns of table:
        - date, price: raw price, type: 'peak'

        Parameter
        -----
        original_price_data: raw price time serise
        smoothed_price_data: smoothed price time serise
        interval: window to locate peak/bottom price on raw price time serise by local extrema of smoothed price time sereise
        """
        peak_indexs = argrelextrema(smoothed_price_data["Data"].to_numpy(), np.greater)[0]

        peaks = original_price_data.iloc[peak_indexs]["Close"]
        peaks = pd.DataFrame(peaks)

        peak_dates = []
        peak_close = []
        for index in peak_indexs:
            lower_boundary = index - interval
            if lower_boundary < 0:
                lower_boundary = 0
            upper_boundary = index + interval + 1
            if upper_boundary > len(original_price_data) - 1:
                upper_boundary = len(original_price_data)
            stock_data_in_interval = original_price_data.iloc[list(range(lower_boundary, upper_boundary))]
            peak_dates.append(stock_data_in_interval["Close"].idxmax())
            peak_close.append(stock_data_in_interval["Close"].max())
        peaks = pd.DataFrame({"Close": peak_close, "Type": "Peak"}, index=peak_dates)
        peaks.index.name = "Date"
        return peaks

    def get_local_minima(self, original_close_data: pd.Series, smoothed_data: pd.Series, interval: int=5) -> pd.DataFrame:
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

        bottom_indexs = argrelextrema(smoothed_data["Data"].to_numpy(), np.less)[0]
        bottoms = original_close_data.iloc[bottom_indexs]["Close"]
        bottoms = pd.DataFrame(bottoms)
        
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
            bottom_dates.append(stock_data_in_interval["Close"].idxmin())
            bottom_close.append(stock_data_in_interval["Close"].min())
        bottoms = pd.DataFrame({"Close": bottom_close, "Type": "Bottom"}, index=bottom_dates)
        bottoms.index.name = "Date"
        return bottoms
    
    def get_local_extrema(self, original_close_data: pd.Series, smoothed_data: pd.Series, interval: int=5) -> pd.DataFrame:
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
        peaks = self.get_local_maxima(original_close_data, smoothed_data, interval)
        bottoms = self.get_local_minima(original_close_data, smoothed_data, interval)
        extrema = pd.concat([peaks, bottoms])
        return extrema.sort_index()

    def list_close_stock_peak_bottom_info(self):
        result = []
        stock_close_data = self.get_stock_close_data()
        smoothed_close_data = self.smoothen(stock_close_data["Close"], N=self.smooth_data_N)
        stock_close_data = stock_close_data[self.smooth_data_N:-self.smooth_data_N]
        smoothed_close_data = smoothed_close_data[self.smooth_data_N:-self.smooth_data_N]

        peaks_and_bottoms = self.get_extrema(stock_close_data, smoothed_close_data, interval=self.find_extrema_interval)
        for date, info in peaks_and_bottoms.iterrows():
            extrema_info = {}
            extrema_info.update({"date": date.strftime("%Y-%m-%d"), "close": info["Close"], "signal": info["Type"], "different_against_previous": None})
            # Calculate different against previous if this is not the first item.
            if result:
                extrema_info.update({"different_against_previous": (info["Close"] - last_close_value) / last_close_value})
            result.append(extrema_info)
            last_close_value = info["Close"]
        return result

    def plot_close_stock_peak_bottom_info(self):
        stock_close_data = self.get_stock_close_data()
        smoothed_close_data = self.smoothen(stock_close_data["Close"], N=self.smooth_data_N)
        stock_close_data = stock_close_data[self.smooth_data_N:-self.smooth_data_N]
        smoothed_close_data = smoothed_close_data[self.smooth_data_N:-self.smooth_data_N]

        peaks_and_bottoms = self.get_extrema(stock_close_data, smoothed_close_data, interval=self.find_extrema_interval)

        plt.plot(stock_close_data, label="Closing price", color="slategrey")
        plt.plot(smoothed_close_data, label="Smoothed data", color="darkturquoise")
        plt.plot(peaks_and_bottoms[peaks_and_bottoms["Type"]=="Peak"]["Close"], "x", color="forestgreen")
        plt.plot(peaks_and_bottoms[peaks_and_bottoms["Type"]=="Bottom"]["Close"], "x", color="red")
        for date, value in peaks_and_bottoms.iterrows():
            plt.annotate(date.strftime("%Y-%m-%d") + ", {:.2f}".format(value["Close"]), (date, value["Close"]))
        plt.legend()
        plt.show()


    
if __name__ == "__main__":
    stock = Stock("NVDA", start="2021-01-01", end="2021-08-16")
    logger.debug(stock.list_close_stock_peak_bottom_info())
    stock.plot_close_stock_peak_bottom_info()