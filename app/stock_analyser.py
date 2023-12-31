from loguru import logger
import traceback
import numpy as np
import warnings
import pandas as pd
from scipy.signal import argrelextrema, butter,filtfilt
from matplotlib import pyplot as plt
import yfinance as yf
import pandas_market_calendars as mcal
import math
from tabulate import tabulate
import zigzag as zz
import time
from datetime import date, timedelta
import argparse
import sys
import enum
import re
import os

class DayType(enum.Enum):
    BUYPT=1
    SELLPT=2
    BREAKPT=3
    NAN=0   

class BuyptFilter(enum.Enum):
    # buy point filters
    IN_UPTREND = 1
    CONVERGING_DROP = 2
    RISING_PEAK = 3
    MA_SHORT_ABOVE_LONG = 4
    RSI_OVER =5

    ### ADD FILTER EXAMPLE
    SOME_FILTER = 99

class StockAnalyser():

    ## CONSTANT ##
    PEAK =1
    BOTTOM =-1
    UPTRD =1
    DOWNTRD =-1


    def __init__(self):

        self.tickers= None
        self.start = None
        self.pre_start=None # start day to calculate longest required MA
        self.close_price = None # close price start from pre_start
        self.actual_start_inx = None    # idx of first business day after user specified start day
        self.end = None
        self.stock_data = None
        self.smooth_data_N = 10
        self.find_extrema_interval = 5
        self.peaks = None
        self.bottoms = None
        self.extrema = None
        self.smoothen_price = None
        self.all_vertex= None
        self.peak_indexes=[]
        self.bottom_indexes=[]
        self.buypt_dates=[]
        self.zzthres=0.09

    def next_trading_day(self, date)->pd.Timestamp:

        date = pd.to_datetime(date)
        nyse = mcal.get_calendar('NYSE')
        end = date + timedelta(5)
        days = nyse.valid_days(start_date=date, end_date=end)
        return days[0].tz_localize(None)

    def download(self, tickers: str, start: str, end: str, pre_start: str=None)->pd.DataFrame:
        """
        parameter
        -----
        `tickers`: stock ticker | `start`: user specified start date yyyy-mm-dd | `end`: end date | `pre_start`: actual date start download stock data for caculate MA
        
        return 

        self.stock_data

        ---
        """
        # Load stock info
        if pre_start is not None:
            stock_info = yf.download(tickers, start=pre_start, end=end)
        else:
            stock_info = yf.download(tickers, start=start, end=end)
        self.tickers=tickers
        self.start=start
        actual_start = self.next_trading_day(start)
        self.end = end
        self.close_price =  pd.DataFrame(stock_info['Close']).rename(columns={'Close':'close'})


        self.actual_start_inx = self.close_price.index.get_loc(actual_start)
        self.stock_data = pd.DataFrame(self.close_price[self.actual_start_inx:], index=self.close_price.index[self.actual_start_inx:])
        
        self.data_len = len(self.stock_data)
       


        ## CONSTANT ##
        # PEAK =1 | BOTTOM =-1  | UPTREND =1 | DOWNTREND =-1
        self.SCATTER_MARKER_SIZE=1/self.data_len*6000

        return self.stock_data
    
    
    def get_close_price(self) -> pd.DataFrame:
        """
        Return: DataFrame of colsing price with date
        """
        return pd.DataFrame(self.stock_data['close'])
    
    def get_stock_data(self)-> pd.DataFrame:
        """
        get self.stock_data
        """
        return pd.DataFrame(self.stock_data)
    
    def print_stock_data(self, file_name: str='', writeToTxt: bool=False)->None:
        """
        pretty print self.stock_data
        writeToTxt: option to write table to `file_name`.txt
        """
        
        logger.debug(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=("", ".2f", "g",".2%", "g", "g", "g",".2f", ".2f", ".2f",".2f",".4f",".4f","g")))
        if writeToTxt and file_name=='':
            file_name = f"../../{self.ticker_name}.txt"

        if writeToTxt:
            with open(file_name, 'w') as fio:
                fio.write(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=("", ".2f", "g",".2%", "g", "g", "g",".2f", ".2f", ".2f",".2f",".4f",".4f","g")))
            logger.info(f"stock_data wrote to {file_name}")
    
    def stock_data_to_csv(self, csv_dir: str=None)->None:
        """
        print self.stock_data to csv

        parameter
        ----

        `csv_dir`: directory to save csv file

        return

        None
        """

        if csv_dir is None:
            csv_dir='../../back_test_result/'
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)

        add_num =1
        csv_dir += f"/stock_data_{self.tickers}_{self.start}_{self.end}"
        save_path_norepeat = csv_dir
        while os.path.isfile(f'{save_path_norepeat}.csv'):
            save_path_norepeat = csv_dir + f'_{add_num}'
            add_num +=1

        save_path_norepeat += '.csv'
        self.stock_data.to_csv(save_path_norepeat)
        logger.info(f"csv saved to {save_path_norepeat}")

    def get_peaks(self)-> pd.DataFrame:
        """
        Return: DataFrame of peaks with date
         
        TO BE IMPLEMENT
        """
        
    
    def get_bottoms(self)-> pd.DataFrame:
        """
        Return: DataFrame of bottoms with date
        
        TO BE IMPLEMENT
        """
    
    def get_peak_idx_lst(self)-> list:
        """
        return: list of index of all peaks
        """
        return self.peak_indexes
    
    def get_bottom_idx_lst(self)->list:
        """
        return: list of index of all bottoms
        """
        return self.bottom_indexes
    
    def get_extrema(self)-> pd.DataFrame:
        """
        Return: DataFrame of extrema with date
        """
        return pd.DataFrame(self.extrema)
    
    def get_col(self, col_name: str)->pd.Series:
        """
        return self.stock_data[col_nmae]
        """
        return self.stock_data[col_name]

    def get_all_col_name(self)->list:
        """
        return list of str of all col in self.stock_data
        """
        return self.stock_data.columns.tolist()
    
    
    def add_column_ma(self,  src_data: pd.Series=None, mode: str='ma', period: int=9)->pd.Series:
        """
        add a column of moving average (MA) to stock_data
        return: ma

        Parameter
        -----
        - src_data: pd.Series of source stock price to cal ma
        - period: time period (day)
        - mode options: moving average:'ma', exponential moving average:'ema', displaced moving average:'dma', linear-weighted ma: 'lwma'
        - 
        """

        if f'{mode}{period}' in self.stock_data:
            return self.stock_data[f'{mode}{period}']
        
        if src_data is None:
            src_data = self.close_price
        
        DMA_DISPLACEMEN = math.floor(period/2)*(-1)

        if(mode =='ma'):
            self.stock_data[f'ma{period}'] = np.nan
            ma = self.close_price.rolling(period).mean()
            self.stock_data[f'ma{period}'] = ma[self.actual_start_inx:]
            self.stock_data[f'ma{period}'].dropna(inplace=True)
            return ma
            
        elif mode =='dma':
            
            ma = self.close_price.rolling(period).mean()[self.actual_start_inx:]
            ma.dropna(inplace=True)
            self.stock_data[f"dma{period}"] = ma.shift(DMA_DISPLACEMEN)
            return self.stock_data[f"dma{period}"]

        elif(mode=='ema'):
            ma = self.close_price.ewm(span=period, adjust=False).mean()
            self.stock_data[f'ema{period}'] = ma[self.actual_start_inx:]
            return ma
        
        elif(mode=='lwma'):
             
            #Result not good
            DMA_DISPLACEMEN = math.floor(period/4)*(-1)
            weights = np.arange(1, period + 1)
            lwma = self.close_price.rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)[self.start:]
            lwma.dropna(inplace=True)
            self.stock_data[f"lwma{period}"] = lwma.shift(DMA_DISPLACEMEN)
            return self.stock_data[f"lwma{period}"]
            
        else:
            raise Exception("ma mode not given or wrong!")
        return


        
    
    def add_col_slope(self, src_data)->pd.Series:
        """
        calculate slope of segment of given pd.Series
        - src_data: source of data to calculate slope pd.Series

        return 

        pd.Serise showing slope of src_data

        """
        slope_lst=[np.nan]
        for i in range(1, len(src_data)):
            if(src_data[i-1]==0 or src_data[i]==0):
                slope_lst.append(np.nan)
            else:
                slope_lst.append(src_data[i] - src_data[i-1])
        self.stock_data[f'slope {src_data.name}'] = slope_lst
        return self.stock_data[f'slope {src_data.name}']

    def __add_col_macd_group(self)->None:
        """"
        add column of ema12, ema26, macd, signal line and slope of macd
        no return
        """
        pre_ma12 = self.close_price.ewm(span=12, adjust=False).mean()
        pre_ma26 = self.close_price.ewm(span=26, adjust=False).mean()
        self.add_column_ma(None, 'ema', 12)
        self.add_column_ma(None, 'ema', 26)
        self.macd = pd.DataFrame(pre_ma12 - pre_ma26, index=pre_ma12.index)
        signal = self.macd.ewm(span=9, adjust=False).mean()
        self.stock_data['MACD']=self.stock_data['ema12'] - self.stock_data['ema26']
        self.stock_data['signal'] = signal[self.actual_start_inx:]
        self.add_col_slope(self.stock_data['MACD'])
        self.add_col_slope(self.stock_data['signal'])

    
    def __add_col_macd(self, src_data: pd.Series)->None:
        """"
        add column of ema12, ema26, macd, signal line and slope of macd
        no return
        """
        if 'MACD' in self.stock_data:
            return self.stock_data['MACD']
        else:
            self.add_column_ma(src_data, 'ema', 12)
            self.add_column_ma(src_data,'ema', 26)
            self.stock_data['MACD']=self.stock_data['ema12'] - self.stock_data['ema26']
            self.stock_data['signal'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
            self.add_col_slope(self.stock_data['MACD'])
            return self.stock_data['MACD']

    

    
    def set_butter(self, filter_period: int, src_data: pd.Series)->pd.Series:
        """
        filter frequency smaller than filter_period by Butterworth Low Pass Filter
        result is put into self.stock_data['buttered {src_col}']

        parameter
        ---
        - src_data: source data to be filtered
        - filter_period: smallest period to keep (vibration of smaller period will be filtered out)

        return:

        filtered sequence

        exmaple: filter fluctuation within 10days=> set filter_period=10
        ref: https://nehajirafe.medium.com/using-fft-to-analyse-and-cleanse-time-series-data-d0c793bb82e3
        """
        cutoff = 1/filter_period
        fs = 1.0 # since frequency of sampling data is fix (1 point per day), just fix fs=1/0
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2 # 2 mean sth like assume stock price is a function of order 2 (quardratic/2nd order polynomial)
        normalized_cutoff = cutoff / nyq
        b_coeff, a_coeff = butter(order, normalized_cutoff, btype='low', analog=False)
        # b, a are coefficients in the formula
        # H(z) = (b0 + b1 * z^(-1) + b2 * z^(-2) + ... + bM * z^(-M)) / (1 + a1 * z^(-1) + a2 * z^(-2) + ... + aN * z^(-N))
        self.stock_data[f'buttered {src_data.name} T={filter_period}'] = filtfilt( b_coeff, a_coeff, src_data)
        return self.stock_data[f'buttered {src_data.name} T={filter_period}']


    
    def set_smoothen_price_blackman(self, src_data: pd.Series, N: int=10)->pd.Series:
        """
        smoothen ['col_name'] of self.stock_data
        - set fcuntion of self.smoothen_price
        
        return
        -----
        smoothed price

        Parameter
        -----
        N: extend of smoothening. smaller->More accurate; larger -> more smooth
        """
        window = np.blackman(N)
        smoothed_data = np.convolve(window / window.sum(), src_data, mode="same")
        smoothed_data_chop = smoothed_data[1:-1]
        #exclude last and first raw
        self.smoothen_price = pd.DataFrame(smoothed_data_chop, index=src_data.index[1:-1], columns=["Data"])
        return self.smoothen_price

    
    def set_smoothen_price_polyfit(self, src_data: pd.Series)->pd.Series:         
        """
        smoothen ['col_name'] of self.stock_data by polyfit (mutating)
        not work to smooth ma potentially due to NaN value
        - set fcuntion of self.smoothen_price

        Parameter
        -----
        N: extend of smoothening. smaller->More accurate; larger -> more smooth

        return
        -----
        smoothed price
        """
        degree = 10

    

        X = np.array(src_data.reset_index().index)
        Y = src_data.to_numpy()

        
            
        poly_fit = np.poly1d(np.polyfit(X, Y, degree))

        self.smoothen_price = pd.DataFrame(poly_fit(X), columns=["Data"], index=self.stock_data.index)
        return self.smoothen_price
    


    def smoothen_non_mutate(self, original_data: pd.Series, N: int=10) -> pd.DataFrame:     #can be delete later
        """
        Return: 1-col-DataFrame of smoothen data (length differ with original data)
        won't save to self
        Parameter
        ------
        - original_data: time Series of stock price

        return
        -----
        smoothed price
        """
        # Smaller N -> More accurate
        # Larger N -> More smooth
        # Ref: https://books.google.com.hk/books?id=m2T9CQAAQBAJ&pg=PA189&lpg=PA189&dq=numpy+blackman+and+convolve&source=bl&ots=5lqrOE_YHL&sig=ACfU3U3onrK4g3uAo3a9FLT_3yMcQXGfKQ&hl=en&sa=X&ved=2ahUKEwjE8p-l-rbyAhVI05QKHfJnAL0Q6AF6BAgQEAM#v=onepage&q=numpy%20blackman%20and%20convolve&f=false
        window = np.blackman(N)
        smoothed_data = np.convolve(window / window.sum(), original_data, mode="same")
        smoothed_data = pd.DataFrame(smoothed_data, index=original_data.index, columns=["Data"])

        return smoothed_data
    
    def add_col_all_vertex(self, src_data: pd.Series)-> None:
        """
        locate all vertex of stock price by comparing with prev 1 and back 1 data 
        set function of self.vertex, type (from vertex) and p-b change (from vertex) column of stock_data
        - point that is higher than the prev 1 point and back 1 point is defined as peak, 
        - point that is lower than the prev 1 point and back 1 point is defined as bottom 

        return : None
        """
        peaks_lst=[]
        peak_dates=[]
        for i in range(1, self.data_len-1):
            if (src_data[i] > src_data[i-1] ) & (src_data[i] > src_data[i+1]):
                peaks_lst.append(src_data[i])
                peak_dates.append(src_data.index[i])
        bottoms_lst=[]
        bottom_dates=[]
        for i in range(1, self.data_len-1):
            if (src_data[i] < src_data[i-1] ) & (src_data[i] < src_data[i+1]):
                bottoms_lst.append(src_data[i])
                bottom_dates.append(src_data.index[i])

        peaks = pd.DataFrame({'close': peaks_lst, "type": self.PEAK}, index=peak_dates)
        bottoms = pd.DataFrame({'close': bottoms_lst, "type": self.BOTTOM}, index=bottom_dates)
      
        #self.all_vertex = pd.concat([peaks, bottoms]).sort_index()
        self.vertex = pd.concat([peaks, bottoms]).sort_index()
        self.stock_data['type (from vertex)'] = self.vertex['type']
        self.stock_data['p-b change (from vertex)'] = self.vertex['percentage change']
    
    
 


            

    def set_extrema(self, src_data: pd.Series, close_price: pd.Series, interval: int=0, window_dir: str='left', stock: str='')->None:
        """
        set function of self.extrema, self.peak_indexes, self.bottom_indexes
        
        Parameter
        ---------
        - src_data: col name of source to calculate extrema
        - close_price: pd.Series of close price
        - interval: window to locate peak/bottom price on original price by source price
        - window_dir: 'left' : only create window to left hand side

        """


        self.bottom_indexs = argrelextrema(src_data.to_numpy(), np.less)[0]
        self.peak_indexes = argrelextrema(src_data.to_numpy(), np.greater)[0]
            
        extrema_idx_lst=[]
        for i in self.bottom_indexs:
            extrema_idx_lst.append((i, 0))  # 0 =bottom
        
        for i in self.peak_indexes:
            extrema_idx_lst.append((i, 1))  #1=peak

        extrema_idx_lst.sort()
        extrema_dates = []
        extrema_close=[]


        ## check does peak-bottom appear alternatingly

        for i in range(1, len(extrema_idx_lst)):
            if extrema_idx_lst[i][1] == extrema_idx_lst[i-1][1]: # 2 consecutive peak or bottom

                
                if extrema_idx_lst[i][1] == self.PEAK:
                    btm = float('inf')
                    btm_idx=0
                    for j in range(extrema_idx_lst[i-1][0], extrema_idx_lst[i][0]):
                        if close_price[j] < btm:
                            btm = close_price[j]
                            btm_idx = j

                    extrema_idx_lst.insert(i, (btm_idx, self.BOTTOM))

                else: # 2 bottoms
                    pk = float('-inf')
                    pk_idx=0
                    for j in range(extrema_idx_lst[i-1][0], extrema_idx_lst[i][0]):
                        if close_price[j] > pk:
                            pk = close_price[j]
                            pk_idx = j

                    extrema_idx_lst.insert(i, (pk_idx, self.PEAK))

        
        prev_idx = 0

        for i in range(0, len(extrema_idx_lst)):
        
            lower_boundary = max(0, extrema_idx_lst[i-1][0] if i>0 else 0, extrema_idx_lst[i][0]-interval)
            if window_dir=='left':
                upper_boundary = min(extrema_idx_lst[i][0] + 1,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(close_price))

            else :
                upper_boundary = min(extrema_idx_lst[i][0] + 1 +interval,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(close_price))
            stock_data_in_interval = close_price.iloc[list(range(lower_boundary, upper_boundary))]
            
            extrema_dates.append(stock_data_in_interval.idxmax() if extrema_idx_lst[i][1] else stock_data_in_interval.idxmin())
            extrema_close.append((stock_data_in_interval.max(),self.PEAK) if extrema_idx_lst[i][1] else (stock_data_in_interval.min(), self.BOTTOM))


        self.extrema = pd.DataFrame(extrema_close, columns=['close', 'type'], index=extrema_dates)

        self.extrema = self.extrema[~self.extrema.index.duplicated()]
        self.extrema.index.name = "date"


        percentage_change_lst =[np.nan]
        for i in range(1, len(self.extrema)):
            percentage_change = (self.extrema['close'][i]-self.extrema['close'][i-1])/self.extrema['close'][i-1]
            percentage_change_lst.append(percentage_change)

        self.extrema['percentage change'] = percentage_change_lst

        self.stock_data['type'] = self.extrema['type'].astype(int)
        #self.stock_data['type']=self.stock_data['type'].astype(np.int64)

        self.stock_data['p-b change'] = self.extrema['percentage change']
        # calculate peak-to-bottom-time
        self.stock_data['bar'] = np.nan
        self.extrema['bar'] = np.nan
        self.extrema['back to peak time'] = np.nan

        

        # df.iloc[row_num, col_num]
        # col
        
        # verify does peak-bottom appear alternatingly


        for i in range(1, len(self.extrema)):
            
            try:

                if self.extrema['type'][i] ==self.BOTTOM:
                    try:
                        
                        assert self.extrema['type'][i-1]==self.PEAK
                    except AssertionError:
                        logger.warning("peak bottom does not appear alternatively, possible wrong setting")
            except IndexError as err:
                    logger.warning(err)
                    logger.warning("possibly because day range too short to get local extrema\nProgram Exit")
                    exit(1)

        # calculate peak-to-bottom change

        

        
        bar_col = self.stock_data.columns.get_loc('bar')
        bar_excol = self.extrema.columns.get_loc('bar')
        disp=0
        idx=0
        for i in range(0, self.data_len):
            if self.stock_data['type'][i] ==self.BOTTOM or self.stock_data['type'][i] ==self.PEAK:
                disp +=1
                if idx ==0: # first extrema => can't cal bar
                    self.stock_data.iloc[i, bar_col] = 0
                    self.extrema.iloc[idx, bar_excol] = 0
                else:
                    self.stock_data.iloc[i, bar_col] = disp if disp>0 else np.nan  
                    self.extrema.iloc[idx, bar_excol] = disp if disp>0 else np.nan              
                disp = 0
                idx+=1
            else:
                disp +=1
                # find prev peak date
                   
            
        logger.debug("set extrema done")




    def set_zigizag_trend(self, src_data: pd.Series, upthres: float=0.09, downthres: float=0.09) -> None:
        """
        set function of column 'zigzag', 'zz trend' of self.stock_data |
        - col 'zigzag': value of zigzag indicator, 1=peak, 0=bottom
        - col 'zz trend': up/down trend according to zigzag indicator -1=downtrend 1=uptrend

        Parameter 
        ---
        - `upthres`: up threshold of zigzag indicator
        - `downthres`: down threshold of zigzag indicator


        source of zigzag indicator: https://pypi.org/project/zigzag/

        """
        self.zzthres = upthres
        self.stock_data['zigzag'] = np.nan
        self.stock_data['zigzag'] = zz.peak_valley_pivots(src_data, upthres, -downthres)
        self.stock_data.iloc[-1, self.stock_data.columns.get_loc('zigzag')] = (-1)* self.stock_data[self.stock_data['zigzag'] !=0]['zigzag'][-2]
        # correct the problem that last point of zigzag is flipped sometime
        self.extrema['zigzag'] = self.stock_data[self.stock_data['zigzag'] !=0]['zigzag']
        logger.debug("set zigzag done")

        self.stock_data['zz trend'] = np.nan
        cur_trend=0
        trend_col= self.stock_data.columns.get_loc('zz trend')
        for i in range(1, self.data_len):
            cur_trend= self.stock_data['zigzag'][i-1] *(-1)
            if cur_trend:
                self.stock_data.iloc[i, trend_col] = cur_trend
            else:
                self.stock_data.iloc[i, trend_col] = self.stock_data['zz trend'][i-1]
        logger.debug("set trend done")

    def _add_col_zzuptrend_detected(self, upthres: float=0.09, downthres: float=0.09)->pd.DataFrame:
        """
        
        - set function of column 'zz uptrend detected' of self.stock_data

        for rows in stock data in up trend of zigzag indicator, check if that datte has acutally rise > upthres from previous bottom
        - purpose: minimize repainting effect of zigzag indicator (try to see if time roll back to that day, will up trend still be detected)

        parameter
        --
        - `upthres`: up threshold of zigzag indicator
        return: self.stock_data['zz uptrend detected']
        """
        try:
            assert 'zigzag' in self.stock_data
        except AssertionError:
            logger.error("stock_data doesn't contain column: \'zigzag\', cannot run is_zz_uptrend()")
            return None
        
        
        self.stock_data['zz uptrend detected']=False
        col_d=self.stock_data.columns.get_loc('zz uptrend detected')
        cur_big_btm = float('inf')

        for i in range(0, self.data_len-1):
            if self.stock_data['zigzag'][i]==1:
                cur_big_btm = self.stock_data['close'][i]
            if (self.stock_data['type'][i] == -1 
                and self.stock_data['zz trend'][i]==1 
                and self.stock_data['close'][i] > cur_big_btm*(1+upthres)):
                self.stock_data.iloc[i, col_d] = True
        return self.stock_data['zz uptrend detected']
    
    def __is_uptrend(self, row: int, trend_col_name: str=None, zz_thres: float=0.09)->bool:
        """
        return
        ---
        whether that row of self.stock_data is in uptrend
        - return True if trend_col_name not specified (assume no need to filter out downtrend)
        
        Parameter
        ---
        - `trend_col_name`: name of col in self.stock_data to use as source to calculate trend , with value >0 indicate uptrend, < 0 indicate downtrend
        - `upthres`: up threshold of zigzag indicator (only need if `trend_col_name` is 'zigzag')

        - `row`: which row of self.stock_data
        
        """
        if trend_col_name == 'zigzag':
            if 'zz uptrend detected' not in self.stock_data:
                self._add_col_zzuptrend_detected(zz_thres)

            return self.stock_data['zz uptrend detected'][row]
        if trend_col_name is None:
            return True
        return self.stock_data[trend_col_name][row] > 0
    


    def __get_conv_drop_rise_peak_list(self, conv_drop_filter: bool, rise_peak_filter: bool, trend_col_name: str=None, zzupthres: float=0.09)->list:
        
        """
        return: list of index of converging bottom and/or rising peak
        
        parameter

        - `conv_drop_filter`:  whether to check peak has drop < previos peak, true=to check, false=don't check
        - `rise_peak_filter`: whether to check rise above previous peak before next peak, true=to check, false=don't check 

        
        """
        try:
            assert 'type' in self.stock_data
            assert 'p-b change' in self.stock_data
            assert 'close' in self.stock_data
        except AssertionError:
            logger.error("self.stock_data must contain column: \'type\', \'p-b change\' \nprogram exit")
            exit(1)
            

        

        POS_INF = float('inf')
        prev_pbc = POS_INF
        prev_peak = POS_INF
        star_lst=[]
        i=0
        while i+1 < self.data_len-1:
            next_peak_offset=0
            # if in uptrend and is small bottom
            if self.__is_uptrend(i, trend_col_name, zzupthres) and self.stock_data['type'][i] ==-1 :
                l =0
                while self.stock_data['type'][i+l] !=1:
                    if i+l-1 >=0:
                        l-=1
                    else:
                        break
                prev_peak = self.stock_data['close'][i+l] if l !=0 else prev_peak

                rise_back_flag = False
                break_pt_found_flag = False

                potential_bp = POS_INF

                
                rise_back_offset=0
                k =0
                while self.__is_uptrend(i+k, trend_col_name, zzupthres):
                    if self.stock_data['close'][i+k] >= prev_peak: # record closest date rise back to prev peak
                        if not rise_back_flag:
                            rise_back_offset = k
                            rise_back_flag = True
                            potential_bp = i+rise_back_offset
                            break
                    if self.stock_data['type'][i] == 1:
                        next_peak_offset=k
                        potential_bp = i+next_peak_offset
                        break
                    k +=1
                    if i+k+1 > self.data_len-1:
                        break
                
                if (potential_bp< POS_INF
                        and ( (not conv_drop_filter) or self.stock_data['p-b change'][i] > prev_pbc )
                        and ( (not rise_peak_filter) or rise_back_flag )
                        ):  
                    break_pt_found_flag = True
                if break_pt_found_flag:
                    star_lst.append(potential_bp)
                prev_pbc = self.stock_data['p-b change'][i]

            i+=max(1, next_peak_offset)
        return star_lst

        
    
    def __add_col_conv_drop_rise_peak(self, conv_drop_filter: bool, rise_peak_filter: bool, trend_col_name: str=None):
        lst = self.__get_conv_drop_rise_peak_list(conv_drop_filter=conv_drop_filter, 
                                                rise_peak_filter=rise_peak_filter,
                                                trend_col_name=trend_col_name)
        """
        return 

         dataframe of bool
         with True= is converging bottom and / or rising peak
        """
        
        self.stock_data['conv drop rise peak']=False
        star_col = self.stock_data.columns.get_loc('conv drop rise peak')
        for item in lst:
            self.stock_data.iloc[item, star_col]= True
        return self.stock_data['conv drop rise peak']
            


        
    def add_col_ma_above(self, stock_data: pd.DataFrame, short: int, long: int):
        """
        calculate is col ma{short} > col ma{long} of each row, make colume ma{short} above ma{long} in stock_data
        """
        
        if f'ma{short} above ma{long}' in stock_data:
            return stock_data[f'ma{short} above ma{long}']
        
        if f'ma{short}' not in stock_data or f'ma{long}' not in stock_data:
            raise Exception("ma has to be set before calculating ma above")

        stock_data[f'ma{short} above ma{long}']=np.nan
        col = stock_data.columns.get_loc(f'ma{short} above ma{long}')

        for i in range(0, len(stock_data)):
            if pd.isna(stock_data[f'ma{short}'][i]) or pd.isna(stock_data[f'ma{long}'][i]):
                continue
            if stock_data[f'ma{short}'][i] > stock_data[f'ma{long}'][i]:
                stock_data.iloc[i, col] = True
            else:
                stock_data.iloc[i, col] = False
        return stock_data[f'ma{short} above ma{long}']
    
    def add_col_rsi(self, stock_data: pd.DataFrame):
        """
        not implemented, just an empty function as example
        """
        pass

    def is_ma_above(self, stock_data: pd.DataFrame, row: int, short: list, long: list)-> bool:
        """
        return 

        is ma{long} of that row > ma{short}

        short: list of int
        long: list of int
        """

        res = True # and of all comparison of MAs 
        if len(short) != len(long):
                raise Exception("list of short is not same as list of long, cannot get is_ma_above, program exit")
                exit(1)
        
        for i in range(0, len(short)):
            res = res and stock_data[f'ma{short[i]} above ma{long[i]}'][row]
            #print(f"checking row: {stock_data.index[row]}, res={res}")
        return res

        
    
    def is_rsi_above(self, stock_data: pd.DataFrame, row: int, thres: float):
        """
        return 
        is stock_data['rsi'] of that row > thres
        """
        if 'rsi' not in stock_data:
            self.add_col_rsi(stock_data)
        return stock_data['rsi'][row] > thres


    ### ADD FILTER EXAMPLE

    ## 1. add an "add column function"
    def add_col_something(self, stock_data: pd.DataFrame)->pd.DataFrame:
        """
        EXAMPLE FUNCTION
        """
        #stock_data['something'] = something
        # return stock_data['something']

        pass
    
    ## 2. add a function to check if row of some column meet certain condition

    def is_some_condition_met(self,  stock_data: pd.DataFrame, row: int, some_arg)->bool:
        """
        EXAMPLE FUNCTION
        
        input:    - row: index of row to check
        return: bool
        """
        res: bool
        # res = stock_data['some column'][row] > 1    # just an example
        # return res
        return True     # this is only example, so return True here
        
    
    def __set_breakpoint(self, 
                       trend_col_name: str=None,
                       bpfilters: set=set(),
                       ma_short: list=None, ma_long: list=None,
                       rsi_thres: float=0, zz_thres: float=0)->None :
        """
        - set function of 'buy pt' column of self.stock_data

        Parameter 
        ---------
        required col of stock_data: | type | p-b change | zigzag (if uptrend src selected as zigzag) | {trend_col_name}
        - trend_col_name: name of col of trend in stock_data, (not required if uptrend_src=='zz'), with value >0 indicate uptrend, < 0 indicate downtrend
        - col 'type': (int/bool) mark peak as 1, bottom as 0, index as pd.dateTime
        - col 'p-b change': (float) mark peak-to-bottom percentage change at each row of bottom, index as pd.dateTime
        - col {trend_col_name}: (float), >0 indicate uptrend, < 0 indicate downtrend
        - zzupthres: required if trend_col_name=='zigzag'
        - bpfilters: set of class BuyptFilter
        """
        try:
            assert 'type' in self.stock_data
            assert 'p-b change' in self.stock_data
        except AssertionError:
            logger.error("stock_data must contain column: \'type\', \'p-b change\' \nprogram exit")
            exit(1)
        
        ## -- parameter -- ##

        incl_1st_btm = True

        
        ## -- flags -- ##

        to_find_bp_flag = True
        self.stock_data['buy pt'] = False

        if not bpfilters:
            to_find_bp_flag = False
            logger.warning("no break point filters set. no break point will be plotted")
            return self.stock_data['buy pt']
        
        ## Set up buy point filter flags
        
        ## add new filters here if required

        conv_filter = True if BuyptFilter.CONVERGING_DROP in bpfilters else False
        rp_filter = True if BuyptFilter.RISING_PEAK in bpfilters else False
        uptr_filter = True if BuyptFilter.IN_UPTREND in bpfilters else False
        sma_abv_filter = True if BuyptFilter.MA_SHORT_ABOVE_LONG in bpfilters else False
        rsi_abv_filter = True if BuyptFilter.RSI_OVER in bpfilters else False

        ### ADD FILTER EXAMPLE
        some_filter = True if BuyptFilter.SOME_FILTER in bpfilters else False


        in_uptrend_flag: bool=None
        conv_drop_flag: bool=None
        rise_peak_flag: bool=None
        sma_above_flag: bool=None
        rsi_above_flag: bool=None


        if not uptr_filter:
            in_uptrend_flag = True   
        if not conv_filter:
            conv_drop_flag = True
        if not rp_filter:
            rise_peak_flag = True
        if not sma_abv_filter:
            sma_above_flag = True
        if not rsi_abv_filter:
            rsi_above_flag = True

        buy_point_found = False
        buy_point_list=[]

        logger.debug("filter received in set breakpoint:")
        if conv_filter:
            logger.debug("converging bottom", sep='')
        if rp_filter:
            logger.debug("rising peak", sep='')
        if( sma_abv_filter):
            logger.debug("sma above", sep='')
        if uptr_filter:
            logger.debug("in up trend", sep='')
        if rsi_abv_filter:
            logger.debug("rsi above", sep='')

        ### ADD FILTER EXAMPLE
        if some_filter:
            logger.debug("some filter", sep='')
        
        # since point of converging bottom and rising peak is sparse
        # if converging bottom or rising peak filter is applied, 
        # only check those points for other filters to save time

        # check neccessary ma exist
        if len(ma_short) != len(ma_long):
            raise Exception("list of short is not same as list of long, cannot get is_ma_above, program exit")


        for i in range(0, len(ma_short)):
            if f'ma{ma_short[i]}' not in self.stock_data:
                self.add_column_ma(None, 'ma', ma_short[i])
            if f'ma{ma_long[i]}' not in self.stock_data:
                self.add_column_ma(None, 'ma', ma_long[i])

            if f'ma{ma_short[i]} above ma{ma_long[i]}' not in self.stock_data:

                self.add_col_ma_above(self.stock_data, short=ma_short[i], long=ma_long[i])

        if conv_filter or rp_filter:    
            conv_drop_rise_peak_list = self.__get_conv_drop_rise_peak_list(conv_filter, rp_filter, trend_col_name, zz_thres)
            for idx in conv_drop_rise_peak_list:

                if (   ( (not sma_abv_filter) or self.is_ma_above(self.stock_data, idx, ma_short, ma_long))
                    and ( (not rsi_abv_filter) or self.is_rsi_above(self.stock_data, idx, rsi_thres))
                    and ( (not some_filter) or self.is_some_condition_met(self.stock_data, idx, rsi_thres)) ### ADD FILTER EXAMPLE
                    ):
                    ### some_filter is JUST EXMAPLE !!

                    buy_point_found=True
                    buy_point_list.append(idx)
                    #print("sma above:", self._is_ma_above(self.stock_data, idx, ma_short, ma_long) )
                    #print(f"date: {self.stock_data.index[idx]}, buy point: true")
                else:
                    buy_point_found = False


        else:
            for idx in range(0, self.data_len):
                if (    
                        (not uptr_filter or self.__is_uptrend(idx, trend_col_name, zz_thres))
                    and (not sma_abv_filter or self.is_ma_above(self.stock_data, idx, ma_short, ma_long))
                    and (not rsi_abv_filter or self.is_rsi_above(self.stock_data, idx, rsi_thres))
                    and ( (not some_filter) or self.is_some_condition_met(self.stock_data, idx, rsi_thres)) ### ADD FILTER EXAMPLE
                    ):
                    ### some_filter is JUST EXMAPLE !!
                    buy_point_found=True
                    buy_point_list.append(idx)

                else:
                    buy_point_found = False


        
        bp_col = self.stock_data.columns.get_loc('buy pt')
        for item in buy_point_list:
            self.stock_data.iloc[item, bp_col]= True


    
    
        

    def set_buy_point(self, source: pd.Series)->pd.Series:
        """
        - set function of 'day of interest' column of self.stock_data

        parameter: 

        - source: pd.Serise with cell=1 indicate buy point 
        """
        self.stock_data['day of interest']=np.nan
        col_doi = self.stock_data.columns.get_loc('day of interest')
        for i in range(0, self.data_len):
            if source[i] ==1:
                self.stock_data.iloc[i, col_doi] = DayType.BUYPT
                self.buypt_dates.append(i)
        
        return self.stock_data['day of interest']
    
    def plot_peak_bottom(self, extrema: pd.Series,
                         line_cols: list=[], 
                  scatter_cols: list=[], plt_title: str='Extrema', 
                  annot: bool=True, text_box: str='', annotfont: float=6, 
                     showOption: str='show', savedir: str='', figsize: tuple=(36, 16)) :
        """
            extrema: pd Sereise with peak=1, bottom=0

            TO BE IMPLEMENT

        """
        color_list=['mediumslateblue', 'cyan',  'green', 'tomato','peru', 'olive', 'tan', 'darkred']
        color_idx=0
        for i in range(0, len(line_cols)):   
            plt.plot(line_cols[i], label=line_cols[i].name, 
                     alpha=0.6, linewidth=1.5, color=color_list[i])
            
        #plt.plot(peak, "x", color='limegreen', markersize=4)
        #plt.plot(btm, "x", color='salmon', markersize=4)

        # annot_y_offset= self.stock_data['close'][-1]*0.001
        # if annot:
            
        #     for i in range(0, len(self.extrema)):
        #         bar = ", %d bar"%(self.extrema['bar'][i]) if self.extrema['bar'][i]>0 else ''
        #         if self.extrema['type'][i]==self.PEAK:
                    
        #             ax.annotate("{:.2f}".format(self.extrema['close'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) +bar,
        #                     (self.extrema.index[i], self.extrema['close'][i]+annot_y_offset), fontsize=annotfont, ha='left', va='bottom' )
        #         if self.extrema['type'][i]==self.BOTTOM:
        #             ax.annotate("{:.2f}".format(self.extrema['close'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) 
        #                         +bar,
        #                     (self.extrema.index[i], self.extrema['close'][i]-annot_y_offset*3), fontsize=annotfont, ha='left', va='top' )

    
    def plot_cols(self, line_cols: list=[], 
                  scatter_cols: list=[], plt_title: str='Extrema', 
                  annot: bool=True, text_box: str='', annotfont: float=6, 
                     showOption: str='show', savedir: str='', figsize: tuple=(36, 16)) :
        
        """
        TO BE IMPLEMENT

        plot columns
        - line_cols: list of pd.Series to plot line graph
        - scatter_cols: list of pd.Series to plot scatter plot
        """

        for i in range(0, len(line_cols)):   
            plt.plot(line_cols[i], label=line_cols[i].name, 
                     alpha=0.6, linewidth=1.5, color=color_list[i])
            
        for i in range(0, len(scatter_cols)):   
            plt.plot(scatter_cols[i], label=scatter_cols[i].name, 
                     alpha=0.6, linewidth=1.5, color=color_list[i])

        plt.plot(figsize=figsize, dpi=200)
        color_list=['fuchsia', 'cyan',  'green', 'tomato','peru', 'olive', 'tan', 'darkred']



    
    def plot_extrema_from_self(self, 
                               stock_data: pd.DataFrame, 
                               extrema: pd.DataFrame=None, cols: list=[], 
                               to_plot_bp: bool=True, to_plot_zz: bool= True, to_shade_updown: bool=True,
                               plt_title: str='Extrema', annot: bool=True, text_box: str='', annotfont: float=6,
                               
                                showOption: str='show', savedir: str='', figsize: tuple=(36, 16), figdpi: int=200)->None :

        """
        - default plot function, plot close price, smoothen_price, extrema, break point, MAs, MACD
        
        Paramter
        -------
        - `stock_data`: soucr of stock data to plot
        - cols: col names to plot | 
        - `extrema` : df of peak, bottom
        - `to_plot_bp`: to plot break point or not |  `to_plot_zz`: to plot zigzag indicator or not  |  `to_shade_updown`: to shade region of MACD Signal >0 and <0 or not
        - `plt_title`: graph title 
        - `annot`: to annotate date of every peak, bottom or not | `text_box`: string in text box to print | `annotfont`: font of annotation
        - `figsize`: figsize pass to plt | `figdpi`: figdpi pass to plt 
        - showOption: depreciated
        - savedir: depreciated

         """
        
        ## Calculate neccessary info
        UP_PLT_UPLIM=stock_data['close'].max() *1.03
        UP_PLT_DOWNLIM=stock_data['close'].min() *0.95

        LOW_PLT_UPLIM = stock_data['MACD'].max()*1.1
        LOW_PLT_DOWNLIM = stock_data['MACD'].min()*1.1
         
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=figdpi, gridspec_kw={'height_ratios': [7, 1]})
        fig.subplots_adjust(hspace=0.05)  # adjust space between axes
        ax1.grid(which='major', color='lavender', linewidth=3)
        ax1.grid(which='minor', color='lavender', linewidth=3)
        
        ax1.plot(stock_data['close'], label='close price', color='blue', alpha=0.8, linewidth=1)
        ax1.set_ylim( UP_PLT_DOWNLIM, UP_PLT_UPLIM)
        ax2.set_ylim(LOW_PLT_DOWNLIM, LOW_PLT_UPLIM)

        color_list=['fuchsia', 'cyan',  'green', 'tomato','peru', 'olive', 'tan', 'darkred']
        color_idx=0

        ## PLOT EXTRA COLUMNS (including MAs) here

        for item in cols:
            line_sty='dashed' if 'ma' in item else '-' 
            ax1.plot(stock_data[item], 
                    label=item if isinstance(item, str) else '',
                    alpha=0.6, linewidth=0.8, color=color_list[color_idx], linestyle=line_sty)
            color_idx+=1
            
        if self.smoothen_price is not None:
            ax1.plot(self.smoothen_price[self.smoothen_price>0], color='gold')

        if extrema is not None:
            ax1.plot(extrema[extrema["type"]==self.PEAK]['close'], "x", color='limegreen', markersize=4)
            ax1.plot(extrema[extrema["type"]==self.BOTTOM]['close'], "x", color='red', markersize=4)
        
            ## Annotation ##
            annot_y_offset= stock_data['close'][-1]*0.001
            if annot:
                
                for i in range(0, len(extrema)):
                    pbday = ", %d bar"%(extrema['bar'][i]) if extrema['bar'][i]>0 else ''
                    if extrema['type'][i]==self.PEAK:
                        
                        ax1.annotate("{:.2f}".format(extrema['close'][i]) + ", {:.2%}".format(extrema['percentage change'][i]) +pbday,
                                (extrema.index[i], extrema['close'][i]+annot_y_offset), fontsize=annotfont, ha='left', va='bottom' )
                    if extrema['type'][i]==self.BOTTOM:
                        ax1.annotate("{:.2f}".format(extrema['close'][i]) + ", {:.2%}".format(extrema['percentage change'][i]) 
                                 +pbday,
                                (extrema.index[i], extrema['close'][i]-annot_y_offset*3), fontsize=annotfont, ha='left', va='top' )
                ax1.scatter(stock_data.index[-1], stock_data['close'][-1], s=self.SCATTER_MARKER_SIZE/2, color='blue')

                
        
            ## Textbox on left-top corner ##
            # textbox is plot on relative position of graph regardless of value of x/y axis
            ax1.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=ax1.transAxes) 

            ## Textbox of drop from last high ##
            if self.peak_indexes is not None:
                #percentage change from last peak
                maxval=float('-inf')
                idx=-1
                while stock_data['type'][idx+1] != 1:        # find latest peak
                    
                    if stock_data['close'][idx] > maxval:
                        maxval=stock_data['close'][idx]
                        maxdate = stock_data.index[idx]
                        maxidx=idx
                    idx-=1
                if maxidx==idx+1:
                    plot_latest_high =False
                else:
                    plot_latest_high = True
                
                
                logger.debug(f"latest price: {stock_data['close'].iloc[-1]}")
                perc = ( stock_data['close'].iloc[-1] - maxval)/maxval              
                ax1.text(0.9, 1.1, "lastest high: "+"{:.2f}".format(maxval), fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.text(0.9, 1.08, "latest price:  "+"{:.2f}".format(stock_data['close'].iloc[-1]), fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.text(0.9, 1.06, 'drop from last high: '+'{:.2%}'.format(perc)+f',{(maxidx+1)*(-1)} bar', fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.scatter(maxdate, maxval, s=self.SCATTER_MARKER_SIZE, marker='d', color='lime')
                if plot_latest_high:
                    ax1.text(maxdate-pd.DateOffset(1), maxval + annot_y_offset*2, "{:.2f}".format(maxval), fontsize=7,  ha='left', va='top', color='limegreen')
                ax1.text(stock_data.index[-1] + pd.DateOffset(1), stock_data['close'][-1] *0.995 , 'drop from last high: \n'
                         +'{:.2%}'.format(perc) 
                         +f',{(maxidx+1)*(-1)} bar', fontsize=8)

            

        ## PLOT BREAKPOINTS
        if to_plot_bp:
            try:
                assert 'buy pt' in stock_data
            except AssertionError:
                logger.warning("breakpoint must be set before plot")
                return
            
            filtered= stock_data[stock_data['buy pt']>0]['close']

            annot_y_offset = min(stock_data['close'][-1]*0.01, 10)
            marker_y_offset = stock_data['close'][-1]*0.01

        
            ax1.scatter(stock_data[stock_data['buy pt']>0].index, 
                        stock_data[stock_data['buy pt']>0]['close']-annot_y_offset/2, 
                        color='gold', s=self.SCATTER_MARKER_SIZE*2, marker=6, zorder=1)
            logger.info("break point dates: ")
            str_to_print = ''
            if len(filtered) >20:
                sep=', \t'
            else:
                sep='\n'            
            for ind, val in filtered.items():   # item is float
                str_to_print+=(ind.strftime("%Y-%m-%d")+sep)
                #logger.info(ind.strftime("%Y-%m-%d"))
                ax1.annotate("BP: "+ind.strftime("%Y-%m-%d")+", $"+"{:.2f}".format(val), (ind, val-annot_y_offset), fontsize=annotfont, ha='left', va='top', color='darkgoldenrod')
            logger.info(str_to_print)

        ### --- cutom plot here  --- ###

        #plt.plot(stock_data['buttered Close T=20'], alpha=0.8, linewidth=1.5, label='buttered Close T=20', color='cyan')
        #plt.plot(stock_data['buttered Close T=60'], alpha=0.8, linewidth=1.5, label='buttered Close T=60', color='magenta')
        
        
       ## shade green /red color as up/down trend by MACD signal

        if to_shade_updown and 'MACD' in stock_data and 'slope signal' in stock_data:
            ax1.fill_between(stock_data.index, UP_PLT_UPLIM, UP_PLT_DOWNLIM, where=stock_data['slope signal']>0, facecolor='palegreen', alpha=.12)
            ax1.fill_between(stock_data.index, UP_PLT_UPLIM, UP_PLT_DOWNLIM, where=stock_data['slope signal']<0, facecolor='pink', alpha=.12)
            ax2.plot(stock_data['MACD'], label='MACD', alpha=0.8, linewidth=1, color='indigo')
            ax2.plot(stock_data['signal'], label='signal', alpha=0.8, linewidth=1, color='darkorange')
            ax2.fill_between(stock_data.index, LOW_PLT_UPLIM, LOW_PLT_DOWNLIM, where=stock_data['slope signal']>0, facecolor='palegreen', alpha=.12)
            ax2.fill_between(stock_data.index, LOW_PLT_UPLIM, LOW_PLT_DOWNLIM, where=stock_data['slope signal']<0, facecolor='pink', alpha=.12)
            ax2.xaxis.grid(which='major', color='lavender', linewidth=3)
            ax2.xaxis.grid(which='minor', color='lavender', linewidth=3)


        ## PLOT ZIGZAG

        if to_plot_zz:
            try: 
                assert 'zigzag' in stock_data
            except AssertionError:
                logger.warning("to_plot_zz set to True but no 'zigzag' col found in stock data")

            else:

                up_offset = stock_data['close'][-1]*0.01
                down_offset = (-1)*stock_data['close'][-1]*0.012
                
                #plt.plot(stock_data['close'], label='close price', color='blue', alpha=0.9)
                ax1.scatter(stock_data[stock_data['zigzag'] ==1].index, stock_data[stock_data['zigzag'] ==1]['close'], color='lime', s=self.SCATTER_MARKER_SIZE, alpha=.6) #peak
                ax1.scatter(stock_data[stock_data['zigzag'] ==-1].index, stock_data[stock_data['zigzag'] ==-1]['close'], color='salmon',s=self.SCATTER_MARKER_SIZE, alpha=.6)  #bottom
                ax1.plot(stock_data[stock_data['zigzag'] !=0].index, stock_data[stock_data['zigzag'] !=0]['close'], 
                        label='zigzag indicator',color='dimgrey', alpha=0.6, linewidth=1)
                
                for i in range(0, len(stock_data['close'])):
                    if stock_data['zigzag'][i] ==-1:
                        ax1.annotate(stock_data.index[i].strftime("%Y-%m-%d"), (stock_data.index[i], stock_data['close'][i] + down_offset), fontsize=6, ha='left', va='top')

                for i in range(0, len(stock_data['close'])):
                    if stock_data['zigzag'][i] ==1:
                        ax1.annotate(stock_data.index[i].strftime("%Y-%m-%d"), (stock_data.index[i], stock_data['close'][i] + up_offset), fontsize=6, ha='left', va='bottom')



        

        
        fig.legend()
        fig.suptitle(plt_title)
        


    def plot_zigzag(self, plt_title: str='Zigzag Indicator', annot: bool=True, text_box: str='', annotfont: float=6, showOption: str='show', savedir: str='') :
        """
        TO BE DEVELOPPED
        """
        #plt.figure(figsize=(24, 10), dpi=200)

        up_offset = self.stock_data['close'][-1]*0.01
        down_offset = (-1)*self.stock_data['close'][-1]*0.012
        
        #plt.plot(self.stock_data['close'], label='close price', color='blue', alpha=0.9)
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==1].index, self.stock_data[self.stock_data['zigzag'] ==1]['close'], color='g', s=self.SCATTER_MARKER_SIZE) #peak
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==-1].index, self.stock_data[self.stock_data['zigzag'] ==-1]['close'], color='red',s=self.SCATTER_MARKER_SIZE)  #bottom
        plt.plot(self.stock_data[self.stock_data['zigzag'] !=0].index, self.stock_data[self.stock_data['zigzag'] !=0]['close'], 
                 label='zigzag indicator',color='dimgrey', alpha=0.8, linewidth=1.5)
        
        for i in range(0, len(self.stock_data['close'])):
            if self.stock_data['zigzag'][i] ==-1:
                plt.annotate(self.stock_data.index[i].strftime("%Y-%m-%d"), (self.stock_data.index[i], self.stock_data['close'][i] + down_offset), fontsize=6, ha='left', va='top')

        for i in range(0, len(self.stock_data['close'])):
            if self.stock_data['zigzag'][i] ==1:
                plt.annotate(self.stock_data.index[i].strftime("%Y-%m-%d"), (self.stock_data.index[i], self.stock_data['close'][i] + up_offset), fontsize=6, ha='left', va='bottom')

        #plt.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=plt.gca().transAxes)
        # plt.legend()
        # plt.grid(which='major', color='lavender')
        # plt.grid(which='minor', color='lavender')
        # plt.title(plt_title)
        
        
    
    def plot_break_pt(self):
        """
        TO BE DEVELOPPED
        """
        try:
            assert 'buy pt' in self.stock_data
        except AssertionError:
            logger.warning("breakpoint must be set before plot")
            return
        
        filtered= self.stock_data[self.stock_data['buy pt']>0]['close']

        annot_y_offset = min(self.stock_data['close'][-1]*0.01, 10)
        marker_y_offset = self.stock_data['close'][-1]*0.01

      
        plt.scatter(self.stock_data[self.stock_data['buy pt']>0].index, 
                    self.stock_data[self.stock_data['buy pt']>0]['close']-annot_y_offset/2, 
                    color='gold', s=1/self.data_len*6000, marker=6, zorder=1)
        logger.info("break point dates: ")
        
        for ind, val in filtered.items():   # item is float
            # print("type(item): ", type(item))
            # print(item)
            logger.info(ind.strftime("%Y-%m-%d"))
            plt.annotate("Break pt: "+ind.strftime("%Y-%m-%d")+", $"+"{:.2f}".format(val), (ind, val-annot_y_offset*2), fontsize=4, ha='left', va='top', color='darkgoldenrod')
            



    def default_analyser(self, tickers: str, start: str, end: str,
            method: str='', T: int=0, 
            window_size=10, smooth_ext=10, smooth:bool=False, zzupthres: float=0.09, zzdownthres: float=0.09,
            trend_col_name: str='slope signal',
           bp_filters:set=set(),
           ma_short_list: list=[], ma_long_list=[],
           plot_ma: list=[],
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str=None, figsize: tuple=(36,24), annotfont: float=6, figdpi:int=200,
           print_stock_data: bool=True,
           csv_dir: str=None) ->pd.DataFrame:

        """
        run everything

        return
        ---
        `self.stock_data`, dataframe of stock informartion

            
        Parameter
        ----

        - `method`: price source to calculate extrema, options: 'ma', 'ema', 'dma', 'butter', 'close'|
        - `T`: period of moving average if method set to 'ma', 'ema' or any kind with period required (no effect if method set to 'close')
       - `window_size`: window size to locate extrema from price source specified in `method` (no effect if method set to 'close')
        - `smooth`: if set to true, will set smoothen price and save to `self.smoothen_price`
            - has no effect on break point
            - smooth close price by linear convolution with np.blackman

        - `smooth_ext`: smooth extend to apply if `smooth`=true
        - `zzupthres`, `zzdownthres`: up/down threshold of zigzag indicator
        - `trend_col_name`:  source of uptrend signal, e.g. "slope signal" to use MACD Signal as trend
        - `bp_filters`: set class `BuyptFilter` to use
        - `extra_text_box`: text to print on graph (left top corner)
        - ma_short_list, ma_long_list: list of int, e.g. [3, 20], [9]  |  plot_ma: list of string, e.g. ma9, ema12 |
        - `graph_showOption`: 'save', 'show', 'no' |    `graph_dir`: dir to save graph 
        - `figsize`: figure size of graph | recommend: 1-3 months: figsize=(36,16)
        - `annotfont`: font size of annotation of peak bottom | recommend: 4-6
        - `figdpi`: dpi of graph
        - `csv_dir`: directory to save csv file of stock data and backtest result
    
        """
        plot_ma_num=[]
        for item in plot_ma:
            match =  re.match(r'([a-zA-Z]+)(\d+)', item)
            if match:
                
                num = int(match.group(2))
                plot_ma_num.append(num)
        
        max_ma_num = max(plot_ma_num) if len(plot_ma_num)>0 else 0
        
        max_ma_long = max(ma_long_list) if len(ma_long_list) >0 else 0
        pre_period = math.ceil(max(T,max_ma_long, 78, max_ma_num) *365/252) # adjust by proportion of trading day in a year calculate
        # 78=26*3 is for ploting ema26 which is required for MACD
        logger.info(f"days require before user's specified start day: {pre_period}")

        # since we need start earlier to calculate ma
        pre_start = pd.to_datetime(start) - pd.DateOffset(pre_period)
        logger.info(f"download stock start from: {pre_start}")

        self.download(tickers, start, end, pre_start)

        logger.info(f"analysing stock: {tickers}...")
        if self.data_len < 27:
            logger.warning("number of trading days <=26, analysis may not be accurate")

        logger.debug(f"config set:\nmethod={method}\tT={T}\ntrend={trend_col_name}\t{extra_text_box}")

        extra_col_name =set()

        ## Parameter Checking

        if T<1 and method != 'close':
            raise Exception("T must >=1")
        
        if BuyptFilter.MA_SHORT_ABOVE_LONG in bp_filters:
            if len(ma_short_list)==0 or len(ma_long_list)==0:
                raise Exception("buy point filter contain MA_SHORT_ABOVE_LONG, but ma_short and/or ma_long is not provided")
                

        if method=='close':
            self.set_extrema(src_data=self.stock_data['close'], close_price=self.stock_data['close'], interval=0)

        else:
            if method =='ma' or method =='ema' or  method =='dma':
                self.add_column_ma(src_data=self.close_price, mode=method, period=T)
                #self.add_col_slope(f"{method}{T}")
                extra_col_name.add(f"{method}{T}")

                # smooth
                if smooth:
                    if (method =='ma' or method=='ema') :
                        self.set_smoothen_price_blackman(self.stock_data[f"{method}{T}"], N=smooth_ext)
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, stock=tickers)
                            
                    elif method =='dma':
                        self.set_smoothen_price_blackman(self.stock_data[f"{method}{T}"], N=smooth_ext)
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, window_dir='both', stock=tickers)
                    else:
                        self.set_smoothen_price_blackman('close', N=smooth_ext)
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, stock=tickers)


                # no smooth
                if not smooth:

                    if method=='ma' or method=='ema':
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, stock=tickers)
                    elif method =='dma':
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, window_dir='both', stock=tickers)
                        
                    else:
                        self.set_extrema(src_data=self.stock_data[f"{method}{T}"], close_price=self.stock_data['close'], interval=window_size, stock=tickers)

            elif method =='butter':
                
                self.butter(T)
                self.set_extrema(src_data=self.stock_data[f'buttered Close T={T}'], close_price=self.stock_data['close'], window_dir='both', stock=tickers)
                extra_col_name.add(f'buttered Close T={T}')
            else:
                raise Exception("invalid method")
            
        ## ADD required MAs

        for item in ma_long_list:
            self.add_column_ma(self.close_price, 'ma', item)
            extra_col_name.add(f'ma{item}')
        for item in ma_short_list:
            self.add_column_ma(self.close_price, 'ma', item)
            extra_col_name.add(f'ma{item}')

        for item in plot_ma:
            match =  re.match(r'([a-zA-Z]+)(\d+)', item)
            if match:
                char = match.group(1)
                num = int(match.group(2))
                self.add_column_ma(self.close_price, char, num)
                extra_col_name.add(item)
        
        

        
        self.set_zigizag_trend(self.stock_data['close'], upthres=zzupthres, downthres=zzdownthres)
        

        self.__add_col_macd_group()
        

        self.__set_breakpoint( trend_col_name=trend_col_name,
                        bpfilters=bp_filters,
                        ma_short=ma_short_list,
                        ma_long=ma_long_list,
                        zz_thres=zzupthres,
                             )
        self.set_buy_point(self.stock_data['buy pt'])
 


        logger.debug(f"-- Stock Data of {tickers} (after all set)--")
        self.print_stock_data()
        logger.debug(f"number of price point: {len(self.get_stock_data())}" )

        logger.debug(f"-- Extrema of {tickers} (after all set)--")
        logger.debug(tabulate(self.get_extrema(), headers='keys', tablefmt='psql', floatfmt=("", ".2f","g", ".2%",)))
        logger.debug(f"number of extrema point: {len(self.get_extrema())}")



        rt = time.time()

        if graph_showOption != 'no':
            plot_start = time.time()
            logger.info("plotting graph..")

            T_str = T if T>0 else ''
            self.plot_extrema_from_self(stock_data=self.stock_data, extrema=self.extrema,
                            cols=extra_col_name, 
                            to_plot_bp=True, to_plot_zz=True, to_shade_updown=True,
                            plt_title=f"{tickers} {method}{T_str}", annot=True, 
                            text_box=f"{tickers}, {start} - {end}\n{extra_text_box}", 
                            figsize=figsize,figdpi=figdpi,
                            annotfont=annotfont, showOption=graph_showOption, savedir=graph_dir)
            #self.plot_zigzag(plt_title=f"{tickers} Zigzag Indicator", text_box=f"{tickers}, {start} - {end}, zigzag={zzupthres*100}%, {zzdownthres*100}%")
            

            
            plot_end = time.time()

            if graph_showOption == 'save':
                if graph_dir is None:
                    graph_dir='../../graph/'
                if not os.path.isdir(graph_dir):
                    os.makedirs(graph_dir)
                
                graph_dir += f'/{self.tickers}_{self.start}_{self.end}'
                add_num =1
                dir_norepeat = graph_dir
                while os.path.isfile(f'{dir_norepeat}.png'):
                    dir_norepeat = graph_dir + f'_{add_num}'
                    add_num +=1
                dir_norepeat += '.png'
                plt.savefig(dir_norepeat)
                plt.close()
                logger.info(f"graph saved as {dir_norepeat}")
            else:
                plt.show()
                logger.info("graph shown")

        else:
            filtered= self.stock_data[self.stock_data['buy pt']>0]['close']
            str_to_print = ''
            if len(filtered) >20:
                sep=', \t'
            else:
                sep='\n'            
            for ind, val in filtered.items():   # item is float
                str_to_print+=(ind.strftime("%Y-%m-%d")+sep)
                #logger.info(ind.strftime("%Y-%m-%d"))
            logger.info(str_to_print)

        if csv_dir is not None and print_stock_data:
            self.stock_data_to_csv(csv_dir)
                
        return self.stock_data
        
    def wrong_fn(self):     #for testing
        assert 0==1
        return

def default_analyser_runner(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
            macd_signal_T: int=9,
            bp_filters:set=set(),
            ma_short_list: list=[], ma_long_list=[],
            plot_ma: list=[],
            trend_col_name: str='slope signal',
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(30,30), annotfont: float=6, figdpi: int=200,
           csv_dir: str=None):
    stock = StockAnalyser()
    if extra_text_box =='':
        extra_text_box = 'filters of buy point: '
        for item in bp_filters:
            extra_text_box+= f'{item}, '
    stock.default_analyser(tickers=tickers, start=start, end=end,
                          method=method, T=T,
                        window_size=window_size, smooth_ext=smooth_ext,
                        zzupthres=zzupthres, zzdownthres=zzdownthres,
                        trend_col_name=trend_col_name,
                        bp_filters=bp_filters,
                        ma_long_list=ma_long_list, ma_short_list=ma_short_list,
                        plot_ma=plot_ma,
                        extra_text_box=extra_text_box,
                        graph_showOption=graph_showOption,
                        graph_dir=graph_dir,
                        figsize=figsize,annotfont=annotfont,
                        figdpi=figdpi,
                        csv_dir=csv_dir
    )           



    


    
def trial_runner():

    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    
    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='INFO'

    )
    stock=StockAnalyser()

   
    #df = stock.download('pdd', '2023-07-01', '2023-08-01')
    df = stock.default_analyser(tickers='tsm', start='2022-08-01', end='2023-08-16',
                            method='close',
                            bp_filters={BuyptFilter.CONVERGING_DROP, BuyptFilter.IN_UPTREND, BuyptFilter.RISING_PEAK, BuyptFilter.MA_SHORT_ABOVE_LONG},
                            ma_short_list=[3, 20], ma_long_list=[9, 50],
                            
                               graph_showOption='save' )
    



    df2= stock.get_stock_data()


    #print(tabulate(stock.close_price))
    stock.stock_data_to_csv()

    # stock.set_extrema(df['close'])
    # result = stock.get_stock_data()
    # print("close",result['close'].dtype)
    # print("type", result['type'].dtype)
    # print("p-b change", result['p-b change'].dtype)
    # print(result['close'].dtype==np.float64)
    #print("DOI", result['day of interest'].dtype)


     

if __name__ == "__main__":

    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='INFO'

    )
    logger.add(
        f"../log/stockAnalyser_{date.today()}_log.log",
        level='DEBUG'

    )
    logger.info("-- ****  NEW RUN START **** --")




    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker','-t', help='stock symbol',type=str, default=None)
    parser.add_argument('--start', '-s', help='start date', type=str, default='')
    parser.add_argument('--end', '-e', help='end date', type=str, default='')
    parser.add_argument('--stocklist_file','-f', help='stock list file dir',type=str, default=None)
    parser.add_argument('--graph_dir','-g', type=str, default='../../')  # end with /, no .png
    parser.add_argument('--csv_dir', '-v', help='csv folder dir (file name is pre-set), default=../../', type=str, default='../../')
    parser.add_argument('--figsize', type=tuple, default=(40,20))
    parser.add_argument('--figdpi', type=int, default=200)
    parser.add_argument('--showopt', '-o', help='graph show option: \'save\', \'show\', \'no\'', type=str, default='save')
    args=parser.parse_args()



    stockticker=args.ticker
    try:
        assert isinstance(stockticker, str)
    except Exception:
        pass
    else:
        stockticker=stockticker.upper()
        logger.info(f"stock given in cmd prompt: {stockticker}")

    stockstart = args.start
    stockend = args.end
    stock_lst_file = args.stocklist_file
    graph_file_dir = args.graph_dir
    graph_figsize=args.figsize
    graph_dpi=args.figdpi
    graph_show_opt = str(args.showopt).strip()
    csv_dir=args.csv_dir
    

    allow_direct_run_flag = False

    ## -- INFO -- ##
    ## RECOMMENDED graph dimension
    ## 1-3 months: figsize=(36,16), dpi=100-200, annotation fontsize=10
    # 12 months up :  figsize=(40,20), dpi=250, annotation fontsize=4

    ## Here to try the class

    ## -- Watch List -- ##

    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    

    ## run by watch list in code
    if 'watch_list' in locals() and allow_direct_run_flag:  
        logger.info("watch list found, command line stock ticker ommitted")
        try:
            for item in watch_list:
                logger.info(f"getting info of {item}")
                default_analyser_runner(item, stockstart, stockend,
                        method='close', 
                        bp_filters={BuyptFilter.CONVERGING_DROP, BuyptFilter.IN_UPTREND, BuyptFilter.RISING_PEAK, BuyptFilter.MA_SHORT_ABOVE_LONG},
                        figsize=graph_figsize, annotfont=4,
                        graph_dir=f'{graph_file_dir}_{item}.png',
                        graph_showOption=graph_show_opt, figdpi=graph_dpi,
                        csv_dir=csv_dir
                            )
                logger.info(f"{item} analyse done")
            
            logger.info("--  watch list run done  --")
        except NameError:
            logger.error("no watch list in code found!")
            logger.warning("Program proceed with cmd line arguments")

    ## run by watch list file provided in cmd     
    elif stock_lst_file != None:
        logger.info(f"stock list file got: {stock_lst_file}")
        with open(stock_lst_file, 'r') as fio:
            lines = fio.readlines()
        
        lines = [i.strip(" []\n") for i in lines]
        for item in lines:
            logger.info(f"getting info of {item}")
            
            default_analyser_runner(item, stockstart, stockend,
                method='close', 
                ma_short_list=[3, 20],
                ma_long_list=[9, 50],
                bp_filters={BuyptFilter.CONVERGING_DROP, BuyptFilter.IN_UPTREND, BuyptFilter.RISING_PEAK, BuyptFilter.MA_SHORT_ABOVE_LONG},
                figsize=graph_figsize, annotfont=4,
                graph_dir=graph_file_dir,
                extra_text_box='',
                 graph_showOption=graph_show_opt,
                 figdpi=graph_dpi,
                 csv_dir=csv_dir )
            logger.info(f"{item} analyse done")
        
        logger.info(f"{item} analyse done")


    ## run one stock from cmd
    else:
        # filter: peak bottom and sma above
        # default_analyser_runner(stockticker, stockstart, stockend,
        #         method='close', 
        #         ma_short_list=[3, 50],
        #         ma_long_list=[13, 150],
        #         bp_filters={BuyptFilter.Converging_drop, BuyptFilter.IN_UPTREND, BuyptFilter.RISING_PEAK, BuyptFilter.MA_SHORT_ABOVE_LONG},
        #         figsize=graph_figsize, annotfont=3,
        #         graph_dir=graph_file_dir,
        #         graph_showOption=graph_show_opt,
        #         csv_dir=csv_dir )

        # filter:sma cross
        default_analyser_runner(stockticker, stockstart, stockend,
                method='close', 
                ma_short_list=[3, 12],
                ma_long_list=[6, 20],
                bp_filters={ BuyptFilter.CONVERGING_DROP,BuyptFilter.RISING_PEAK, },
                figsize=graph_figsize, annotfont=3,
                graph_dir=graph_file_dir,
                graph_showOption=graph_show_opt, figdpi=graph_dpi,
                csv_dir=csv_dir )
    

    ## -- Example -- ##

    ## run in command line

    ## PDD from 2020, save graph in folder ../../stock_analyser_graph, save stock data csv in ../../csv
    # python stock_analyser.py -t=pdd -s=2020-01-01 -e=2023-08-16 -o=save -g=../../stock_analyser_graph -v=../../csv

    ## TSLA from 2020, don't plot graph
    # python stock_analyser.py -t=tsla -s=2020-01-01 -e=2023-08-16 -o=no

    ## run stock in app/configs/2stocks.txt by command line
    # python stock_analyser.py -f=./configs/2stocks.txt -s=2023-08-01 -e=2023-08-16 -o=no