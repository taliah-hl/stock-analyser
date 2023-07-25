from loguru import logger
import traceback
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, butter,filtfilt
from matplotlib import pyplot as plt
import yfinance as yf
import math
from tabulate import tabulate
import warnings


class StockAnalyser():
    def __init__(self, tickers: str, start: str, end: str):
        # Load stock info
        
        stock_info = yf.download(tickers, start=start, end=end)
        self.stock_data = pd.DataFrame(stock_info["Close"])
        self.smooth_data_N = 10
        self.find_extrema_interval = 5
        self.peaks = None
        self.bottoms = None
        self.extrema = None
        self.smoothen_price = None
        self.all_vertex= None
        self.peak_indexes=[]
        self.bottom_indexes=[]
    
    
    def get_close_price(self) -> pd.DataFrame:
        """
        Return: DataFrame of colsing price with date
        """
        return pd.DataFrame(self.stock_data["Close"])
    
    def get_stock_data(self)-> pd.DataFrame:
        """
        get self.stock_data
        """
        return pd.DataFrame(self.stock_data)
    
    def print_stock_data(self)->None:
        """
        pretty print self.stock_data
        """
        print(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=(".2f")))
    
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
    
    def add_column_ma(self,  mode: str='ma', period: int=9)->None:
        """
        add a column of moving average (MA) to stock_data
        
        Parameter
        -----
        - stock_data: DataFrame with column named['Close'] which is closing price of each day
        - period: time period (day)
        - mode options: moving average:'ma', exponential moving average:'ema', displaced moving average:'dma'
        - price_col_name: name of column of original stock price
        """
        DMA_DISPLACEMEN = math.floor(period/2)*(-1)

        if(mode =='ma'):
            self.stock_data[f'ma{period}'] = self.stock_data['Close'].rolling(period).mean()
            self.stock_data[f'ma{period}'].dropna(inplace=True)
            
        elif mode =='dma':
            ma = self.stock_data['Close'].rolling(period).mean()
            ma.dropna(inplace=True)
            self.stock_data[f"dma{period}"] = ma.shift(DMA_DISPLACEMEN)

        elif(mode=='ema'):
            self.stock_data[f'ema{period}'] = self.stock_data['Close'].ewm(span=period, adjust=False).mean()
        else:
            print("ma mode not given or wrong!")
        return

    def add_column_lwma(self,  mode: str='ma', period: int=9)->None:
        """
        Result not good
        
        """
        DMA_DISPLACEMEN = math.floor(period/4)*(-1)
        weights = np.arange(1, period + 1)
        lwma = self.stock_data['Close'].rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
        lwma.dropna(inplace=True)
        self.stock_data[f"lwma{period}"] = lwma.shift(DMA_DISPLACEMEN)
        return
        
    
    def add_col_slope(self, col_name: str)->None:
        """
        calculate slope of segment of given col name
        """
        slope_lst=[np.nan]
        for i in range(0, len(self.stock_data[col_name])-1):
            if(self.stock_data[col_name][i+1]==0 or self.stock_data[col_name][i]==0):
                slope_lst.append(np.nan)
            else:
                slope_lst.append(self.stock_data[col_name][i+1] - self.stock_data[col_name][i])
        self.stock_data[f'slope {col_name}'] = slope_lst

    
    def get_col(self, col_name: str)->pd.Series:
        """
        return self.stock_data[col_nmae]
        """
        return self.stock_data[col_name]

    def butter(self, filter_period: int, src_col: str='Close')->None:
        """
        filter frequency smaller than filter_period by Butterworth Low Pass Filter
        result is put into self.stock_data['buttered {src_col}']

        inputs

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
        self.stock_data[f'buttered {src_col}'] = filtfilt( b_coeff, a_coeff, self.stock_data[src_col])



    def get_smoothen_price(self)-> pd.DataFrame:
        """
        Return: DataFrame of smoothen price with date
        """
        return pd.DataFrame(self.smoothen_price)
    
    def set_smoothen_price_blackman(self, col_name: str, N: int=10)->None:
        """
        smoothen ['col_name'] of self.stock_data (mutating)
        - set fcuntion of self.smoothen_price
        - no return

        Parameter
        -----
        N: extend of smoothening. smaller->More accurate; larger -> more smooth
        """
        window = np.blackman(N)
        smoothed_data = np.convolve(window / window.sum(), self.stock_data[f"{col_name}"], mode="same")
        smoothed_data_chop = smoothed_data[1:-1]
        #exclude last and first raw
        self.smoothen_price = pd.DataFrame(smoothed_data_chop, index=self.stock_data.index[1:-1], columns=["Data"])

    
    def set_smoothen_price_polyfit(self, col_name: str)->None: #not work to smooth ma
        # potentially due to NaN value
        
        """
        smoothen ['col_name'] of self.stock_data by polyfit (mutating)
        - set fcuntion of self.smoothen_price
        - no return

        Parameter
        -----
        N: extend of smoothening. smaller->More accurate; larger -> more smooth
        """
        degree = 10
        print(f"---{col_name}---")
        print(self.stock_data[f"{col_name}"])
    

        X = np.array(self.stock_data[f"{col_name}"].reset_index().index)
        Y = self.stock_data[f"{col_name}"].to_numpy()

        
            
        poly_fit = np.poly1d(np.polyfit(X, Y, degree))

        self.smoothen_price = pd.DataFrame(poly_fit(X), columns=["Data"], index=self.stock_data.index)
        print(self.smoothen_price)
    


    def smoothen_non_mutate(self, original_data: pd.Series, N: int=10) -> pd.DataFrame:
        """
        Return: 1-col-DataFrame of smoothen data (length differ with original data)
        non-mutating
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
    
    def set_all_local_extrema(self):
        """
        just locate all vertex by comparing with prev 1 and foraward 1 data 
        """
        peaks_lst=[]
        peak_dates=[]
        for i in range(1, len(self.stock_data)-1):
            if (self.stock_data['Close'][i] > self.stock_data['Close'][i-1] ) & (self.stock_data['Close'][i] > self.stock_data['Close'][i+1]):
                peaks_lst.append(self.stock_data['Close'][i])
                peak_dates.append(self.stock_data.index[i])
        bottoms_lst=[]
        bottom_dates=[]
        for i in range(1, len(self.stock_data)-1):
            if (self.stock_data['Close'][i] < self.stock_data['Close'][i-1] ) & (self.stock_data['Close'][i] < self.stock_data['Close'][i+1]):
                bottoms_lst.append(self.stock_data['Close'][i])
                bottom_dates.append(self.stock_data.index[i])

        peaks = pd.DataFrame({"price": peaks_lst, "type": 'peak'}, index=peak_dates)
        bottoms = pd.DataFrame({"price": bottoms_lst, "type": 'bottom'}, index=bottom_dates)
      
        #self.all_vertex = pd.concat([peaks, bottoms]).sort_index()
        self.extrema = pd.concat([peaks, bottoms]).sort_index()
    
    
    def get_local_maxima(self, original_price_data: pd.DataFrame, smoothed_price_data: pd.DataFrame, interval: int=5) -> pd.DataFrame:
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
    
    def get_local_minima(self,original_close_data: pd.DataFrame, smoothed_price_data: pd.DataFrame, interval: int=5) -> pd.DataFrame:
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
        # print(type(bottom_indexs))
        # print(bottom_indexs)
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
    
    def set_extrema_left_window(self, data: str='', interval: int=5): # only shift window leftward
            """
            
            NOT MAINTAINED ANYMORE

            set function of self.extrema
            - if data not provided, calulate base on self.smoothen_price 
            - need to set self.smoothen_price first
            
            Parameter
            ---------
            data: col name of to calculate extrema
            interval: window to locate peak/bottom price on raw price by local extrema of smoothed price

            """
            if data=='':
                data_src = self.smoothen_price
            else:
                data_src = self.stock_data[data]

            self.bottom_indexs = argrelextrema(data_src.to_numpy(), np.less)[0]
            self.peak_indexes = argrelextrema(data_src.to_numpy(), np.greater)[0]
            
            extrema_idx_lst=[]
            for i in self.bottom_indexs:
                extrema_idx_lst.append((i, 0))  # 0 =bottom
            
            for i in self.peak_indexes:
                extrema_idx_lst.append((i, 1))  #1=peak
  
            extrema_idx_lst.sort()
            
            bottom_dates = []
            bottom_close = []
            extrema_dates = []
            extrema_close=[]
            
            prev_idx = 0
            for item in extrema_idx_lst:
                
                lower_boundary = max(0, prev_idx, item[0] - interval)
                upper_boundary = min(item[0] + 1, len(self.stock_data['Close']))
                stock_data_in_interval = self.stock_data['Close'].iloc[list(range(lower_boundary, upper_boundary))]
                
                extrema_dates.append(stock_data_in_interval.idxmax() if item[1] else stock_data_in_interval.idxmin())
                extrema_close.append((stock_data_in_interval.max(),'peak') if item[1] else (stock_data_in_interval.min(), 'bottom'))
                prev_idx = item[0]
            self.extrema = pd.DataFrame(extrema_close, columns=['price', 'type'], index=extrema_dates)
            self.extrema = self.extrema[~self.extrema.index.duplicated()]
            self.extrema.index.name = "date"

            percentage_change_lst =[np.nan]
            print(len(percentage_change_lst))
            for i in range(1, len(self.extrema)):
                #print(local_extrema['price'][i])
                percentage_change = (self.extrema['price'][i]-self.extrema['price'][i-1])/self.extrema['price'][i-1]
                #print(percentage_change)
                percentage_change_lst.append(percentage_change)

            self.extrema['percentage change'] = percentage_change_lst


            

    def set_extrema(self, data: str='', interval: int=5, window_dir: str='left'):
        """
        set function of self.extrema, self.peak_indexes, self.bottom_indexes
        - if data not provided, calulate base on self.smoothen_price 
        - need to set self.smoothen_price first
        
        Parameter
        ---------
        data: col name of to calculate extrema
        interval: window to locate peak/bottom price on raw price by local extrema of smoothed price

        """

        interval_dir=window_dir
        if data=='':
            data_src = self.smoothen_price
        else:
            data_src = self.stock_data[data]

        self.bottom_indexs = argrelextrema(data_src.to_numpy(), np.less)[0]
        self.peak_indexes = argrelextrema(data_src.to_numpy(), np.greater)[0]
            
        extrema_idx_lst=[]
        for i in self.bottom_indexs:
            extrema_idx_lst.append((i, 0))  # 0 =bottom
        
        for i in self.peak_indexes:
            extrema_idx_lst.append((i, 1))  #1=peak

        extrema_idx_lst.sort()
        
        bottom_dates = []
        bottom_close = []
        extrema_dates = []
        extrema_close=[]
        
        prev_idx = 0
        for item in extrema_idx_lst:
            
            lower_boundary = max(0, prev_idx, item[0] - interval)
            if window_dir=='left':
                upper_boundary = min(item[0] + 1, len(self.stock_data['Close']))
            else:
                upper_boundary = min(item[0] + 1 + interval, len(self.stock_data['Close']))
            stock_data_in_interval = self.stock_data['Close'].iloc[list(range(lower_boundary, upper_boundary))]
            
            extrema_dates.append(stock_data_in_interval.idxmax() if item[1] else stock_data_in_interval.idxmin())
            extrema_close.append((stock_data_in_interval.max(),'peak') if item[1] else (stock_data_in_interval.min(), 'bottom'))

            prev_idx = item[0]
        self.extrema = pd.DataFrame(extrema_close, columns=['price', 'type'], index=extrema_dates)
        
        self.extrema = self.extrema[~self.extrema.index.duplicated()]
        self.extrema.index.name = "date"

        percentage_change_lst =[np.nan]
        print(len(percentage_change_lst))
        for i in range(1, len(self.extrema)):
            #print(local_extrema['price'][i])
            percentage_change = (self.extrema['price'][i]-self.extrema['price'][i-1])/self.extrema['price'][i-1]
            #print(percentage_change)
            percentage_change_lst.append(percentage_change)

        self.extrema['percentage change'] = percentage_change_lst

        # calculate peak-to-bottom-time
        for i in range(0, len(self.extrema)):
            if self.extrema['type'][i] =='peak':
                if i+1 < len(self.extrema):
                    #assert self.extrema['type'][i+1]=='bottom'
                    pass
                    # TO DO 

    def plot_extrema(self, cols: list=[], plt_title: str='Extrema', annot: bool=True, text_box: str='') :

        """
        default plot function, plot closing price of self.stock_data, self.smoothen_price and self.extrema
        
        Paramter
        -------
        cols: col names to plot | text_box: string in text box to print

         """
        plt.figure(figsize=(28, 10), dpi=100)
        plt.plot(self.stock_data['Close'], label='close price', color='midnightblue', alpha=0.9)
        for item in cols:    
            try:
                plt.plot(self.stock_data[item], 
                    label=item if isinstance(item, str) else '',
                    alpha=0.8)
            except:
                print(f"column {item} does not exist")
        if self.smoothen_price is not None:
            plt.plot(self.smoothen_price[self.smoothen_price>0], color='gold')

        if self.extrema is not None:
            plt.plot(self.extrema[self.extrema["type"]=="peak"]['price'], "x", color='limegreen', markersize=8)
            plt.plot(self.extrema[self.extrema["type"]=="bottom"]['price'], "x", color='red', markersize=8)
        

            if annot:
                for date, extrema, percent in zip(self.extrema[self.extrema["type"]=="peak"].index, self.extrema[self.extrema["type"]=="peak"]['price'], self.extrema[self.extrema["type"]=="peak"]['percentage change']):
                    plt.annotate("{:.2f}".format(extrema)
                        + ", {:.2%}".format(percent), (date, extrema+4), fontsize=5)
                    
                for date, extrema, percent in zip(self.extrema[self.extrema["type"]=="bottom"].index, self.extrema[self.extrema["type"]=="bottom"]['price'], self.extrema[self.extrema["type"]=="bottom"]['percentage change']):
                    plt.annotate("{:.2f}".format(extrema)
                        + ", {:.2%}".format(percent), (date, extrema-6), fontsize=5)
                
        

            plt.text(0.01, 1,  text_box, fontsize=7, color='saddlebrown', ha='left', va='bottom',  transform=plt.gca().transAxes) 

            if self.peak_indexes is not None:
                #percentage change from last peak

                maxval = self.stock_data['Close'].iloc[list(range(self.peak_indexes[-1]-1, len(self.stock_data)))].max()
                print("maxval: ", maxval)
                print("cur price: ", self.stock_data['Close'].iloc[-1])
                perc = ( self.stock_data['Close'].iloc[-1] - maxval)/maxval

                plt.text(0.9, 1.1, "lastest high: "+"{:.2f}".format(maxval), fontsize=6,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.08, "current:  "+"{:.2f}".format(self.stock_data['Close'].iloc[-1]), fontsize=6,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.06, 'drop from last high: '+'{:.2%}'.format(perc), fontsize=6,  ha='left', va='top',  transform=plt.gca().transAxes)


        # plot on relative position of graph regardless of value of x/y axis
        plt.legend()
        plt.grid(which='major', color='lavender')
        plt.grid(which='minor', color='lavender')
        plt.title(plt_title)
        
        plt.show()

def runner(tickers: str, start: str, end: str, 
           ma_mode: str='', ma_T: int=0, 
           smooth: bool=False, wind=10, smooth_ext=10,
           all_vert =False):
    
    stock = StockAnalyser(tickers, start, end)
    extra_col =[]

    if all_vert:
        stock.set_all_local_extrema()

    else:
        if ma_mode !='' and ma_T !=0:
            stock.add_column_ma(ma_mode, ma_T)
            stock.add_col_slope(f"{ma_mode}{ma_T}")
            extra_col=[f"{ma_mode}{ma_T}"]

        # smooth
        if smooth:
            if (ma_mode =='ma' or ma_mode=='ema') :
                stock.set_smoothen_price_blackman(f"{ma_mode}{ma_T}", N=smooth_ext)
                stock.set_extrema(interval=wind)
                    
            elif ma_mode =='dma':
                stock.set_smoothen_price_blackman(f"{ma_mode}{ma_T}", N=smooth_ext)
                stock.set_extrema(interval=wind, window_dir='both')
            else:
                stock.set_smoothen_price_blackman('Close', N=smooth_ext)
                stock.set_extrema(interval=wind)


        # no smooth
        if not smooth:

            if ma_mode=='ma' or ma_mode=='ema':
                stock.set_extrema(data=f"{ma_mode}{ma_T}", interval=wind)
            elif ma_mode =='dma':
                stock.set_extrema(data=f"{ma_mode}{ma_T}", interval=wind, window_dir='both')
                print("hi ema")
            else:
                stock.set_extrema('Close', interval=wind)

    stock.print_stock_data()
    print("-- smoothen price --")
    print(tabulate(stock.get_smoothen_price(), headers='keys', tablefmt='psql'))
    print("-- extrema --")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=(None,".2f", None,  ".2%")))
    
    stock.plot_extrema(cols=extra_col, plt_title=f"{tickers} {ma_mode}{ma_T}", annot=True, text_box=f"{tickers}, {start} - {end}, window={wind}")

def runner_noma(tickers: str, start: str, end: str,smooth: bool=False, wind: int=10, smooth_ext: int=10):
    stock = StockAnalyser(tickers, start, end)
    #stock.add_column_ma(ma_mode, ma_T)
   
    stock.set_smoothen_price_blackman('Close')
    stock.set_extrema(interval=wind)
   
    print("-- smoothen price --")
    print(tabulate(stock.get_smoothen_price(), headers='keys', tablefmt='psql'))
    print(f"-- extrema, window={wind}--")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql'))

    stock.plot_extrema(plt_title=f"{tickers}")

def runner_polyfit(tickers: str, start: str, end: str,
           smooth: bool=False, wind=10, smooth_ext=10,
           ):
    stock = StockAnalyser(tickers, start, end)
    stock.set_smoothen_price_polyfit('Close')
    stock.set_extrema(interval=wind)
    stock.print_stock_data()
    print("-- smoothen price --")
    print(tabulate(stock.get_smoothen_price(), headers='keys', tablefmt='psql'))
    print("-- extrema --")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=(None,".2f", None,  ".2%")))
    
    stock.plot_extrema(plt_title=f"{tickers}", annot=True)
    
    
if __name__ == "__main__":
    #runner('TSLA', '2022-04-20', '2023-07-22', ma_mode='ema', ma_T=10, smooth=False, wind=5, smooth_ext=0)
    #runner_polyfit('NVDA', '2022-10-20', '2023-07-22',wind=10)
    stock=StockAnalyser('TSLA', '2022-04-20', '2023-07-22')
    stock.add_column_ma('ema', 20)
    stock.butter(10)

    stock.plot_extrema(cols=['buttered Close', 'ema20'])
    
    stock.print_stock_data()

    # stock.set_extrema_left_window('ema10', 0)

    # stock_data = stock.get_close_price()
    # print(stock_data)

    # degree = 20
    # X = np.array(stock_data.reset_index().index)
    # Y =stock_data['Close'].to_numpy()

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", np.RankWarning)
    #     poly_fit = np.poly1d(np.polyfit(X, Y, degree))

    # smoothen_price = pd.DataFrame(poly_fit(X), columns=["Data"], index=stock_data.index)
    
    # stock.plot_extrema(plt_title='TSLA')



            