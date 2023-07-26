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
            raise Exception("ma mode not given or wrong!")
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
        self.stock_data[f'buttered {src_col} T={filter_period}'] = filtfilt( b_coeff, a_coeff, self.stock_data[src_col])



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

    
    def set_smoothen_price_polyfit(self, col_name: str)->None:         
        """
        smoothen ['col_name'] of self.stock_data by polyfit (mutating)
        not work to smooth ma potentially due to NaN value
        - set fcuntion of self.smoothen_price

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
        NOT MAINTAINED ANYMORE

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
        NOT MAINTAINED ANYMORE

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
        - if data not specified, calulate base on self.smoothen_price 
        
        Parameter
        ---------
        data: col name of source to calculate extrema
        interval: window to locate peak/bottom price on original price by source price

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
        extrema_dates = []
        extrema_close=[]
        
        prev_idx = 0

        for i in range(0, len(extrema_idx_lst)):
        
            lower_boundary = max(0, extrema_idx_lst[i-1][0] if i>0 else 0, extrema_idx_lst[i][0]-interval)
            if window_dir=='left':
                upper_boundary = min(extrema_idx_lst[i][0] + 1,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(self.stock_data['Close']))

            else :
                upper_boundary = min(extrema_idx_lst[i][0] + 1 +interval,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(self.stock_data['Close']))
            stock_data_in_interval = self.stock_data['Close'].iloc[list(range(lower_boundary, upper_boundary))]
            
            extrema_dates.append(stock_data_in_interval.idxmax() if extrema_idx_lst[i][1] else stock_data_in_interval.idxmin())
            extrema_close.append((stock_data_in_interval.max(),'peak') if extrema_idx_lst[i][1] else (stock_data_in_interval.min(), 'bottom'))


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
        self.extrema['peak-to-bottom day'] = np.nan
        self.extrema['back to peak time'] = np.nan

        for i in range(0, len(self.extrema)):
            if self.extrema['type'][i] =='bottom' and i>0:
                try:
                    assert self.extrema['type'][i-1]=='peak'
                except AssertionError:
                    print("peak bottom does not appear alternatively, possible wrong setting")
                self.extrema['peak-to-bottom day'][i] = int((self.extrema.index[i] - self.extrema.index[i-1]).days)
        
        print('type of time:' , type(self.extrema['peak-to-bottom day'][1]))
        print('type of date:' , type(self.extrema.index[1]))



    def plot_extrema(self, cols: list=[], plt_title: str='Extrema', annot: bool=True, text_box: str='') :

        """
        default plot function, plot closing price of self.stock_data, self.smoothen_price and self.extrema
        
        Paramter
        -------
        cols: col names to plot | text_box: string in text box to print

         """
         
        plt.figure(figsize=(24, 10), dpi=100)
        plt.plot(self.stock_data['Close'], label='close price', color='royalblue', alpha=0.9)

        color_list=['violet', 'cyan', 'tomato', 'peru', 'green', 'olive', 'tan', 'darkred']

        for i in range(0, len(cols)):    
            
            plt.plot(self.stock_data[cols[i]], 
                    label=cols[i] if isinstance(cols[i], str) else '',
                    alpha=0.8, linewidth=1.5, color=color_list[i])
            
        if self.smoothen_price is not None:
            plt.plot(self.smoothen_price[self.smoothen_price>0], color='gold')

        if self.extrema is not None:
            plt.plot(self.extrema[self.extrema["type"]=="peak"]['price'], "x", color='limegreen', markersize=8)
            plt.plot(self.extrema[self.extrema["type"]=="bottom"]['price'], "x", color='red', markersize=8)
        
            ## Annotation ##
            if annot:
                for i in range(0, len(self.extrema)):
                    if self.extrema['type'][i]=='peak':
                        y_offset= 1
                        plt.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]),
                                (self.extrema.index[i], self.extrema['price'][i]+y_offset), fontsize=7, ha='left', va='bottom' )
                    if self.extrema['type'][i]=='bottom':
                        y_offset= -2
                        plt.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) 
                                 + ", %d bar"%(self.extrema['peak-to-bottom day'][i]),
                                (self.extrema.index[i], self.extrema['price'][i]+y_offset), fontsize=7, ha='left', va='bottom' )

                
        
            ## Textbox on left-top corner ##
            # textbox is plot on relative position of graph regardless of value of x/y axis
            plt.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=plt.gca().transAxes) 

            ## Textbox of drop from last high ##
            if self.peak_indexes is not None:
                #percentage change from last peak

                maxval = self.stock_data['Close'].iloc[list(range(self.peak_indexes[-1]-1, len(self.stock_data)))].max()
                print("maxval: ", maxval)
                print("cur price: ", self.stock_data['Close'].iloc[-1])
                perc = ( self.stock_data['Close'].iloc[-1] - maxval)/maxval              
                plt.text(0.9, 1.1, "lastest high: "+"{:.2f}".format(maxval), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.08, "current:  "+"{:.2f}".format(self.stock_data['Close'].iloc[-1]), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.06, 'drop from last high: '+'{:.2%}'.format(perc), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)


        ### --- cutom plot here  --- ###

        #plt.plot(self.stock_data['buttered Close T=20'], alpha=0.8, linewidth=1.5, label='buttered Close T=20', color='cyan')
        #plt.plot(self.stock_data['buttered Close T=60'], alpha=0.8, linewidth=1.5, label='buttered Close T=60', color='magenta')

        
        plt.legend()
        plt.grid(which='major', color='lavender')
        plt.grid(which='minor', color='lavender')
        plt.title(plt_title)
        
        plt.show()

def runner(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            wind=10, smooth_ext=10,
           all_vertex =False):
    """
    Parameter

    - method: options: 'ma', 'ema', 'dma', 'butter' |
    - T: day range of taking ma/butterworth low pass filter |
    - all_vertex: get all vertex from orginal stock price |
    - wind: window to locate extrema from approx. price
    """
    
    stock = StockAnalyser(tickers, start, end)
    extra_col =[]
    smooth=False

    ## Parameter Checking

    if T<1:
        raise Exception("T must >=1")

    if all_vertex:
        stock.set_all_local_extrema()

    else:
        if method =='ma' or method =='ema' or  method =='dma':
            stock.add_column_ma(method, T)
            stock.add_col_slope(f"{method}{T}")
            extra_col=[f"{method}{T}"]

            # smooth
            if smooth:
                if (method =='ma' or method=='ema') :
                    stock.set_smoothen_price_blackman(f"{method}{T}", N=smooth_ext)
                    stock.set_extrema(interval=wind)
                        
                elif method =='dma':
                    stock.set_smoothen_price_blackman(f"{method}{T}", N=smooth_ext)
                    stock.set_extrema(interval=wind, window_dir='both')
                else:
                    stock.set_smoothen_price_blackman('Close', N=smooth_ext)
                    stock.set_extrema(interval=wind)


            # no smooth
            if not smooth:

                if method=='ma' or method=='ema':
                    stock.set_extrema(data=f"{method}{T}", interval=wind)
                elif method =='dma':
                    stock.set_extrema(data=f"{method}{T}", interval=wind, window_dir='both')
                    print("hi ema")
                else:
                    stock.set_extrema('Close', interval=wind)

        elif method =='butter':
            
            stock.butter(T)
            stock.set_extrema(f'buttered Close T={T}', window_dir='both')
            extra_col=[f'buttered Close T={T}']
        else:
            raise Exception("invalid method")
        


    print("-- Stock Data --")
    stock.print_stock_data()

    print("-- Extrema --")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=(None,".2f", None,  ".2%")))

    stock.plot_extrema(cols=extra_col, plt_title=f"{tickers} {method}{T}", annot=True, text_box=f"{tickers}, {start} - {end}, window={wind}")


def runner_polyfit(tickers: str, start: str, end: str,
           smooth: bool=False, wind=10, smooth_ext=10,
           ):
    stock = StockAnalyser(tickers, start, end)
    stock.set_smoothen_price_polyfit('Close')
    stock.set_extrema(interval=wind)
    print("-- Stock Data --")
    stock.print_stock_data()
    print("-- Extrema --")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=(None,".2f", None,  ".2%")))
    stock.plot_extrema(plt_title=f"{tickers}", annot=True)
    
    
if __name__ == "__main__":

    ## Here to try the class
    runner('PDD', '2022-10-20', '2023-07-22', method='ema', T=5)

    ## -- Example -- ##
    ## E.g. Plot PDD 2022-10-20 to 2023-07-22, get extrema with EMA5
    # runner('PDD', '2022-10-20', '2023-07-22', method='ema', T=5)

    ## E.g. Plot NVDA 2022-10-20 to 2023-07-22, get extrema with EMA10
    # runner('NVDA', '2022-10-20', '2023-07-22', method='ema', T=10)

    ## E.g. Plot TSLA 2023-02-20 to 2023-07-22, get extrema with butterworth low pass filter with period=10 day
    # runner('TSLA', '2023-02-20', '2023-07-22', method='butter', T=10)



    ####　### ####
    #runner_polyfit('NVDA', '2022-10-20', '2023-07-22',wind=10)
    # stock=StockAnalyser('TSLA', '2023-01-20', '2023-07-22')
    
    # #stock.butter(10)
    # stock.butter(10)
    # stock.set_extrema('buttered Close T=10', window_dir='both')

    # stock.plot_extrema(plt_title='TSLA 2023-01-20 to 2023-07-22: extrema with butter T=10')
    
    # stock.print_stock_data()

 



            