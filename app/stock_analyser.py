from loguru import logger
import traceback
import numpy as np
import warnings
import pandas as pd
from scipy.signal import argrelextrema, butter,filtfilt
from matplotlib import pyplot as plt
import yfinance as yf
import math
from tabulate import tabulate
import zigzag as zz
import time
from datetime import date
import argparse
import sys


class StockAnalyser():

    ## CONSTANT ##
    PEAK =1
    BOTTOM =-1
    UPTRD =1
    DOWNTRD =-1


    def __init__(self):

        self.tickers= None
        self.start_date = None
        self.end_date = None
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


    def download(self, tickers: str, start: str, end: str):
        # Load stock info
        
        stock_info = yf.download(tickers, start=start, end=end)
        self.tickers=tickers
        self.start_date = start
        self.end_date = end
        self.stock_data = pd.DataFrame(stock_info["Close"])
        self.data_len = len(self.stock_data)
       


        ## CONSTANT ##
        # PEAK =1 | BOTTOM =-1  | UPTREND =1 | DOWNTREND =-1
        self.SCATTER_MARKER_SIZE=1/self.data_len*6000
    
    
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
    
    def print_stock_data(self, file_name: str='', writeToTxt: bool=False)->None:
        """
        pretty print self.stock_data
        writeToTxt: option to write table to `file_name`.txt
        """
        
        logger.debug(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=("", ".2f",".2f", "g",".2%", "g", "g", )))
        if writeToTxt and file_name=='':
            file_name = f"../../{self.ticker_name}.txt"

        if writeToTxt:
            with open(file_name, 'w') as fio:
                fio.write(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=("", ".2f",".2f", "g",".2%", "g", "g", )))
            logger.info(f"stock_data wrote to {file_name}")
    
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
    
    def add_column_ma(self,  src_data: pd.Series, mode: str='ma', period: int=9)->pd.Series:
        """
        add a column of moving average (MA) to stock_data
        return: ma
        Parameter
        -----
        - src_data: pd.Series of source stock price to cal ma
        - period: time period (day)
        - mode options: moving average:'ma', exponential moving average:'ema', displaced moving average:'dma'
        - 
        """
        DMA_DISPLACEMEN = math.floor(period/2)*(-1)

        if(mode =='ma'):
            self.stock_data[f'ma{period}'] = src_data.rolling(period).mean()
            self.stock_data[f'ma{period}'].dropna(inplace=True)
            return self.stock_data[f'ma{period}']
            
        elif mode =='dma':
            ma = src_data.rolling(period).mean()
            ma.dropna(inplace=True)
            self.stock_data[f"dma{period}"] = ma.shift(DMA_DISPLACEMEN)
            return self.stock_data[f"dma{period}"]

        elif(mode=='ema'):
            self.stock_data[f'ema{period}'] = src_data.ewm(span=period, adjust=False).mean()
            return self.stock_data[f"ema{period}"]
        else:
            raise Exception("ma mode not given or wrong!")
            return

    def add_column_lwma(self, src_data: pd.Series, mode: str='ma', period: int=9)->pd.Series:
        """
        Result not good
        
        """
        DMA_DISPLACEMEN = math.floor(period/4)*(-1)
        weights = np.arange(1, period + 1)
        lwma = src_data.rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
        lwma.dropna(inplace=True)
        self.stock_data[f"lwma{period}"] = lwma.shift(DMA_DISPLACEMEN)
        return self.stock_data[f"lwma{period}"]
        
    
    def add_col_slope(self, src_data)->pd.Series:
        """
        calculate slope of segment of given pd.Series
        - src_data: source of data to calculate slope pd.Series

        """
        slope_lst=[np.nan]
        for i in range(1, len(src_data)):
            if(src_data[i-1]==0 or src_data[i]==0):
                slope_lst.append(np.nan)
            else:
                slope_lst.append(src_data[i] - src_data[i-1])
        self.stock_data[f'slope {src_data.name}'] = slope_lst
        return self.stock_data[f'slope {src_data.name}']

    def add_col_macd_group(self, src_data: pd.Series)->None:
        """"
        add column of ema12, ema26, macd, signal line and slope of macd
        no return
        """
        self.add_column_ma(src_data, 'ema', 12)
        self.add_column_ma(src_data,'ema', 26)
        self.stock_data['MACD']=self.stock_data['ema12'] - self.stock_data['ema26']
        self.stock_data['signal'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
        self.add_col_slope(self.stock_data['MACD'])
        self.add_col_slope(self.stock_data['signal'])

    
    def add_col_macd(self, src_data: pd.Series)->None:
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

    

    
    def get_col(self, col_name: str)->pd.Series:
        """
        return self.stock_data[col_nmae]
        """
        return self.stock_data[col_name]

    def set_butter(self, filter_period: int, src_data: pd.Series)->pd.Series:
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
        self.stock_data[f'buttered {src_data.name} T={filter_period}'] = filtfilt( b_coeff, a_coeff, src_data)
        return self.stock_data[f'buttered {src_data.name} T={filter_period}']


    
    def set_smoothen_price_blackman(self, src_data: pd.Series, N: int=10)->pd.Series:
        """
        smoothen ['col_name'] of self.stock_data
        - set fcuntion of self.smoothen_price
        - no return

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
        non-mutating
        Argument
        ------
        - original_data: time Series of stock price
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
        just locate all vertex by comparing with prev 1 and foraward 1 data 
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

        peaks = pd.DataFrame({"price": peaks_lst, "type": self.PEAK}, index=peak_dates)
        bottoms = pd.DataFrame({"price": bottoms_lst, "type": self.BOTTOM}, index=bottom_dates)
      
        #self.all_vertex = pd.concat([peaks, bottoms]).sort_index()
        self.extrema = pd.concat([peaks, bottoms]).sort_index()
        self.stock_data['type'] = self.extrema['type']
        self.stock_data['p-b change'] = self.extrema['percentage change']
    
    
 


            

    def set_extrema(self, src_data: pd.Series, interval: int=5, window_dir: str='left', stock: str=''):
        """
        set function of self.extrema, self.peak_indexes, self.bottom_indexes
        - if data not specified, calulate base on self.smoothen_price 
        
        Parameter
        ---------
        data: col name of source to calculate extrema
        interval: window to locate peak/bottom price on original price by source price

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
                        if src_data[j] < btm:
                            btm = src_data[j]
                            btm_idx = j

                    extrema_idx_lst.insert(i, (btm_idx, self.BOTTOM))

                else: # 2 bottoms
                    pk = float('-inf')
                    pk_idx=0
                    for j in range(extrema_idx_lst[i-1][0], extrema_idx_lst[i][0]):
                        if src_data[j] > pk:
                            pk = src_data[j]
                            pk_idx = j

                    extrema_idx_lst.insert(i, (pk_idx, self.PEAK))

        
        prev_idx = 0

        for i in range(0, len(extrema_idx_lst)):
        
            lower_boundary = max(0, extrema_idx_lst[i-1][0] if i>0 else 0, extrema_idx_lst[i][0]-interval)
            if window_dir=='left':
                upper_boundary = min(extrema_idx_lst[i][0] + 1,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(src_data))

            else :
                upper_boundary = min(extrema_idx_lst[i][0] + 1 +interval,
                                     extrema_idx_lst[i+1][0] if i<len(extrema_idx_lst)-1 else extrema_idx_lst[i][0] + 1, 
                                     len(src_data))
            stock_data_in_interval = src_data.iloc[list(range(lower_boundary, upper_boundary))]
            
            extrema_dates.append(stock_data_in_interval.idxmax() if extrema_idx_lst[i][1] else stock_data_in_interval.idxmin())
            extrema_close.append((stock_data_in_interval.max(),self.PEAK) if extrema_idx_lst[i][1] else (stock_data_in_interval.min(), self.BOTTOM))


        self.extrema = pd.DataFrame(extrema_close, columns=['price', 'type'], index=extrema_dates)
        
        self.extrema = self.extrema[~self.extrema.index.duplicated()]
        self.extrema.index.name = "date"


        percentage_change_lst =[np.nan]
        for i in range(1, len(self.extrema)):
            percentage_change = (self.extrema['price'][i]-self.extrema['price'][i-1])/self.extrema['price'][i-1]
            percentage_change_lst.append(percentage_change)

        self.extrema['percentage change'] = percentage_change_lst

        self.stock_data['type'] = self.extrema['type']
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
        self.stock_data['zigzag'] = np.nan
        self.stock_data['zigzag'] = zz.peak_valley_pivots(src_data, upthres, -downthres)
        self.stock_data.iloc[-1, self.stock_data.columns.get_loc('zigzag')] = (-1)* self.stock_data[self.stock_data['zigzag'] !=0]['zigzag'][-2]
        # correct the problem that last point of zigzag is flipped sometime
        self.extrema['zigzag'] = self.stock_data[self.stock_data['zigzag'] !=0]['zigzag']
        logger.debug("set zigzag done")

        self.stock_data['trend'] = np.nan
        cur_trend=0
        trend_col= self.stock_data.columns.get_loc('trend')
        for i in range(1, self.data_len):
            cur_trend= self.stock_data['zigzag'][i-1] *(-1)
            if cur_trend:
                self.stock_data.iloc[i, trend_col] = cur_trend
            else:
                self.stock_data.iloc[i, trend_col] = self.stock_data['trend'][i-1]
        logger.debug("set trend done")

    
        
    
    def set_breakpoint(self, 
                       close_price: pd.Series, zz: pd.Series=None,
                       trend_src: pd.Series=None,
                       zzupthres: int=0.09, 
                       bp_filter_conv_drop: bool=True,
                       bp_filter_rising_peak: bool=True,
                       bp_filter_uptrend: bool=True) -> None:
        """
        find break point of stock price
        - bp_condition_num: mode for condition of finding break point
        - trend_src: pd.Series with >0 indicate uptrend, < 0 indicate downtrend
        """

        ## -- checking -- ##
        uptrddays=[]
        checking_flag = 0
        try:
            assert 'trend' in self.stock_data
            assert 'type' in self.stock_data
            assert 'p-b change' in self.stock_data
        except AssertionError:
            logger.warning("zigzag must set before set trend!\nProgram Exit")
            exit(1)
        
        ## -- parameter -- ##

        incl_1st_btm = True
        uptrend_src = 'other'     # uptend source: 'zz': zigzag, 'other': anysource
        
        ## -- flags -- ##

        to_find_bp_flag = True

        if not (bp_filter_conv_drop or bp_filter_rising_peak or bp_filter_uptrend):
            logger.warning("break point filters all set to false. no break point will be plotted")
            to_find_bp_flag = False

        POS_INF = float('inf')


        self.stock_data['starred point'] = np.nan
        prev_pbc = POS_INF
        chck_date_idx = np.nan
        star_lst =[]

        

        # converging bottom condition set 1:
        # 1. peak-to-bottom drop less than previous peak-to-bottom drop
        # 2. next little peak rise above previous little peak
        # 3.  cur price rise above prev big bottom * 1+ zigzag threshold (up trend already detected on that day)
        prev_pbc = POS_INF
        prev_peak = POS_INF


        
        i=0
        if uptrend_src=='zz':
            while i< self.data_len-2:
                # if zz[i] ==1:
                #     prev_pbc = POS_INF
                #     chck_date_idx = np.nan
                
                if zz[i] ==-1:     # encounter big bottom
                    
                    
                    chck_date_idx = np.nan
                    
                    j = 1 if incl_1st_btm else 0
                    cur_big_btm = close_price[i]
                    while zz[i+j] != 1 :   # not encounter big peak yet
                        rise_back_offset=18250      # random large number
                        next_peak_offset =0
                        if self.stock_data['type'][i+j] == -1:
                            # 1. find prev little peak
                            l=0
                            while self.stock_data['type'][i+j+l] !=1: # find prev little peak
                                if i+j+l-1 >=0:
                                    l-=1
                                else:
                                    break
                            prev_peak = close_price[i+j+l] if l !=0 else prev_peak

                            rise_back_flag = False
                            break_pt_found_flag = False
                        
                            

                            while self.stock_data['type'][i+j+next_peak_offset] != 1 and zz[i+j+next_peak_offset] != 1: #find next little peak
                                if close_price[i+j+next_peak_offset] >= prev_peak: # record closest date rise back to prev peak
                                    if not rise_back_flag:
                                        rise_back_offset = next_peak_offset
                                        rise_back_flag = True

                                next_peak_offset +=1
                                if i+j+next_peak_offset+1>self.data_len-1:
                                    break
                            
                            #potential break point = neif xt little peak or date of rise back to prev peak, which ever earlier
                            potential_bp = min(i+j+rise_back_offset, i+j+next_peak_offset)  

                        
                            if (to_find_bp_flag 
                                and ( (not bp_filter_conv_drop) or self.stock_data['p-b change'][i+j] > prev_pbc )
                                and ( (not bp_filter_rising_peak) or rise_back_flag )
                                and ( (not bp_filter_uptrend) or close_price[potential_bp] > cur_big_btm*(1+zzupthres) ) 
                                ):  
                                break_pt_found_flag = True
                                
                            if break_pt_found_flag:
                                chck_date_idx = self.stock_data.index[potential_bp]
                                star_lst.append(potential_bp)

                            prev_pbc = self.stock_data['p-b change'][i+j]


                        j = j + max(next_peak_offset, 1)
                        if i+j+1 > self.data_len-1:
                            break
                    i=i+max(j, 1)
                i+=1

        elif uptrend_src=='other':
            i=0
            while i< self.data_len-2:
                next_peak_offset=0  
                if trend_src[i] >0 and self.stock_data['type'][i] ==-1 :
                    
                    ## find prev little peak
                    l =0
                    while self.stock_data['type'][i+l] !=1:
                        if i+l-1 >=0:
                            l-=1
                        else:
                            break
                    prev_peak = close_price[i+l] if l !=0 else prev_peak

                    rise_back_flag = False
                    break_pt_found_flag = False
                    potential_bp_found_flag=False
                    potential_bp = POS_INF
                    next_peak_found=False
                    
                    rise_back_offset=0      # random large number
                    k =0
                    while trend_src[i+k] >0:
                        if close_price[i+k] >= prev_peak: # record closest date rise back to prev peak
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
                    
                    if (to_find_bp_flag 
                        and potential_bp< POS_INF
                            and ( (not bp_filter_conv_drop) or self.stock_data['p-b change'][i] > prev_pbc )
                            and ( (not bp_filter_rising_peak) or rise_back_flag )
                            ):  
                        break_pt_found_flag = True
                    if break_pt_found_flag:
                        star_lst.append(potential_bp)
                    prev_pbc = self.stock_data['p-b change'][i]

                i+=max(1, next_peak_offset)





        star_col = self.stock_data.columns.get_loc('starred point')
        for item in star_lst:
            self.stock_data.iloc[item, star_col]= 1
        

    def set_sell_point(self):
        self.stock_data['sell pt'] = self.stock_data['starred point']
    
    def plot_peak_bottom(self, extrema: pd.Series,
                         line_cols: list=[], 
                  scatter_cols: list=[], plt_title: str='Extrema', 
                  annot: bool=True, text_box: str='', annotfont: float=6, 
                     showOption: str='show', savedir: str='', figsize: tuple=(36, 16)) :
        """
            extrema: pd Sereise with peak=1, bottom=0

            TO BE IMPLEMENT

        """
        color_list=['fuchsia', 'cyan', 'tomato', 'peru', 'green', 'olive', 'tan', 'darkred']
        for i in range(0, len(line_cols)):   
            plt.plot(line_cols[i], label=line_cols[i].name, 
                     alpha=0.6, linewidth=1.5, color=color_list[i])
            
        #plt.plot(peak, "x", color='limegreen', markersize=4)
        #plt.plot(btm, "x", color='salmon', markersize=4)

        # annot_y_offset= self.stock_data['Close'][-1]*0.001
        # if annot:
            
        #     for i in range(0, len(self.extrema)):
        #         bar = ", %d bar"%(self.extrema['bar'][i]) if self.extrema['bar'][i]>0 else ''
        #         if self.extrema['type'][i]==self.PEAK:
                    
        #             ax.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) +bar,
        #                     (self.extrema.index[i], self.extrema['price'][i]+annot_y_offset), fontsize=annotfont, ha='left', va='bottom' )
        #         if self.extrema['type'][i]==self.BOTTOM:
        #             ax.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) 
        #                         +bar,
        #                     (self.extrema.index[i], self.extrema['price'][i]-annot_y_offset*3), fontsize=annotfont, ha='left', va='top' )

    
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
        color_list=['fuchsia', 'cyan', 'tomato', 'peru', 'green', 'olive', 'tan', 'darkred']



    
    def plot_extrema_from_self(self, cols: list=[], plt_title: str='Extrema', annot: bool=True, text_box: str='', annotfont: float=6,
                               to_plot_bp: bool=True,
                     showOption: str='show', savedir: str='', figsize: tuple=(36, 16)) :

        """
        default plot function, plot closing price of self.stock_data, self.smoothen_price and self.extrema
        
        Paramter
        -------
        cols: col names to plot | text_box: string in text box to print |
        showOption: 'show': show by plt.show, 'save': save graph without showing (suitable for env without GUI)
        savedir: dir to save plot |

         """
        
        ## Calculate neccessary info
        UP_PLT_UPLIM=self.stock_data['Close'].max() *1.05
        UP_PLT_DOWNLIM=self.stock_data['Close'].min() *0.9

        LOW_PLT_UPLIM = self.stock_data['MACD'].max()*1.1
        LOW_PLT_DOWNLIM = self.stock_data['MACD'].min()*1.1
         
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=250, gridspec_kw={'height_ratios': [7, 1]})
        fig.subplots_adjust(hspace=0.05)  # adjust space between axes
        ax1.plot(self.stock_data['Close'], label='close price', color='blue', alpha=0.8, linewidth=0.8)
        ax1.set_ylim( UP_PLT_DOWNLIM, UP_PLT_UPLIM)
        ax2.set_ylim(LOW_PLT_DOWNLIM, LOW_PLT_UPLIM)

        color_list=['fuchsia', 'cyan', 'tomato', 'peru', 'green', 'olive', 'tan', 'darkred']

        for i in range(0, len(cols)):    
            
            ax1.plot(self.stock_data[cols[i]], 
                    label=cols[i] if isinstance(cols[i], str) else '',
                    alpha=0.6, linewidth=1.5, color=color_list[i])
            
        if self.smoothen_price is not None:
            ax1.plot(self.smoothen_price[self.smoothen_price>0], color='gold')

        if self.extrema is not None:
            ax1.plot(self.extrema[self.extrema["type"]==self.PEAK]['price'], "x", color='limegreen', markersize=4)
            ax1.plot(self.extrema[self.extrema["type"]==self.BOTTOM]['price'], "x", color='salmon', markersize=4)
        
            ## Annotation ##
            annot_y_offset= self.stock_data['Close'][-1]*0.001
            if annot:
                
                for i in range(0, len(self.extrema)):
                    pbday = ", %d bar"%(self.extrema['bar'][i]) if self.extrema['bar'][i]>0 else ''
                    if self.extrema['type'][i]==self.PEAK:
                        
                        ax1.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) +pbday,
                                (self.extrema.index[i], self.extrema['price'][i]+annot_y_offset), fontsize=annotfont, ha='left', va='bottom' )
                    if self.extrema['type'][i]==self.BOTTOM:
                        ax1.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) 
                                 +pbday,
                                (self.extrema.index[i], self.extrema['price'][i]-annot_y_offset*3), fontsize=annotfont, ha='left', va='top' )
                ax1.scatter(self.stock_data.index[-1], self.stock_data['price'][-1], s=self.SCATTER_MARKER_SIZE/2, color='blue')

                
        
            ## Textbox on left-top corner ##
            # textbox is plot on relative position of graph regardless of value of x/y axis
            ax1.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=ax1.transAxes) 

            ## Textbox of drop from last high ##
            if self.peak_indexes is not None:
                #percentage change from last peak
                maxval=float('-inf')
                idx=-1
                while self.stock_data['type'][idx+1] != 1:        # find latest peak
                    
                    if self.stock_data['Close'][idx] > maxval:
                        maxval=self.stock_data['Close'][idx]
                        maxdate = self.stock_data.index[idx]
                        maxidx=idx
                    idx-=1
                if maxidx==idx+1:
                    plot_latest_high =False
                else:
                    plot_latest_high = True
                
                
                logger.debug(f"latest price: {self.stock_data['Close'].iloc[-1]}")
                perc = ( self.stock_data['Close'].iloc[-1] - maxval)/maxval              
                ax1.text(0.9, 1.1, "lastest high: "+"{:.2f}".format(maxval), fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.text(0.9, 1.08, "latest price:  "+"{:.2f}".format(self.stock_data['Close'].iloc[-1]), fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.text(0.9, 1.06, 'drop from last high: '+'{:.2%}'.format(perc)+f',{(maxidx+1)*(-1)} bar', fontsize=7,  ha='left', va='top',  transform=ax1.transAxes)
                ax1.scatter(maxdate, maxval, s=self.SCATTER_MARKER_SIZE, marker='d', color='lime')
                if plot_latest_high:
                    ax1.text(maxdate-pd.DateOffset(1), maxval + annot_y_offset*2, "{:.2f}".format(maxval), fontsize=7,  ha='left', va='top', color='limegreen')
                ax1.text(self.stock_data.index[-1] + pd.DateOffset(1), self.stock_data['Close'][-1] *0.98 , 'drop from last high: \n'
                         +'{:.2%}'.format(perc) 
                         +f',{(maxidx+1)*(-1)} bar', fontsize=8)

            

        ## PLOT BREAKPOINTS
        if to_plot_bp == True:
            try:
                assert 'starred point' in self.stock_data
            except AssertionError:
                logger.warning("breakpoint must be set before plot")
                return
            
            filtered= self.stock_data[self.stock_data['starred point']>0]['Close']

            annot_y_offset = min(self.stock_data['Close'][-1]*0.01, 10)
            marker_y_offset = self.stock_data['Close'][-1]*0.01

        
            ax1.scatter(self.stock_data[self.stock_data['starred point']>0].index, 
                        self.stock_data[self.stock_data['starred point']>0]['Close']-annot_y_offset/2, 
                        color='gold', s=1/self.data_len*6000, marker=6, zorder=1)
            logger.info("break point dates: ")
            
            for ind, val in filtered.items():   # item is float
                # print("type(item): ", type(item))
                # print(item)
                logger.info(ind.strftime("%Y-%m-%d"))
                ax1.annotate("Break pt: "+ind.strftime("%Y-%m-%d")+", $"+"{:.2f}".format(val), (ind, val-annot_y_offset*2), fontsize=4, ha='left', va='top', color='darkgoldenrod')


        ### --- cutom plot here  --- ###

        #plt.plot(self.stock_data['buttered Close T=20'], alpha=0.8, linewidth=1.5, label='buttered Close T=20', color='cyan')
        #plt.plot(self.stock_data['buttered Close T=60'], alpha=0.8, linewidth=1.5, label='buttered Close T=60', color='magenta')
        
        ax1.fill_between(self.stock_data.index, UP_PLT_UPLIM, UP_PLT_DOWNLIM, where=self.stock_data['slope signal']>0, facecolor='palegreen', alpha=.2)
        ax1.fill_between(self.stock_data.index, UP_PLT_UPLIM, UP_PLT_DOWNLIM, where=self.stock_data['slope signal']<0, facecolor='pink', alpha=.2)
       

        if 'MACD' in self.stock_data and 'slope signal' in self.stock_data:
            ax2.plot(self.stock_data['MACD'], label='MACD', alpha=0.8, linewidth=1, color='indigo')
            ax2.plot(self.stock_data['signal'], label='signal', alpha=0.8, linewidth=1, color='darkorange')
            ax2.fill_between(self.stock_data.index, LOW_PLT_UPLIM, LOW_PLT_DOWNLIM, where=self.stock_data['slope signal']>0, facecolor='palegreen', alpha=.2)
            ax2.fill_between(self.stock_data.index, LOW_PLT_UPLIM, LOW_PLT_DOWNLIM, where=self.stock_data['slope signal']<0, facecolor='pink', alpha=.2)
            ax2.xaxis.grid(which='major', color='lavender', linewidth=3)
            ax2.xaxis.grid(which='minor', color='lavender', linewidth=3)


        

        ax1.grid(which='major', color='lavender', linewidth=3)
        ax1.grid(which='minor', color='lavender', linewidth=3)
        
    
 
        fig.suptitle(plt_title)
        fig.legend()


    def plot_zigzag(self, plt_title: str='Zigzag Indicator', annot: bool=True, text_box: str='', annotfont: float=6, showOption: str='show', savedir: str='') :
        #plt.figure(figsize=(24, 10), dpi=200)

        up_offset = self.stock_data['Close'][-1]*0.01
        down_offset = (-1)*self.stock_data['Close'][-1]*0.012
        
        #plt.plot(self.stock_data['Close'], label='close price', color='blue', alpha=0.9)
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==1].index, self.stock_data[self.stock_data['zigzag'] ==1]['Close'], color='g', s=self.SCATTER_MARKER_SIZE) #peak
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==-1].index, self.stock_data[self.stock_data['zigzag'] ==-1]['Close'], color='red',s=self.SCATTER_MARKER_SIZE)  #bottom
        plt.plot(self.stock_data[self.stock_data['zigzag'] !=0].index, self.stock_data[self.stock_data['zigzag'] !=0]['Close'], 
                 label='zigzag indicator',color='dimgrey', alpha=0.8, linewidth=1.5)
        
        for i in range(0, len(self.stock_data['Close'])):
            if self.stock_data['zigzag'][i] ==-1:
                plt.annotate(self.stock_data.index[i].strftime("%Y-%m-%d"), (self.stock_data.index[i], self.stock_data['Close'][i] + down_offset), fontsize=6, ha='left', va='top')

        for i in range(0, len(self.stock_data['Close'])):
            if self.stock_data['zigzag'][i] ==1:
                plt.annotate(self.stock_data.index[i].strftime("%Y-%m-%d"), (self.stock_data.index[i], self.stock_data['Close'][i] + up_offset), fontsize=6, ha='left', va='bottom')

        #plt.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=plt.gca().transAxes)
        # plt.legend()
        # plt.grid(which='major', color='lavender')
        # plt.grid(which='minor', color='lavender')
        # plt.title(plt_title)
        
        
    
    def plot_break_pt(self):
        try:
            assert 'starred point' in self.stock_data
        except AssertionError:
            logger.warning("breakpoint must be set before plot")
            return
        
        filtered= self.stock_data[self.stock_data['starred point']>0]['Close']

        annot_y_offset = min(self.stock_data['Close'][-1]*0.01, 10)
        marker_y_offset = self.stock_data['Close'][-1]*0.01

      
        plt.scatter(self.stock_data[self.stock_data['starred point']>0].index, 
                    self.stock_data[self.stock_data['starred point']>0]['Close']-annot_y_offset/2, 
                    color='gold', s=1/self.data_len*6000, marker=6, zorder=1)
        logger.info("break point dates: ")
        
        for ind, val in filtered.items():   # item is float
            # print("type(item): ", type(item))
            # print(item)
            logger.info(ind.strftime("%Y-%m-%d"))
            plt.annotate("Break pt: "+ind.strftime("%Y-%m-%d")+", $"+"{:.2f}".format(val), (ind, val-annot_y_offset*2), fontsize=4, ha='left', va='top', color='darkgoldenrod')
            



    def default_analyser(self, tickers: str, start: str, end: str,
            method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True, bp_filter_uptrend: bool=True,
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(36,24), annotfont: float=6) ->pd.DataFrame:

        """
        run everything

        return: self.stock_data, dataframe of stock informartion

            
        Parameter

        - method: options: 'ma', 'ema', 'dma', 'butter', 'close'|
        - T: day range of taking ma/butterworth low pass filter |
        - window_size: window to locate extrema from approx. price |
        - extra_text_box: extra textbox to print on graph|
        - graph_showOption: 'save', 'show', 'no'
    
        """

        ## To Do: need check fuction
        self.download(tickers, start, end)
        logger.info(f"analysing stock: {tickers}...")
        if self.data_len < 27:
            logger.warning("number of trading days <=26, analysis may not be accurate")

        extra_col_name =[]
        smooth=False

        ## Parameter Checking

        if T<1 and method != 'close':
            raise Exception("T must >=1")

        if method=='close':
            self.set_extrema(src_data=self.stock_data['Close'], interval=0)

        else:
            if method =='ma' or method =='ema' or  method =='dma':
                self.add_column_ma(src_data=self.stock_data['Close'], method=method, period=T)
                #self.add_col_slope(f"{method}{T}")
                extra_col_name=[f"{method}{T}"]

                # smooth
                if smooth:
                    if (method =='ma' or method=='ema') :
                        self.set_smoothen_price_blackman(self.stock_data[f"{method}{T}"], N=smooth_ext)
                        self.set_extrema(interval=window_size, stock=tickers)
                            
                    elif method =='dma':
                        self.set_smoothen_price_blackman(self.stock_data[f"{method}{T}"], N=smooth_ext)
                        self.set_extrema(interval=window_size, window_dir='both', stock=tickers)
                    else:
                        self.set_smoothen_price_blackman('Close', N=smooth_ext)
                        self.set_extrema(interval=window_size, stock=tickers)


                # no smooth
                if not smooth:

                    if method=='ma' or method=='ema':
                        self.set_extrema(self.stock_data[f"{method}{T}"], interval=window_size, stock=tickers)
                    elif method =='dma':
                        self.set_extrema(self.stock_data[f"{method}{T}"], interval=window_size, window_dir='both', stock=tickers)
                        
                    else:
                        self.set_extrema(self.stock_data['Close'], interval=window_size, stock=tickers)

            elif method =='butter':
                
                self.butter(T)
                self.set_extrema(self.stock_data[f'buttered Close T={T}'], window_dir='both', stock=tickers)
                extra_col_name=[f'buttered Close T={T}']
            else:
                raise Exception("invalid method")
        
        self.set_zigizag_trend(self.stock_data['Close'], upthres=zzupthres, downthres=zzdownthres)
        

        self.add_col_macd_group(self.stock_data['Close'])
        self.set_breakpoint(self.stock_data['Close'], trend_src=self.stock_data['slope signal'],
                            bp_filter_conv_drop=bp_filter_conv_drop, bp_filter_rising_peak=bp_filter_rising_peak)
        
        logger.debug(f"-- Stock Data of {tickers} (after all set)--")
        self.print_stock_data()
        logger.debug("number of price point:", len(self.get_stock_data()))

        logger.debug(f"-- Extrema of {tickers} (after all set)--")
        logger.debug(tabulate(self.get_extrema(), headers='keys', tablefmt='psql', floatfmt=("", ".2f","g", ".2%",)))
        logger.debug(f"number of extrema point: {len(self.get_extrema())}")



        rt = time.time()

        if graph_showOption != 'no':
            plot_start = time.time()
            logger.info("plotting graph..")

       
            self.plot_extrema_from_self(cols=[], plt_title=f"{tickers} {method}{T}", annot=True, 
                            text_box=f"{tickers}, {start} - {end}, uptrend scr=signal slope>0\n{extra_text_box}", 
                            annotfont=annotfont, showOption=graph_showOption, savedir=graph_dir)
            # self.plot_zigzag(plt_title=f"{tickers} Zigzag Indicator", text_box=f"{tickers}, {start} - {end}, zigzag={zzupthres*100}%, {zzdownthres*100}%")
            

            
            plot_end = time.time()

            if graph_showOption == 'save':
                plt.savefig(graph_dir)
                logger.info(f"graph saved as {graph_dir}")
            else:
                plt.show()
                logger.info("graph shown")
        return self.stock_data
        


def runner_analyser(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
            macd_signal_T: int=9,
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True, bp_filter_uptrend: bool=True,
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(30,30), annotfont: float=6):
    stock = StockAnalyser()
    stock.default_analyser(tickers=tickers, start=start, end=end,
                          method=method, T=T,
                        window_size=window_size, smooth_ext=smooth_ext,
                        zzupthres=zzupthres, zzdownthres=zzdownthres,
                        bp_filter_conv_drop=bp_filter_conv_drop,
                        bp_filter_rising_peak=bp_filter_rising_peak,
                        bp_filter_uptrend=bp_filter_uptrend,
                        extra_text_box=extra_text_box,
                        graph_showOption=graph_showOption,
                        graph_dir=graph_dir,
                        figsize=figsize,annotfont=annotfont
    )


def runner(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
           all_vertex =False, 
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True, bp_filter_uptrend: bool=True,
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(36,24), annotfont: float=6) :
    """
    Parameter

    - method: options: 'ma', 'ema', 'dma', 'butter', 'close'|
    - T: day range of taking ma/butterworth low pass filter |
    - window_size: window to locate extrema from approx. price |
    - extra_text_box: extra textbox to print on graph
    """
    runner_start = time.time()
    
    stock = StockAnalyser(tickers, start, end)
    extra_col =[]
    smooth=False

    

    ## Parameter Checking

    if T<1 and method != 'close':
        raise Exception("T must >=1")

    if method=='close':
        stock.set_extrema(data='Close', interval=0)

    else:
        if method =='ma' or method =='ema' or  method =='dma':
            stock.add_column_ma(method, T)
            #stock.add_col_slope(f"{method}{T}")
            extra_col=[f"{method}{T}"]

            # smooth
            if smooth:
                if (method =='ma' or method=='ema') :
                    stock.set_smoothen_price_blackman(f"{method}{T}", N=smooth_ext)
                    stock.set_extrema(interval=window_size)
                        
                elif method =='dma':
                    stock.set_smoothen_price_blackman(f"{method}{T}", N=smooth_ext)
                    stock.set_extrema(interval=window_size, window_dir='both')
                else:
                    stock.set_smoothen_price_blackman('Close', N=smooth_ext)
                    stock.set_extrema(interval=window_size)


            # no smooth
            if not smooth:
                if method=='ma' or method=='ema':
                    stock.set_extrema(data=f"{method}{T}", interval=window_size)
                elif method =='dma':
                    stock.set_extrema(data=f"{method}{T}", interval=window_size, window_dir='both')
                    print("hi ema")
                else:
                    stock.set_extrema('Close', interval=window_size)
        elif method =='butter':
            
            stock.butter(T)
            stock.set_extrema(f'buttered Close T={T}', window_dir='both')
            extra_col=[f'buttered Close T={T}']
        else:
            raise Exception("invalid method")
    
    stock.set_zigizag_trend( upthres=zzupthres, downthres=zzdownthres)
    

    stock.add_col_macd()
    
    logger.debug("-- Stock Data --")
    stock.print_stock_data()
    logger.debug("number of price point:", len(stock.get_stock_data()))

    logger.debug("-- Extrema --")
    logger.debug(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=("", ".2f","g", ".2%",)))
    logger.debug("number of extrema point:", len(stock.get_extrema()))


    #stock.set_breakpoint(zzupthres=zzupthres, 
    #                     bp_filter_conv_drop=bp_filter_conv_drop, bp_filter_rising_peak=bp_filter_rising_peak, bp_filter_uptrend=bp_filter_uptrend)
     

    rt = time.time()
    logger.debug(f"time for data manipulate: {rt -runner_start}", )
    plot_start = time.time()
    logger.info("plotting graph..")

    #fig, ax = plt.subplots()
    #ax.figure(figsize=(36, 16), dpi=400)

    # ax.grid(which='major', color='lavender', linewidth=2)
    # ax.grid(which='minor', color='lavender', linewidth=2)
    stock.plot_extrema_from_self(cols=extra_col, plt_title=f"{tickers} {method}{T}", annot=True, 
                       text_box=f"{tickers}, {start} - {end}, window={window_size}\n{extra_text_box}", 
                       annotfont=annotfont, showOption=graph_showOption, savedir=graph_dir)
    # stock.plot_zigzag(plt_title=f"{tickers} Zigzag Indicator", text_box=f"{tickers}, {start} - {end}, zigzag={zzupthres*100}%, {zzdownthres*100}%")
    # stock.plot_break_pt()
    # plt.legend()
    
    plot_end = time.time()

    if graph_showOption == 'save':
        plt.savefig(graph_dir)
        logger.info("graph saved")
    else:
        plt.show()
        logger.info("graph shown")

    


def runner_polyfit(tickers: str, start: str, end: str,
           smooth: bool=False, wind=10, smooth_ext=10,
           ):
    stock = StockAnalyser(tickers, start, end)
    stock.set_smoothen_price_polyfit('Close')
    stock.set_extrema(interval=wind)
    logger.debug("-- Stock Data --")
    stock.print_stock_data()
    logger.debug("-- Extrema --")
    logger.debug(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=(None,".2f", None,  ".2%")))
    #stock.plot_extrema(plt_title=f"{tickers}", annot=True)
    
    
if __name__ == "__main__":

    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='INFO'

    )
    logger.add(
        f"../../stockAnalyser_{date.today()}_log.log",
        level='DEBUG'

    )
    logger.info("--  NEW RUN START --")


    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='pdd')
    parser.add_argument('--start',  type=str, default='2022-07-20')
    parser.add_argument('--end',  type=str, default='2023-08-03')
    parser.add_argument('--stocklist_file',type=str, default='nan')
    parser.add_argument('--graph_dir',type=str, default='../../')  # no .png
    parser.add_argument('--figsize', type=tuple, default=(100,70))
    parser.add_argument('--figdip', type=int, default=200)
    args=parser.parse_args()

    stockticker=args.ticker
    stockstart = args.start
    stockend = args.end
    stock_lst_file = args.stocklist_file
    graph_file_dir = args.graph_dir

    logger.info(f"stock given in cmd prompt: {stockticker}")

    run_by_code = False

    ## -- INFO -- ##
    ## RECOMMENDED graph dimension
    ## 1-3 months: figsize=(36,16), dpi=100-200, annotation fontsize=10
    # 12 months up :  figsize=(100,70), dpi=250, annotation fontsize=4

    ## Here to try the class

    ## -- Watch List -- ##

    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    


    mode='runner_run'

    if mode=='analyser_run':
        runner_analyser(tickers=stockticker, start=stockstart, end=stockend,
            method='close', zzupthres=0.09, zzdownthres=0.09,
            bp_filter_conv_drop=True, bp_filter_rising_peak=True,
            figsize=(36, 16), annotfont=20,
            graph_dir=f'{graph_file_dir}.png',
            bp_filter_uptrend=True, graph_showOption='save' )
        
    elif mode=='runner_run':
        if 'watch_list' in locals() and run_by_code:
            logger.info("watch list found, command line stock ticker ommitted")
            try:
                for item in watch_list:
                    logger.info(f"getting info of {item}")
                    runner_analyser(item, stockstart, stockend,
                            method='close', 
                            bp_filter_conv_drop=True, bp_filter_rising_peak=True,
                            figsize=(100, 70), annotfont=4,
                            graph_dir=f'{graph_file_dir}_{item}.png',
                            bp_filter_uptrend=True, graph_showOption='save' )
                    logger.info(f"{item} analyse done")
                
                logger.info("--  watch list run done  --")
            except NameError:
                logger.error("no watch list in code found!")
                logger.warning("Program proceed with cmd line arguments")
                
        elif stock_lst_file != 'nan':
            logger.info(f"stock list file got: {stock_lst_file}")
            with open(stock_lst_file, 'r') as fio:
                lines = fio.readlines()
            
            for item in lines:
                item=item.strip()
                logger.info(f"getting info of {item}")
                
                runner_analyser(item, stockstart, stockend,
                    method='close', 
                    bp_filter_conv_drop=True, bp_filter_rising_peak=True,
                    figsize=(100, 70), annotfont=4,
                    graph_dir=f'{graph_file_dir}_{item}.png',
                    bp_filter_uptrend=True, graph_showOption='save' )
                logger.info(f"{item} analyse done")
            
            logger.info(f"{item} analyse done")

        else:

           runner_analyser(stockticker, stockstart, stockend,
                    method='close', 
                    bp_filter_conv_drop=True, bp_filter_rising_peak=True,
                    figsize=(100, 70), annotfont=4,
                    graph_dir=f'{graph_file_dir}.png',
                    bp_filter_uptrend=True, graph_showOption='save' )

    ## -- Example -- ##
    ## E.g. Plot PDD 2022-10-20 to 2023-07-22, get extrema with EMA5
    # runner('PDD', '2023-10-20', '2023-07-22', method='ema', T=5, showOption='save', graph_dir='../graph.png')

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

 



            