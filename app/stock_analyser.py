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
import warnings
import zigzag as zz
import time
import argparse


class StockAnalyser():

    ## CONSTANT ##
    PEAK =1
    BOTTOM =-1
    UPTRD =1
    DOWNTRD =-1

    def __init__(self, tickers: str, start: str, end: str):
        # Load stock info
        
        stock_info = yf.download(tickers, start=start, end=end)
        self.stock_data = pd.DataFrame(stock_info["Close"])
        self.data_len = len(self.stock_data)
        self.smooth_data_N = 10
        self.find_extrema_interval = 5
        self.peaks = None
        self.bottoms = None
        self.extrema = None
        self.smoothen_price = None
        self.all_vertex= None
        self.peak_indexes=[]
        self.bottom_indexes=[]

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
    
    def print_stock_data(self)->None:
        """
        pretty print self.stock_data
        """
        print(tabulate(self.stock_data, headers='keys', tablefmt='psql', floatfmt=("", ".2f",".2f", "g",".2%", "g", "g", )))
    
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
        for i in range(1, self.data_len-1):
            if (self.stock_data['Close'][i] > self.stock_data['Close'][i-1] ) & (self.stock_data['Close'][i] > self.stock_data['Close'][i+1]):
                peaks_lst.append(self.stock_data['Close'][i])
                peak_dates.append(self.stock_data.index[i])
        bottoms_lst=[]
        bottom_dates=[]
        for i in range(1, self.data_len-1):
            if (self.stock_data['Close'][i] < self.stock_data['Close'][i-1] ) & (self.stock_data['Close'][i] < self.stock_data['Close'][i+1]):
                bottoms_lst.append(self.stock_data['Close'][i])
                bottom_dates.append(self.stock_data.index[i])

        peaks = pd.DataFrame({"price": peaks_lst, "type": self.PEAK}, index=peak_dates)
        bottoms = pd.DataFrame({"price": bottoms_lst, "type": self.BOTTOM}, index=bottom_dates)
      
        #self.all_vertex = pd.concat([peaks, bottoms]).sort_index()
        self.extrema = pd.concat([peaks, bottoms]).sort_index()
        self.stock_data['type'] = self.extrema['type']
        self.stock_data['p-b change'] = self.extrema['percentage change']
    
    

   
    
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
                extrema_close.append((stock_data_in_interval.max(), self.PEAK) if item[1] else (stock_data_in_interval.min(), self.BOTTOM))
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
            extrema_close.append((stock_data_in_interval.max(),self.PEAK) if extrema_idx_lst[i][1] else (stock_data_in_interval.min(), self.BOTTOM))


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

        self.stock_data['type'] = self.extrema['type']
        self.stock_data['p-b change'] = self.extrema['percentage change']
        # calculate peak-to-bottom-time
        self.stock_data['peak-to-bottom day'] = np.nan
        self.extrema['peak-to-bottom day'] = np.nan
        self.extrema['back to peak time'] = np.nan

        print("-- Stock Data --")
        self.print_stock_data()



        # df.iloc[row_num, col_num]
        # col
        
        # verify does peak-bottom appear alternatingly
        for i in range(1, len(self.extrema)):
            
            try:

                if self.extrema['type'][i] ==self.BOTTOM:
                    try:
                        
                        assert self.extrema['type'][i-1]==self.PEAK
                    except AssertionError:
                        print("peak bottom does not appear alternatively, possible wrong setting")
            except IndexError as err:
                    print(err)
                    print("possibly because day range too short to get local extrema")
                    exit(1)

        # calculate peak-to-bottom change

        for i in range(1, len(self.extrema)):
            if self.extrema['type'][i] ==self.BOTTOM:
                idx =i
                break
        
        pbc_col = self.stock_data.columns.get_loc('peak-to-bottom day')
        pbc_excol = self.extrema.columns.get_loc('peak-to-bottom day')
        for i in range(1, self.data_len):
            if self.stock_data['type'][i] ==self.BOTTOM:                 
        
                # find prev peak date
                j=0
                while self.stock_data['type'][i-j] != self.PEAK:
                    j+=1
                    if i-j < 0:
                        break
                self.stock_data.iloc[i, pbc_col] = j if j>0 else np.nan
                self.extrema.iloc[idx, pbc_excol] = j if j>0 else np.nan
                idx +=2
        
            
        
        print("-- Stock Data --")
        self.print_stock_data()
        print("-- Extrema --")
        print(tabulate(self.get_extrema(), headers='keys', tablefmt='psql', floatfmt=("", ".2f","g", ".2%",)))




    def set_zigizag(self, thres: float=0.09) -> None:
        self.stock_data['zigzag'] = np.nan
        self.stock_data['zigzag'] = zz.peak_valley_pivots(self.stock_data['Close'], thres, -thres)
        self.stock_data['zigzag'][-1] *= -1 # correct the problem that last point of zigzag musy be flipped
        self.extrema['zigzag'] = self.stock_data[self.stock_data['zigzag'] !=0]['zigzag']

    def set_trend(self) -> None:
        """
        set up/down trend by zigzag indicator
        """
        try:
            assert 'zigzag' in self.stock_data
        except AssertionError:
            print("zigzag must before set before set trend!")
            exit(1)

        self.stock_data['trend'] = np.nan
        cur_trend=0
        trend_col= self.stock_data.columns.get_loc('trend')
        for i in range(1, self.data_len):
            cur_trend= self.stock_data['zigzag'][i-1] *(-1)
            if cur_trend:
                self.stock_data.iloc[i, trend_col] = cur_trend
            else:
                self.stock_data.iloc[i, trend_col] = self.stock_data['trend'][i-1]
        
    
    def set_breakpoint(self, zzthres: int, 
                       bp_filter_conv_drop: bool=True,
                       bp_filter_rising_peak: bool=True,
                       bp_filter_uptrend: bool=True) -> None:
        """
        find break point of stock price
        - bp_condition_num: mode for condition of finding break point
        """
        uptrddays=[]
        checking_flag = 0
        try:
            assert 'trend' in self.stock_data
            assert 'type' in self.stock_data
            assert 'p-b change' in self.stock_data
        except AssertionError:
            print("zigzag must set before set breakpoint!")
            exit(1)

        POS_INF = float('inf')



        filtered = self.stock_data[self.stock_data['trend']==1]     # filter uptrend
        self.stock_data['starred point'] = np.nan
        prev_pbc = POS_INF
        chck_date_idx = np.nan
        star_lst =[]

        

        # converging bottom condition set 1:
        # 1. peak-to-bottom drop less than previous peak-to-bottom drop
        # 2. next little peak rise above previous little peak
        # 3.  cur price rise above prev big bottom * 1+ zigzag threshold (up trend already detected on that day)
        i=0
        while i< self.data_len-2:
            # if self.stock_data['zigzag'][i] ==1:
            #     prev_pbc = POS_INF
            #     chck_date_idx = np.nan
            
            if self.stock_data['zigzag'][i] ==-1:     # encounter big bottom
                prev_pbc = POS_INF
                prev_peak = POS_INF
                chck_date_idx = np.nan
                j = 1
                cur_big_btm = self.stock_data['Close'][i]
                while self.stock_data['zigzag'][i+j] != 1 :   # not encounter big peak yet
                    rise_back_offset=0
                    next_peak_offset =0
                    if self.stock_data['type'][i+j] == -1:
                        # 1. find prev little peak
                        l=0
                        while self.stock_data['type'][i+j+l] !=1: # find prev little peak
                            if i+j+l-1 >=0:
                                l-=1
                        prev_peak = self.stock_data['Close'][i+j+l]

                        rise_back_flag = False
                        break_pt_found_flag = False
                      
                        

                        while self.stock_data['type'][i+j+next_peak_offset] != 1: #find next little peak
                            if self.stock_data['Close'][i+j+next_peak_offset] >= prev_peak: # record closest date rise back to prev peak
                                rise_back_offset = next_peak_offset
                                rise_back_flag = True

                            next_peak_offset +=1
                            if i+j+next_peak_offset>self.data_len-1:
                                break
                        
                        #potential break point = next little peak or date of rise back to prev peak, which ever earlier
                        potential_bp = min(i+j+rise_back_offset, i+j+next_peak_offset)  

                       
                        if ( ( bp_filter_conv_drop and self.stock_data['p-b change'][i+j] > prev_pbc )
                            and ( bp_filter_rising_peak and rise_back_flag )
                            and ( bp_filter_uptrend and self.stock_data['Close'][potential_bp] > cur_big_btm*(1+zzthres) ) 
                            ):  
                            break_pt_found_flag = True
                            
                        if break_pt_found_flag:
                            chck_date_idx = self.stock_data.index[potential_bp]
                            star_lst.append(potential_bp)

                        prev_pbc = self.stock_data['p-b change'][i+j]


                    j = j + max(next_peak_offset, 1)
                    if i+j+1 > self.data_len-1:
                        break
                i=i+j
            i+=1


        star_col = self.stock_data.columns.get_loc('starred point')
        for item in star_lst:
            self.stock_data.iloc[item, star_col]= 1
        print("starred date list: ", star_lst)









        

    
    def plot_extrema(self, cols: list=[], plt_title: str='Extrema', annot: bool=True, text_box: str='', annotfont: float=6, 
                     showOption: str='show', savedir: str='', figsize: tuple=(36, 16)) :

        """
        default plot function, plot closing price of self.stock_data, self.smoothen_price and self.extrema
        
        Paramter
        -------
        cols: col names to plot | text_box: string in text box to print |
        showOption: 'show': show by plt.show, 'save': save graph without showing (suitable for env without GUI)
        savedir: dir to save plot |

         """
         
        
        plt.plot(self.stock_data['Close'], label='close price', color='blue', alpha=0.9)

        color_list=['fuchsia', 'cyan', 'tomato', 'peru', 'green', 'olive', 'tan', 'darkred']

        for i in range(0, len(cols)):    
            
            plt.plot(self.stock_data[cols[i]], 
                    label=cols[i] if isinstance(cols[i], str) else '',
                    alpha=0.6, linewidth=1.5, color=color_list[i])
            
        if self.smoothen_price is not None:
            plt.plot(self.smoothen_price[self.smoothen_price>0], color='gold')

        if self.extrema is not None:
            plt.plot(self.extrema[self.extrema["type"]==self.PEAK]['price'], "x", color='limegreen', markersize=5)
            plt.plot(self.extrema[self.extrema["type"]==self.BOTTOM]['price'], "x", color='salmon', markersize=5)
        
            ## Annotation ##
            annot_y_offset= self.stock_data['Close'][-1]*0.001
            if annot:
                
                for i in range(0, len(self.extrema)):
                    if self.extrema['type'][i]==self.PEAK:
                        
                        plt.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]),
                                (self.extrema.index[i], self.extrema['price'][i]+annot_y_offset), fontsize=annotfont, ha='left', va='bottom' )
                    if self.extrema['type'][i]==self.BOTTOM:
                        pbday = ", %d bar"%(self.extrema['peak-to-bottom day'][i]) if self.extrema['peak-to-bottom day'][i]>0 else ''
                        plt.annotate("{:.2f}".format(self.extrema['price'][i]) + ", {:.2%}".format(self.extrema['percentage change'][i]) 
                                 +pbday,
                                (self.extrema.index[i], self.extrema['price'][i]-annot_y_offset*3), fontsize=annotfont, ha='left', va='top' )

                
        
            ## Textbox on left-top corner ##
            # textbox is plot on relative position of graph regardless of value of x/y axis
            plt.text(0.01, 1,  text_box, fontsize=8, color='saddlebrown', ha='left', va='bottom',  transform=plt.gca().transAxes) 

            ## Textbox of drop from last high ##
            if self.peak_indexes is not None:
                #percentage change from last peak
                maxval=float('-inf')
                idx=-1
                while self.stock_data['type'][idx+1] != 1:        # find latest peak
                    
                    if self.stock_data['Close'][idx] > maxval:
                        maxval=self.stock_data['Close'][idx]
                        maxidx = idx
                    idx-=1

                
                #maxval = self.stock_data['Close'].iloc[list(range(self.peak_indexes[-1]-1, self.data_len))].max()
                #latest_max_date = self.stock_data['Close'].iloc[list(range(self.peak_indexes[-1]-1, self.data_len))].idxmax()
                print("latest price: ", self.stock_data['Close'].iloc[-1])
                perc = ( self.stock_data['Close'].iloc[-1] - maxval)/maxval              
                plt.text(0.9, 1.1, "lastest high: "+"{:.2f}".format(maxval), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.08, "current:  "+"{:.2f}".format(self.stock_data['Close'].iloc[-1]), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.text(0.9, 1.06, 'drop from last high: '+'{:.2%}'.format(perc), fontsize=7,  ha='left', va='top',  transform=plt.gca().transAxes)
                plt.scatter(maxidx, maxval, s=self.SCATTER_MARKER_SIZE, marker='d', color='lime')
                #if maxidx != 
                # TO DO
                plt.text(maxidx-pd.DateOffset(1), maxval + annot_y_offset*2, "{:.2f}".format(maxval), fontsize=7,  ha='left', va='bottom', color='limegreen')
                plt.text(self.stock_data.index[-1] + pd.DateOffset(3), self.stock_data['Close'][-1] *0.9 , 'drop from last high: \n'+'{:.2%}'.format(perc), fontsize=8)


        ### --- cutom plot here  --- ###

        #plt.plot(self.stock_data['buttered Close T=20'], alpha=0.8, linewidth=1.5, label='buttered Close T=20', color='cyan')
        #plt.plot(self.stock_data['buttered Close T=60'], alpha=0.8, linewidth=1.5, label='buttered Close T=60', color='magenta')

        
        plt.title(plt_title)
        
        # if showOption=='save':
        #     plt.savefig(f"{savedir}")
        # else:
        #     plt.show()

    def plot_zigzag(self, plt_title: str='Zigzag Indicator', annot: bool=True, text_box: str='', annotfont: float=6, showOption: str='show', savedir: str='') :
        #plt.figure(figsize=(24, 10), dpi=200)

        up_offset = self.stock_data['Close'][-1]*0.01
        down_offset = (-1)*self.stock_data['Close'][-1]*0.012
        
        #plt.plot(self.stock_data['Close'], label='close price', color='blue', alpha=0.9)
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==1].index, self.stock_data[self.stock_data['zigzag'] ==1]['Close'], color='g', s=self.SCATTER_MARKER_SIZE) #peak
        plt.scatter(self.stock_data[self.stock_data['zigzag'] ==-1].index, self.stock_data[self.stock_data['zigzag'] ==-1]['Close'], color='red',s=self.SCATTER_MARKER_SIZE)  #bottom
        plt.plot(self.stock_data[self.stock_data['zigzag'] !=0].index, self.stock_data[self.stock_data['zigzag'] !=0]['Close'], 
                 label='zigzag indicator',color='dimgrey', alpha=0.8, linewidth=0.8)
        
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
            print("breakpoint must be set before plot")
        
        filtered= self.stock_data[self.stock_data['starred point']>0]['Close']

        annot_y_offset = min(self.stock_data['Close'][-1]*0.02, 20)
        marker_y_offset = self.stock_data['Close'][-1]*0.01

      
        plt.scatter(self.stock_data[self.stock_data['starred point']>0].index, 
                    self.stock_data[self.stock_data['starred point']>0]['Close']*0.99, 
                    color='goldenrod', s=1/self.data_len*24000, marker=6)
        print("break point dates: ")
        
        for ind, val in filtered.items():   # item is float
            # print("type(item): ", type(item))
            # print(item)
            print(ind.strftime("%Y-%m-%d"))
            plt.annotate("Break pt: "+ind.strftime("%Y-%m-%d"), (ind, val-annot_y_offset), fontsize=6, ha='left', va='bottom', color='darkgoldenrod')
            




def runner(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzthres: float=0.09,
           all_vertex =False, 
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True,
           graph_showOption: str='show', graph_dir: str='../../zz.png', figsize: tuple=(36,24), annotfont: float=6) :
    """
    Parameter

    - method: options: 'ma', 'ema', 'dma', 'butter' |
    - T: day range of taking ma/butterworth low pass filter |
    - all_vertex: get all vertex from orginal stock price |
    - wind: window to locate extrema from approx. price
    """
    runner_start = time.time()
    
    stock = StockAnalyser(tickers, start, end)
    extra_col =[]
    smooth=False

    

    ## Parameter Checking

    if T<1:
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
    
    stock.set_zigizag(thres=zzthres)
    stock.set_trend()
    stock.set_breakpoint(zzthres=zzthres)
     
    print("-- Stock Data --")
    stock.print_stock_data()
    print("number of price point:", len(stock.get_stock_data()))

    print("-- Extrema --")
    print(tabulate(stock.get_extrema(), headers='keys', tablefmt='psql', floatfmt=("", ".2f","g", ".2%",)))
    print("number of extrema point:", len(stock.get_extrema()))

    rt = time.time()
    print("time for data manipulate: ", rt -runner_start)
    plot_start = time.time()

    plt.figure(figsize=(36, 16), dpi=200)
    stock.plot_extrema(cols=extra_col, plt_title=f"{tickers} {method}{T}", annot=True, text_box=f"{tickers}, {start} - {end}, window={window_size}", annotfont=annotfont, showOption=graph_showOption, savedir=graph_dir)
    stock.plot_zigzag(plt_title=f"{tickers} Zigzag Indicator", text_box=f"{tickers}, {start} - {end}, zigzag={zzthres*100}%")
    stock.plot_break_pt()
    plt.legend()
    plt.grid(which='major', color='lavender', linewidth=3)
    plt.grid(which='minor', color='lavender', linewidth=3)
    plot_end = time.time()

    if graph_showOption == 'save':
        plt.savefig(graph_dir)
    else:
        plt.show()

    


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
    #stock.plot_extrema(plt_title=f"{tickers}", annot=True)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='')
    parser.add_argument('--start',  type=str, default='2022-07-20')
    parser.add_argument('--end',  type=str, default='2023-07-22')
    args=parser.parse_args()

    stockticker=args.ticker
    stockstart = args.start
    stockend = args.end

    print("getting stock: ", stockticker)

    ## Here to try the class
    runner(stockticker, stockstart, stockend, method='ema', T=5, window_size=5, zzthres=0.09,
           graph_showOption='save', graph_dir=f'../../break point plot/zz.png', figsize=(120, 72), annotfont=5)

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

 



            