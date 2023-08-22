# Stock

## 2023 Version


main class: `StockAnalyser` in app/stock_analyser.py

## Goal

1. to draw bowls on historical stock price in different time frame

2. find break point

## Setting

**Break point condition**



For all bottoms during uptrend:

- filter 1: converging drop

  - peak-to-bottom drop less than previous peak-to-bottom drop

- filter 2: rising peak

  - Price rise above previous peak on or before next peak

- filter 3: uptrend detected
  
  - cur price rise above prev big bottom * 1+ zigzag up-threshold (only applicable if source of uptrend is zigzag)


**Extrema:**

- find extrema from selected price source using `scipy.signal.argrelextrema`

Source of uptrend options:
- zigzag indicator 
  OR 
- MACD Signal

Price source options of finding extrema:

- close price
- moving average (ma, ema, dma, lwma)
- close price filered by Butterworth low-pass filter


## How to use

go to app/stock_analyser.py 

### command line run options

- analyse one stock

```
python stock_analyser.py --ticker=PDD --start=2022-08-01 --end=2023-08-01 --graph_dir=../dir/graph_name
```

- analyse list of stock from txt file

```
python stock_analyser.py --stocklist_file=../dir/stock.txt --start=2022-08-01 --end=2023-08-01 --graph_dir=../dir/graph_name
```

- analyse list of stock by default stock list in code

```
python stock_analyser.py --start=2022-08-01 --end=2023-08-01 --graph_dir=../dir/

```

### Use the class

**main runner method**
```
def default_analyser_runner(tickers: str, start: str, end: str, 
           method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
            macd_signal_T: int=9,
            bp_trend_src: str='signal',
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True, bp_filter_uptrend: bool=True,
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(30,30), annotfont: float=6):
   

  def default_analyser(self, tickers: str, start: str, end: str,
            method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
            bp_trend_src: str='signal',
           bp_filter_conv_drop: bool=True, bp_filter_rising_peak: bool=True, bp_filter_uptrend: bool=True,
           extra_text_box:str='',
           graph_showOption: str='show', graph_dir: str='../../untitled.png', figsize: tuple=(36,24), annotfont: float=6) ->pd.DataFrame:


```
return: pd.Dataframe of stock information with peak, bottom, breakpoint etc

Parameter of `runner_analyser`
- `tickers`: stock symbol
- `method`: price source to calculate extrema
- `T`: period of moving average if method set to 'ma', 'ema' or any kind with period required (no effect if method set to 'close')
- `window_size`: window size to locate extrema from price source specified in `method` (no effect if method set to 'close')
- `zzupthres`, `zzdownthres`: up/down threshold of zigzag indicator
- `bp_trend_src`:  source of uptrend signal, 'zz': zigzag indicator, 'signal': MACD signal
- `bp_filter_conv_drop`: to apply converging bottom filter or not when finding breakpoint
- `bp_filter_rising_peak`: to apply rising peak filter or not when finding breakpoint
- `bp_filter_uptrend`: to apply uptrend detected filter or not when finding breakpoint (only have effect if trend source is zigzag)
- `extra_text_box`: text to print on graph (left top corner)
- `graph_showOption`: 'save', 'show', 'no'
- `graph_dir`: dir to save graph
- `figsize`: figure size of graph 
  - recommend: 1-3 months: figsize=(36,16)
- `annotfont`: font size of annotation of peak bottom 
  - recommend: 4-6


### Example: using `runner_analyser`

**E.g. 1**: PDD, 12 months plot extrema from close price, find breakpoint with converging drop filter and rising peak filter applied, use MACD Signal line as uptrend signal

```
stock = StockAnalyser()   # init class
result_df = stock.default_analyser(
    tickers='PDD', 
    start='2022-08-01', 
    end='2023-08-01',
    method='close', 
    bp_trend_src='signal',
    bp_filter_conv_drop=True, 
    bp_filter_rising_peak=True,
    graph_showOption='save'
)

```

**E.g. 2**: AMD, 12 months plot extrema from ema9, window size set as 5, find breakpoint with converging drop filter=True, rising peak filter=False, use zigzag indicator as uptrend signal

```
stock = StockAnalyser()   # init class
result_df = stock.default_analyser(
    tickers='AMD', 
    start='2022-08-01', 
    end='2023-08-01',
    method='ema', T=9, window_size=5,
    bp_trend_src='zz',
    bp_filter_conv_drop=True, 
    bp_filter_rising_peak=False,
    bp_filter_uptrend=True,
    graph_showOption='save'
)

```

## Example Result

result of plotting extrema from different price source: 
- `result_plot_extrema.pdf`

## Techniques Studied

- Moving Averages (MA, EMA, DMA)
- Butterworth Low Pass Filter
- polyfit
- linear convolution with np.blackman

detail discussion of pros and cons of different techniques see `technique_and_theory.md`


## Bug to be solved:
- ema (hence MACD) in early segment of stock data is not accurate, since ema is calculate base on yesturday's ema, so much earlier data before the specified start is required to get an accurate ema

## 2021 Version
---

- [Stock](#stock)
- [Logic and Design](#logic-and-design)
  - [Peaks and Bottoms](#peaks-and-bottoms)
    - [Example](#example)
    - [Limitations Using Blackman Window](#limitations-using-blackman-window)
    - [Limitations Using Polynomial Regression](#limitations-using-polynomial-regression)
  - [Trend](#trend)
    - [Example](#example-1)
    - [Limitations](#limitations)
- [Reference](#reference)
  - [Smoothing the Data ("Noise" Reduction)](#smoothing-the-data-noise-reduction)
  - [Linear Regression](#linear-regression)

---

# Logic and Design

## Peaks and Bottoms

![](./docs/Screenshot%202021-08-16%20184841.png)

- First smooth the stock data and remove the "noise". (See [Smoothing the Data ("Noise" Reduction)](#smoothing-the-data-noise-reduction))
- Find the approximate peaks and bottoms using the smoothed trend
- Find the actual peaks and bottoms using the real data

### Example

![](./docs/NVDA%20Peaks%20and%20Bottoms.png)

### Limitations Using Blackman Window

For a longer period of time, the value of `smooth_data_N` and `find_extrema_interval` has to change to other value.

`"NVDA", start="2010-01-01", end="2021-08-16"`

`smooth_data_N = 15, find_extrema_interval = 5`

![](./docs/NVDA%20Peaks%20and%20Bottoms%202.png)

Solution:

`smooth_data_N = 100, find_extrema_interval = 25`

![](./docs/NVDA%20Peaks%20and%20Bottoms%203.png)

### Limitations Using Polynomial Regression

The degree cannot be too large when there are a lot of data.

It might not work when the given period (number of record) is too large. 

---

## Trend

- Use linear regression and plot a best-fit line.

### Example

`stock_info = yf.download("^HSI", start="2000-01-01", end="2003-06-15")`

![HSI Linear Regression](./docs/HSI%20Linear%20Regression.png)

`stock_info = yf.download("NVDA", start="2021-01-01", end="2021-08-16")`

![NVDA Linear Regression](./docs/NVDA%20Linear%20Regression.png)

### Limitations

The best-fit line might not be perfect.

`stock_info = yf.download("AAPL", start="2000-01-01", end="2021-08-16")`

![Inaccuracy for long period](./docs/AAPL%20Linear%20Regression.png)

`stock_info = yf.download("AAPL", start="2000-01-01", end="2003-06-15")`

![Sudden drop](./docs/AAPL%20Linear%20Regression%202.png)

---

# Reference

## Smoothing the Data ("Noise" Reduction)

Current approach in smoothing the data (Blackman Window):

https://books.google.com.hk/books?id=m2T9CQAAQBAJ&pg=PA189&lpg=PA189&dq=numpy+blackman+and+convolve&source=bl&ots=5lqrOE_YHL&sig=ACfU3U3onrK4g3uAo3a9FLT_3yMcQXGfKQ&hl=en&sa=X&ved=2ahUKEwjE8p-l-rbyAhVI05QKHfJnAL0Q6AF6BAgQEAM#v=onepage&q=numpy%20blackman%20and%20convolve&f=false

Another approach in smoothing the data with polynomial regression:

https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

Another approach in smoothing the data:

https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

## Linear Regression

https://medium.com/analytics-vidhya/stock-prediction-using-linear-regression-cd1d8351f536

https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d
