# Stock

## 2023 Version


main class: 

`StockAnalyser` in app/stock_analyser.py

`BackTest` in app/back_test.py

`StockAccount` in app/back_test.py

## 1. Goal

1. to draw bowls on historical stock price in different time frame

2. find break point (buy point)

3. set up sell strategy

4. conduct back test and calculate revenue

### 2. Buy point filters

`BuyptFilter` in app/stock_analyser.py

| filter | description |
| ------ | ------ |
| IN_UPTREND |  - set all points in up trend as buy points<br><br> - up trend is defined by specified column name in stock_data, with value >0 indicate uptrend, < 0 indicate downtrend,<br><br> - default is slope of MACD signal line if column of trend source not specified <br><br> - this filter has no effect if either CONVERGING_DROP or RISING_PEAK is used, since  CONVERGING_DROP and  RISING_PEAK intrinsically only looks for bottoms in uptrend  |
|   CONVERGING_DROP     |   For all bottoms during uptrend:<br><br>if (peak-to-bottom drop) <  (previous peak-to-bottom drop)<br><br>the point (rise above previous peak) or (reach next peak), which ever earlier, is defined as break point   |
| RISING_PEAK  | For all bottoms during uptrend:<br><br>if rise above previous peak before next peak, <br><br> the point rise above previous peak is defined as break point |
|MA_SHORT_ABOVE_LONG |   all points for corresponding ma in "ma short" > "ma long" is set as buy points <br><br>e.g. ma short=[3, 20]<br> ma long = [9, 50]<br> all points where ma3> ma9 and ma20>ma50 are set as buy points|


### 3. Sell Strategy
`SellStrategy` in app/back_test.py
| Strategy | description |
| ------ | ------ |
|   TRAILING_STOP     |    Trailing Stop-loss (sell whenever drop n% from high point after buy)    |
|    HOLD_TIL_END    |  Hold until last day       |
| PROFIT_TARGET|  sell when profit target is reached|
| FIXED_STOP_LOSS| sell when drop n% from buy price|
|    TRAILING_AND_FIXED_SL | sell when at least one of trailing Stop-loss or fixed stop loss condition met|
| TRAIL_FIX_SL_AND_PROFTARGET|sell when at least one of trailing Stop-loss, fixed stop loss or profit target met|
|MIX | not defined yet (to be used in future) |




## 4. How to use

### 4.1. run in command line 

To run `stock_analyser`
go to app/stock_analyser.py 

### command line run options

- analyse one stock

```
python stock_analyser.py -t=PDD -s=2022-08-01 -e=2023-08-01 -g=../graph_dir -v=./csv_dir
```

- analyse list of stock from txt file

```
python stock_analyser.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../graph_dir -v=./csv_dir
```

to run `back_test`

- run pdd, 1 year, with captial=$10000, no need to plot graph

```
python backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../back_test_result -g=../graph_dir
```

- run  list of stock from txt file, 1 year, with captial=$10000, save graph

```
python backtest.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-16 -c=10000 -o=save -v=../back_test_result -g=../graph_dir
```
### 4.2. run by config in command line (.json)

**Example config (Json)**

```
{
  "ticker": "pdd",
  "start": "2023-08-01",
  "end": "2023-08-20",
  "ma short": [3, 20],
  "ma long": [9, 50],
  "plot ma": ["ma3", "ema9", "ma15"],
  "buy point filters": [
    "IN_UPTREND",
    "CONVERGING_DROP",
    "RISING_PEAK",
    "MA_SHORT_ABOVE_LONG"
  ],
  "buy strategy": "FOLLOW_BUYPT_FILTER",
  "sell strategy": ["TRAILING_STOP"],
  "stop loss percent": 0.05,
  "graph show option": "save",
  "graph dir": "../result",
  "csv dir": "../result",
  "print all ac": false
}
```

| param | description |  data type|
| ------ | ------ |------ |
|ticker|     stock ticker   |  str  |
|  start      |    test start date    |  str (yyyy-mm-dd)  |
|  end      |    test end date    |  str (yyyy-mm-dd)  |
|    ma short    |    short ma to use in MA_SHORT_ABOVE_LONG filter    |   list of int |
|   ma long     |   long ma to use in MA_SHORT_ABOVE_LONG filter      |    list of int <br>e.g. ma short=[3, 20]<br> ma long = [9, 50]<br> ==>all points where ma3> ma9 and ma20>ma50 will be set as buy points |
|   plot ma     |     extra ma to plot on graph, but will not affect buy point     |  list of str<br>e.g.['ma3', 'ema9']  |
|   buy point filters     |    filters to find buy point, buy point are set if all filter met    | list of str<br> options:<br> "IN_UPTREND"<br> "CONVERGING_DROP"<br> "RISING_PEAK"<br>"MA_SHORT_ABOVE_LONG"  |
|buy strategy | buy strategy, currently only support follow buy point filter |str<br> currently only option is: "FOLLOW_BUYPT_FILTER" |
| sell strategy| sell strategy |str<br> e.g. "TRAILING_STOP"<br>options:<br>see above section Sell Strategy |
| stop loss percent| percentage of trail stop loss| float |
| fixed stop loss percent|percentage of fixed trail stop loss| float |
| profit target|  prfot target percentage <br>e.g. profit target=0.3 means sell when price reach 130% of buy price|float |
|graph show option | options of how to handle graph plotting| str <br>options:<br>"save"<br>"show"<br>"no" |
|graph dir |directory to save graph |str |
| csv dir|directory to save csv |str |
|print all ac | if run list of stock, to print stock data and roll result of each stock or not  | bool|

**More config example:**
- see folder app/configs

### 4.3. Import class

- see example: `app/stock_analyser_backtest_demo.ipynb`

## 5. More Settings

### 5.1. Source of Extrema:

- find extrema from selected price source using `scipy.signal.argrelextrema`

- default source: close price

- Source of Extrema controlled by: `method` in StockAnalyser.default_analyser 

- `method` options: 'close', 'ma', 'ema', 'dma', 'butter', 


| source to find extrema | description |
| ------ | ------ |
|  'close'      |   directly use close price to find extrema (default setting)   |
|  'ma'      |    use ma to find extrema   |
|  'ema'      |    use ema to find extrema   |
|  'dma'      |    use dma to find extrema   |
|'butter' | apply Butterworth Low Pass Filter on close price, then use it to find extrema |

`T` in StockAnalyser.default_analyser
- the period for ma / ema / dma / butter in `method`
- has no effect if `method` set as 'close'

### 5.2. Source of uptrend

- controlled by: `trend_col_name` in StockAnalyser.default_analyser 
- options: any column, with value >0 indicate uptrend, < 0 indicate downtrend
- default: 'slope signal' which indicate slope of MACD Signal Line [see here](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd)




### 5.3. Parameter of StockAnalyser.default_analyser

main runner of class `StockAnalyser`: `StockAnalyser.default_analyser`

TO BE WRITTEN
| param | description |
| ------ | ------ |
|        |        |
|        |        |


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
