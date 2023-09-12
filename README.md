- [1. Abstract](#1-abstract)
- [2. Buy Strategy](#2-buy-strategy)
- [3. Sell Strategy](#3-sell-strategy)
- [4. How to use](#4-how-to-use)
- [5. Details of `StockAnalyser.stock_data`](#5-details-of-stockanalyserstock_data)
- [6. Details of `StockAccount.txn`](#6-details-of-stockaccounttxn)
- [7. Buy Sell Logic](#7-buy-sell-logic)
- [8. Log](#8-log)
- [9. Advanced Settings](#9-advanced-settings)
- [10. Example Result](#10-example-result)
- [11. Unit Test](#11-unit-test)
- [12. Techniques Studied](#12-techniques-studied)
- [13. Bug to be solved:](#13-bug-to-be-solved)
- [14. Further Study](#14-further-study)


# Stock Analysis and Back Test Project

##  1. <a name='Abstract'></a>Abstract

This project aimed to define some buy and sell strategies for stock trading, and provide a class to test the profit of each strategy by back test simulation.

The project look for buy points and sell points on historic stock price, user can conduct a back test by specifying buy, sell strategy and period.

###  1.1. <a name='Mainclassesintheproject'></a>Main classes in the project

 `StockAnalyser` in app/stock_analyser.py 
- provide analysis information like moving average, MACD, Zizag indicator etc. in a form of a table
- details refer to section [Details of `StockAnalyser.stock_data`](#DetailsofStockAnalyser.stock_data)

`BackTest` in app/back_test.py
- conduct back test with specified buy, sell strategy and period on a stock or group of stocks, calculate gain / loss over a period

`StockAccount` in app/back_test.py
- save the result of back test of a group of stocks
- details refer to section  [Details of `StockAccount.txn`](#DetailsofStockAccount.txn)

###  1.2. <a name='filestructure'></a>file structure

```
.
|   result_plot_extrema.pdf
|   technique_and_theory.md
|   test_script.sh        <--- script to test running program in command line
|   
+---app
|   |   backtest.py        <--- conduct back test
|   |   requirements.txt
|   |   stock_analyser.py  <--- generate stock analysis information
|   |   txt_to_list_transformer.py
|   |   __init__.py
|   |   
|   |               
|   +---configs             <--- json config example
+---demo
|       demo_stock_analyser_backtest.ipynb
|       demo_stock_analyser_backtest.py
|       
+---example                 <--- example result
|       example_result.md
|           
\---tests                   <--- pytest clients
    |   test_backtest.py
    |   test_stock_analyser.py
    |   __init__.py  

```

##  2. <a name='BuyStrategy'></a>Buy Strategy

Available buy strategies are defined in `BuyptFilter` in app/stock_analyser.py

Other than common indicators looking for buy signal such as MACD, MA crossover, this project introduce a way to find break point of price by finding "converging drop pattern"  

the method identify all drops (bowl shape) in the price curve, then identify the points meet the following condition:

- consecutive bowls
- peak-to-bottom drop of the bowl less than peak-to-bottom drop of the previous bowl 

###  2.1. <a name='Allavailablefiltersforfindingbuypoints'></a>All available filters for finding buy points

| filter | description |
| ------ | ------ |
| IN_UPTREND |  - set all points in up trend as buy points<br><br> - up trend is defined by specified column name in stock_data, with value >0 indicate uptrend, < 0 indicate downtrend,<br><br> - default is slope of MACD signal line if column of trend source not specified <br><br> - this filter has no effect if either CONVERGING_DROP or RISING_PEAK is used, since  CONVERGING_DROP and  RISING_PEAK intrinsically only looks for bottoms in uptrend  |
|   CONVERGING_DROP     |   For all bottoms during uptrend:<br><br>if (peak-to-bottom drop) <  (previous peak-to-bottom drop)<br><br>the point (rise above previous peak) or (reach next peak), which ever earlier, is defined as break point   |
| RISING_PEAK  | For all bottoms during uptrend:<br><br>if rise above previous peak before next peak, <br><br> the point rise above previous peak is defined as break point |
|MA_SHORT_ABOVE_LONG |   all points for corresponding ma in "ma short" > "ma long" is set as buy points <br><br>e.g. ma short=[3, 20]<br> ma long = [9, 50]<br> all points where ma3> ma9 and ma20>ma50 are set as buy points|




##  3. <a name='SellStrategy'></a>Sell Strategy

sell points are identified according to the sell strategy set.

Available sell strategies in `SellStrategy` in app/back_test.py
| Strategy | description |
| ------ | ------ |
| DEFAULT| currently set as = TRAILING_STOP|
|   TRAILING_STOP     |    Trailing Stop-loss (sell whenever drop n% from high point after buy)    |
|    HOLD_TIL_END    |  Hold until last day       |
| PROFIT_TARGET|  sell when profit target is reached|
| FIXED_STOP_LOSS| sell when drop n% from buy price|
|    TRAILING_AND_FIXED_SL | sell when at least one of trailing Stop-loss or fixed stop loss condition met|
| TRAIL_FIX_SL_AND_PROFTARGET|sell when at least one of trailing Stop-loss, fixed stop loss or profit target met|
|MIX | not defined yet (to be used in future) |




##  4. <a name='Howtouse'></a>How to use

###  4.1. <a name='Runstock_analyser.py'></a>Run `stock_analyser.py`
####  4.1.1. <a name='Description'></a>Description

 `stock_analyser.py` analyse a stock over a period and produce analysis information like peak, bottom points, up / down trend, zigzag indicator, MACD etc.

####  4.1.2. <a name='Runstock_analyser.pyincommandline'></a>Run `stock_analyser.py` in command line

##### Arguments

arguments of running `stock_analyser.py` in command line

|argument| description|example|
|-----|-----|-----|
|`-t` `--ticker` | stock ticker| PDD
|`-s` `--start` | start date| 2023-01-01|
| `-e` `--end` | end date|2023-08-01|
| `-f` `--stocklist_file`| stock list file (.txt)|./stock_list.txt<br>([exmaple file](app/configs/hot25stocks.txt?ref_type=heads))|
|`-v` `--csv_dir` | file directory of stock data and roll result csv to save in|../result|
| `-g` `--graph_dir` | file directory of graph to save in|../graph_dir|
|`-o` `--showopt` | graph show option |"save" - save to graph_dir<br>"show" - show by plot.show()<br>"no" - don't plot graph   |

##### Example

go to app/stock_analyser.py 

example commands

- analyse one stock

```
python stock_analyser.py -t=PDD -s=2022-08-01 -e=2023-08-01 -g=../graph_dir -v=../csv_dir
```

- analyse list of stock from txt file

```
python stock_analyser.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../graph_dir -v=../csv_dir
```

####  4.1.3. <a name='Outputs'></a>Outputs

1. print analysis data in .csv file 
     - default file name and directory: `../../result/stock_data_{stock ticker}_{start date}_{end date}.csv`
     - [click here to see details of all columns in stock data table](#columnsinStockAnalyser.stock_data:)
     - [click here for example file](example/stock_data_tsla_2022-08-01_2023-08-25_example.csv?ref_type=heads)
2. produce analysis graph in .png file
    - default file name and directory: `../../result/{stock ticker}_{start date}_{end date}.png`
    - [click here for example graph](example/tsla_2022-08-01_2023-08-25_bp_by_peak_bottom.png?ref_type=heads)
3. print analysis data and logging  in ./log/*.log 



###  4.2. <a name='Runbacktest.py'></a>Run `backtest.py`

####  4.2.1. <a name='Description-1'></a>Description

`backtest.py` obtain analysis data from `stock_analyser.py`, then conduct back test to simulate trading a stock. 

In the back test, user can specify which stock to trade, capital to put in, buy and sell strategy etc., and get a trading simulation in the form of a transaction table saved in `StockAccount.txn`.

####  4.2.2. <a name='Runbacktest.pyincommandline'></a>Run `backtest.py` in command line

##### Arguments

|argument| description|example|
|-----|-----|-----|
|`-t` `--ticker` | stock ticker| PDD
|`-s` `--start` | start date| 2023-01-01|
| `-e` `--end` | end date|2023-08-01|
| `-c` `--capital` | capital| 10000|
| `-f` `--stocklist_file`| stock list file (.txt)|./stock_list.txt|
| `-j` `--configfile`| config file (.json)|./config.json|
|`-v` `--csv_dir` | file directory of stock data and roll result csv to save in|../result|
| `-g` `--graph_dir` | file directory of graph to save in|../graph_dir|
|`-o` `--showopt` | graph show option |"save" - save to graph_dir<br>"show" - show by plot.show()<br>"no" - don't plot graph   |

##### Example

go to app/backtest.py

- run pdd, 1 year, with captial=$10000, no need to plot graph

```
python backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../back_test_result
```

- run  list of stock from txt file, 1 year, with captial=$10000, save graph

```
python backtest.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-16 -c=10000 -o=save -v=../back_test_result -g=../graph_dir
```
####  4.2.3. <a name='Runbyconfig.jsonincommandline'></a>Run by config (.json) in command line 

##### Description

- pass argument required by backtest.py by a json config file

##### Table of parameters in JSON config file


| param | description |  data type|example| required|
| ------ | ------ |------ |------ |------ |
|ticker|     stock ticker   |  str<br>/<br>list of str |"PDD"<br>["PDD", "TSLA", "VRTX"] |yes|
|  start      |    test start date<br> (yyyy-mm-dd)    |  str  | "2023-01-01" |yes|
|  end      |    test end date <br>(yyyy-mm-dd)   |  str | "2023-08-01" |yes|
|  capital      |    initial capital for backtest   |  int / float | 10000 |no<br>- if not set, set to default as 10000|
|    ma short    |    short ma to use in MA_SHORT_ABOVE_LONG filter    |   list of int |[3]<br>[3,20]|no|
|   ma long     |   long ma to use in MA_SHORT_ABOVE_LONG filter      |    list of int |[9]<br>[9,50]<br>e.g. ma short=[3, 20]<br> ma long = [9, 50]<br> ==>all points where ma3> ma9 and ma20>ma50 will be set as buy points |no|
|   plot ma     |     extra ma to plot on graph, but will not affect buy point     |  list of str<br>|["ma3", "ema9"]  |no|
|   buy point filters     |    filters to find buy point, buy point are set if all filter met    | list of str|  "IN_UPTREND"<br> "CONVERGING_DROP"<br> "RISING_PEAK"<br>"MA_SHORT_ABOVE_LONG"  |no <br> - if no filter set, no buy points will be found|
|buy strategy | buy strategy, currently only support follow buy point filter |str | "FOLLOW_BUYPT_FILTER"<br>(the only option currently)  |no|
| sell strategy| sell strategy |str| "DEFAULT"(currently set as trailing stop)<br>"TRAILING_STOP"<br>"HOLD_TIL_END"<br>"PROFIT_TARGET"<br>"FIXED_STOP_LOSS"<br>"TRAILING_AND_FIXED_SL"<br>"TRAIL_FIX_SL_AND_PROFTARGET" |no <br> - if no sell strategy, will hold until end|
| stop loss percent| percentage of trail stop loss| float |0.05|no <br> if not set but sell strategy involved trail stop, set to default as 0.05|
| fixed stop loss percent|percentage of fixed trail stop loss| float |0.03|**yes** if sell strategy involve fixed stop loss, else **no**|
| profit target|  prfot target percentage|float |0.3<br>(means sell when price reach 130% of buy price)|**yes** if sell strategy involve profit target, else **no**|
|graph show option | options of how to handle graph plotting| str |"save" - save to graph_dir<br>"show" - show by plt.show()<br>"no" - don"t plot graph  |no<br> - if not set, default="no"|
|graph dir |directory to save graph |str |"../../result"|no<br> - if not set, default="../../result"|
| csv dir|directory to save csv |str |"../../result"|no<br> - if not set, default="../../result"|
|print all ac | if run list of stock, to print stock data and roll result of each stock or not  | bool|true|no<br> if not set, default=false|
| figure size x | x-dimension size of graph  |int | suggested value<br>within 3 months: `20`<br>3 months up: `36` | no <br>if not set, default=36|
| figure size y | y-dimension size of graph  |int | suggested value<br>within 3 months: `10`<br>3 months up: `16` | no<br>if not set, default=16|
| graph dpi | dpi of graph  |int | suggested value<br>within 3 months: `100`<br>3 months up: `200` | no<br>if not set, default=200|
|graph font size | font size of annotation of graph  |float | 4<br>suggested value<br>dpi 0-100: 8-10<br>dpi 100-200: 5-8<br>dpi 200+: 3-4 | no<br>if not set, default=4|

##### Example Command

- use configs/backtest_config_example.json as config file

```
python backtest.py -j=./configs/backtest_config_example.json
```

##### Example config (Json)

```
{
  "ticker": "pdd",
  "start": "2023-08-01",
  "end": "2023-08-20",
  "capital": 20000,
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
  "sell strategy": "TRAIL_FIX_SL_AND_PROFTARGET",
  "stop loss percent": 0.05,
  "fixed stop loss percent": 0.03,
  "profit target": 0.3,
  "graph show option": "save",
  "graph dir": "../unit_test_result",
  "csv dir": "../unit_test_result",
  "print all ac": false,
  "figure size x": 36,
  "figure size y": 16,
  "graph dpi": 100,
  "graph font size": 4
}
```

- More config example:
  - see folder [app/configs](app/configs?ref_type=heads)
####  4.2.4. <a name='Parameterparsinglogicwhenrunincommandline'></a>Parameter parsing logic when run in command line

All param has to come from same source (e.g. all from command line, or all from config)

for example, you cannot set stock ticker in direct pass from command line, but set buy point filter in config

if parameters are passed in from both side, only parameters from config file will be used, all parameters from command line will be ignored

####  4.2.5. <a name='Outputs-1'></a>Outputs
1. print analysis data in .csv file 
   - default file name and directory: `../../result/stock_data_{stock ticker}_{start date}_{end date}.csv`
   -  [click here to see details of all columns in stock data table](#columnsinStockAnalyser.stock_data:)
   - [click here for example file](example/stock_data_tsla_2022-08-01_2023-08-25_example.csv?ref_type=heads)
2. produce analysis graph in .png file
   - default file name and directory: `../../result/{stock ticker}_{start date}_{end date}.png`
   - [click here for example graph](example/tsla_2022-08-01_2023-08-25_bp_by_peak_bottom.png?ref_type=heads)
3. print back test table in .csv file
   - default file name and directory: `../../result/roll_result_{stock ticker}_{start date}_{end date}.csv`
   - [click here to see details of all columns in roll result tabble](#columnsinStockAccount.txn:)
   - [click here for eample file](example/roll_result_tsla_2022-08-01_2023-08-25_example.csv?ref_type=heads)
4. print analysis data and logging  in ./log/*.log 



###  4.3. <a name='Importclass'></a>Import class

- use `stock_analyser.py` and `backtest.py` by importing class
- See demo files: 
  - [demo/demo_stock_analyser_backtest.ipynb](/demo/demo_stock_analyser_backtest.ipynb)
  - [demo/demo_stock_analyser_backtest.py](/demo/demo_stock_analyser_backtest.py)
  
##  5. <a name='DetailsofStockAnalyser.stock_data'></a>Details of `StockAnalyser.stock_data`


main class of `stock_analyser.py` is `StockAnalyser`, which contain attribute `stock_data`

`stock_data` contain the analysis data of the stock

- data type of `stock_data` : 
  - pd.DataFrame

- produce:  
  - stock_data csv file ( `../../result/stock_data_{stock ticker}_ {start date}_{end date}.csv` )
- sample file:  
  - [/example/stock_data_tsla_2022-08-01_2023-08-25_example.csv](example/stock_data_tsla_2022-08-01_2023-08-25_example.csv?ref_type=heads)

###  5.1. <a name='columnsinStockAnalyser.stock_data:'></a>columns in `StockAnalyser.stock_data`:

 | column name  | description |must appear via run default_analyser?| produce by wich method|
| ------ | ------ |------ |------ |
|   Date        | date  (index column)    |  yes| `download`  |
|   close     |  close price of stock in that day  | yes|  `download` |
|  type |  peak / bottom<br>1=peak<br>0=bottom    |yes |  `set_extrema` |
| p-b change  | percentage change of price between peak-bottom    | yes| `set_extrema`  |
|  ma{T}<br>e.g. ma9 | simple moving average    | no<br>will appear if provide in "ma short", "ma long", "plot ma" in config|  `add_column_ma` |
| zigzag  |  zigzag indicator<br>1=peak<br>0=bottom<br>come from [this module](https://pypi.org/project/zigzag/)   | yes| `set_zigizag_trend`  |
|  zz trend |   up/down trend according to zigzag indicator <br>-1=downtrend <br>1=uptrend  |yes | `set_zigizag_trend`  |
|  ema12<br>ema26 | exponential moving average 12 and 26    | yes|  `__add_col_macd_group` |
| MACD  |  MACD  <br>[see here](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd) to learn about MACD and MACD Signal | yes| `__add_col_macd_group`  |
| signal  |  MACD Signal Line  |yes| `__add_col_macd_group`  |
| slope MACD  |  slope of MACD   |yes | `__add_col_macd_group`  |
|  slope signal |  slope of MACD Signal   | yes| `__add_col_macd_group`  |
|  buy pt |  is buy point or not<br>   |yes | `__set_breakpoint`  |
| day of interest  |  mark day of interest <br> currently only marked buy point as DayType.BUYPT  | - need to pass at least one "buy point filter" to get buy point<br>- otherwise this will be empty column|  `set_buy_point` |
|ma{A} above ma{B}| is ma{A} above ma{B}|no <br>will appear if provide in "ma short", "ma long"|`add_col_ma_above`|



##  6. <a name='DetailsofStockAccount.txn'></a>Details of `StockAccount.txn`

class `StockAccount` in `backtest.py` contain the back test information of each stock.

`StockAccount.txn` contain the roll result of back test


- Data type of `StockAccount.txn`:
  - pd.DataFrame
- produce:
  - roll result of back test (`../../result/roll_result_{stock ticker}_{start date}_{end date}.csv`)
- sample file:
  -  [/exmaple/roll_result_tsla_2022-08-01_2023-08-25_example.csv](example/roll_result_tsla_2022-08-01_2023-08-25_example.csv?ref_type=heads)
- If list of stock is run, will produce a csv file record revenue of each stock: `../../result/all_revenue.csv`

  - sample file of [all_revenue](example/all_revenue_top50_SP500_period1_MACD_condition11.csv?ref_type=heads)

###  6.1. <a name='columnsinStockAccount.txn:'></a>columns in  `StockAccount.txn`:

 | column name  | description |
| ------ | ------ |
|   Date        | date  (index column)  
|cash|amount of cash  |
|close price |close price of the considered stock on that day |
|MV | market value of holding on that day <br>calculated by close price|
|action | action taken on that day <br> (only the last action taken will show up, in current setting only one action will be taken each day, see [Buy Sell Logic](#BuySellLogic) Section for detail)  <br>example<br>Action.BUY<br>Action.SELL|
|deal price |deal price of buy / sell stock |
|txn amt |transaction amount<br> =number of share in transaction * deal price |
|total asset | =cash + market value of holding|
|latest high | highest close price reached after the latest buy action |
|+- | only appear in row that have sell action <br>=percentage gain in that buy-sell pair|
| trigger| trigger reason of sell<br>example<br>trail stop (reach trailing stop loss price)<br>fixed SL (reach fixed stop loss price)<br>profit target (reach profit target)<br> last day (forced sell on last day of back test) |
##  7. <a name='BuySellLogic'></a>Buy Sell Logic

conduct in `roll` methed of `BackTest`

pseudo code of `roll`:

```
for each row in stock_data:
  # check sell
  if has holding:
    check_sell()  # check need to sell or not
    if no holding after check_sell():
      iteration jump to next buy point
      continue

  # check buy
  if that day is buy point and not `is_holding`:
    check how many share can buy with available cash
    if share able to buy >=1:
      buy()

  # force sell on last day    
  if is last day:
    force sell

  # fast forward
  if no holding and no more buy point onwards:
    break


```

**buy**:

- use as much available cash as possible to buy stock
- deal at close price
- only check buy when no holding (can be adjusted by removing  not `is_holding` clause in check buy)

**sell**:

- check whether need to sell on day with holding
- sell 100% of stock each time (can be adjusted in param `portion` in `BackTest.sell`)  
- deal at trigger price

|trigger reason| trigger price|
|-----|-----|
| trailing stop loss| latest high price * (1- trail_loss_percent) -> round down to 2d.p.|
| fixed stop loss| buy price * (1- fixed stop loss) -> round down to 2d.p.|
|profit target | buy price * (1+ profit target) -> round up to 2d.p. |
| last day | close price|


##  8. <a name='Log'></a>Log

- log are saved in `./log/`

##  9. <a name='AdvancedSettings'></a>Advanced Settings

###  9.1. <a name='SourceofExtrema:'></a>Source of Extrema:

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

###  9.2. <a name='Sourceofuptrend'></a>Source of uptrend

- controlled by: `trend_col_name` in StockAnalyser.default_analyser 
- options: any column, with value >0 indicate uptrend, < 0 indicate downtrend
- default: 'slope signal' which indicate slope of MACD Signal Line [see here](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd)




###  9.3. <a name='ParameterofStockAnalyser.default_analyser'></a>Parameter of StockAnalyser.default_analyser

main runner of class `StockAnalyser`: `StockAnalyser.default_analyser`



Parameter of `StockAnalyser.default_analyser`
- `tickers`: stock symbol
- `method`: price source to calculate extrema
- `T`: period of moving average if method set to 'ma', 'ema', 'butter' or any kind with period required (no effect if method set to 'close')
- `window_size`: window size to locate extrema from price source specified in `method` (no effect if method set to 'close')
- `smooth`: if set to true, will set smoothen price and save to `self.smoothen_price`
  - has no effect on break point
  - smooth close price by linear convolution with np.blackman

- `smooth_ext`: smooth extend to apply if `smooth`=true
- `zzupthres`, `zzdownthres`: up/down threshold of zigzag indicator
- `trend_col_name`:  source of uptrend signal, 'zz': zigzag indicator, 'signal': MACD signal
- `bp_filters`: set class `BuyptFilter` to use
- `extra_text_box`: text to print on graph (left top corner)
- `graph_showOption`: 'save', 'show', 'no'
- `graph_dir`: dir to save graph
- `csv_dir`: dir to save csv
- `figsize`: figure size of graph 
  - recommend: 1-3 months: figsize=(36,16)
- `annotfont`: font size of annotation of peak bottom 
  - recommend: 4-6




##  10. <a name='ExampleResult'></a>Example Result
###  10.1. <a name='BatchofBackTest'></a>Batch of Back Test
- batch of back test using different period and buy, sell condition is conducted
- shown in [/example](example?ref_type=heads) folder 
- summary of result: [exmaple_result.md](example/example_result.md)

###  10.2. <a name='PlottingExtremafromDifferentSource'></a>Plotting Extrema from Different Source
result of plotting extrema from different price source: 
- [result_plot_extrema.pdf](result_plot_extrema.pdf?ref_type=heads)
- Example plot: [/example/tsla_2022-08-01_2023-08-25_bp_by_peak_bottom.png](example/tsla_2022-08-01_2023-08-25_bp_by_peak_bottom.png?ref_type=heads)
  - break point found by peak-bottom


###  10.3. <a name='Exampleplot'></a>Example plot

![Alt text](example/TSLA_2023-03-01_2023-07-01.png)

##  11. <a name='UnitTest'></a>Unit Test

###  11.1. <a name='Testscript'></a>Test script
the project contain a shell script that will test running `stock_analyser.py` and `backtest.py` on command line
- cd to root dir
```
./test_script.sh
```
####  11.1.1. <a name='ExpectedOutput'></a>Expected Output

if no errors occur, these output files are expected to be produced from the sctipt:
- PDD_2023-05-01_2023-08-20.png
- PLTR_2022-08-01_2023-08-01.png
- TSLA_2022-08-01_2023-08-01.png
- all_revenue.csv
- nvda_2022-08-01_2023-08-20.png
- pdd_2022-08-01_2023-08-20.png
- pdd_2023-08-01_2023-08-20.png
- roll_result_PDD_2022-08-01_2023-08-16.csv
- roll_result_nvda_2022-08-01_2023-08-20.csv
- roll_result_pdd_2022-08-01_2023-08-20.csv
- roll_result_pdd_2023-08-01_2023-08-20.csv
- stock_data_PDD_2022-08-01_2023-08-16.csv
- stock_data_PDD_2023-05-01_2023-08-20.csv
- stock_data_PLTR_2022-08-01_2023-08-01.csv
- stock_data_TSLA_2022-08-01_2023-08-01.csv
- stock_data_nvda_2022-08-01_2023-08-20.csv
- stock_data_pdd_2022-08-01_2023-08-20.csv
- stock_data_pdd_2023-08-01_2023-08-20.csv


###  11.2. <a name='Pytest'></a>Pytest

the project contain 2 test clients utilizing pytest for unit test

just run this in command line to launch pytest
```
cd tests
pytest
```
test clients:
- tests/test_backtest.py
- tests/test_stock_analyser.py

####  11.2.1. <a name='ExpectedOutput-1'></a>Expected Output
- 6 graphs of random stocks and 1 stock data csv file are expected to be produced


##  12. <a name='TechniquesStudied'></a>Techniques Studied
###  12.1. <a name='Stockpricesmoothingtechnique'></a>Stock price smoothing technique
- Moving Averages (MA, EMA, DMA)
- Butterworth Low Pass Filter
- polyfit
- linear convolution with np.blackman

detail discussion of pros and cons of different techniques see `technique_and_theory.md`


##  13. <a name='Bugtobesolved:'></a>Bug to be solved:
- ema (hence MACD) in early segment of stock data is not accurate, since ema is calculate base on yesturday's ema, so much earlier data before the specified start is required to get an accurate ema



##  14. <a name='FurtherStudy'></a>Further Study

1. find maximum draw down
2. maximum amount of fund actually put into market (currently calculated as maximum market value)
     - say I am watching 50 stocks with $10,000 allocated for each ($500,000 in total) 
     - I buy whenver buy signal appear for each stock
     - I want to know maximum amount of money actually held in the market at the same time over the period
