# Examples and Result

## Goal

50 top us stock in S&P 500 is selected to conduct back test in different period and conditions

## Stock in example

50 top us stock in S&P 500 are selected in [slickchart.com](https://www.slickcharts.com/sp500)


## Periods in the tests

1. 2021-11-01 to 2022-11-01 (bearish market)
2. 2020-06-01 2021-06-01 (bullish market)
3. 2008-03-01 to 2009-03-01 (financial crisis)
4. 2008-03-01 to 2010-03-01 (financial crisis to recovery)
5. 2022-08-20 to 2023-08-20 (recent 1 year)

## Buy Point Filter

### Set 1
- "IN_UPTREND"
- in slope of MACD Signal >0

### Set 2
- "CONVERGING_DROP", "RISING_PEAK"
- peak-bottom
- buy at break point of converging bottom in up trend

### Set 3
- "MA_SHORT_ABOVE_LONG"
- buy points at ma3 > 13 and ma 50> ma150

## Sell Strategy

### Set 1
- trailing stop-loss 5% + fixed stop-loss 3%

### Set 2
- trailing stop-loss 5%


## Result

|buy/sell condition|1<br>2021-11-01 to 2022-11-01<br>(bearish)|2<br>2020-06-01 2021-06-01 <br>(bullish)|3<br>2008-03-01 to 2009-03-01 <br>(financial crisis)| 4<br>2008-03-01 to 2010-03-01 <br>(financial crisis to recovery)|5<br> 2022-08-20 to 2023-08-20<br> (recent 1 year)|
| -----| -----| -----| -----| -----| -----|
| [buy 1] + [sell 1]|+10.14% |+30.34% |-12.28% |+36.36% | +26.75%|
|  [buy 1] + [sell 2]|+9.92% | +33.17%|-12.23% |+35.68% | +26.44%|
|  [buy 2] + [sell 1] | +4.57%| +11.43%| -14.38%| -4.62%| +12.32%|
|  [buy 2] + [sell 2] |+4.17% |+15.24% |-13.97% |-2.58% |13.28% |
|  [buy 3] + [sell 1] |+4.39% |+31.27% | -14.91%|+3.64 | +12.22%|
|  [buy 3] + [sell 2]|+3.82% |+32.12% |-14.10% |+4.55% |+12.16%|
|  buy and hold (control group)|-11.15% |+40.51% |-33.98% | +1.74%|+13.64%|


## Config files used in tests
see the .json files
- e.g. `top50_SP500_period3_MACD_condition12.json` means the configs for period 3 by buy condition 1 + sell condition 2
  
## csv data

- see the .csv files (file name align with corresponding config file)