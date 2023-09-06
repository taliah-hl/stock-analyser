# Examples and Result

## Goal

50 top us stock in S&P 500 is selected to conduct back test in different period and conditions

$10,000 initial capital is allocated for each stock independently

## Stock in example

50 top us stock in S&P 500 are selected in [slickchart.com](https://www.slickcharts.com/sp500)


## Periods in the tests

1. 2021-11-01 to 2022-11-01 (bearish market)
2. 2020-06-01 2021-06-01 (bullish market)
3. 2008-03-01 to 2009-03-01 (financial crisis)
4. 2008-03-01 to 2010-03-01 (financial crisis to recovery)
5. 2023-01-01 to 2023-08-20 (YTD)
6. 2022-08-20 to 2023-08-20 (recent 1 year)
7. 2020-08-20 to 2023-08-20 (recent 3 year)
8. 2018-08-20 to 2023-08-20 (recent 5 year)

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
the following are shown
- gain / loss percentage
- maximum MV 
  - amount of market value in the day of largest market value (summing MV of all stocks) during the period

|buy-sell condition / period|1<br>2021-11-01 to 2022-11-01<br>(bearish)|2<br>2020-06-01 2021-06-01 <br>(bullish)|3<br>2008-03-01 to 2009-03-01 <br>(financial crisis)| 4<br>2008-03-01 to 2010-03-01 <br>(financial crisis to recovery)|5 <br> 2023-01-01 to 2023-08-20 <br>(YTD)|6 <br> 2022-08-20 to 2023-08-20<br> (recent 1 year)|7<br>2020-08-20 to 2023-08-20<br>(recent 3 year)|8<br> 2018-08-20 to 2023-08-20<br>(recent 5 year)|
| -----| -----| -----| -----| -----| -----|-----| -----| -----|
| [buy 1] + [sell 1] <br>*(maximum MV)*|+10.14% |+30.34% |-12.28% |+36.36% | +19.85%<br> *$549357* | +26.75%| +84.27% <br> *$849,476*| +357.48% <br>*$2,269,766*|
|  [buy 1] + [sell 2]<br>*(maximum MV)*|+9.92% | +33.17%|-12.23% |+35.68% | +20.64%<br>*$551,691* |+26.44%| +89.22%  <br>*$870,907*| +404.89% <br>*$2,443,801*|
|  [buy 2] + [sell 1]<br>*(maximum MV)* | +4.57%| +11.43%| -14.38%| -4.62%| +7.49%<br>  *$440982*| +12.32%| +35.19% *$542,199* |+83.17%<br> *$657,712* |
|  [buy 2] + [sell 2]<br>*(maximum MV)* |+4.17% |+15.24% |-13.97% |-2.58% |+7.56% <br> *$449,819* |+13.28% |+37.50% <br>*$565,622* |+108.05% <br>*$755,140*|
|  [buy 3] + [sell 1] <br>*(maximum MV)*|+4.39% |+31.27% | -14.91%|+3.64 | +12.61%<br> *$383,697* |+12.22%| +52.02% <br>*$531,189*| +175.08% <br>*$1,077,924*|
|  [buy 3] + [sell 2]<br>*(maximum MV)*|+3.82% |+32.12% |-14.10% |+4.55% | +12.79% <br> *$388997* |+12.16%|+51.76% <br>*$545,116* | +177.45% <br> *$1075594*|
|  buy and hold (control group)<br>*(maximum MV)*|-11.15% |+40.51% |-33.98% | +1.74%|  +22.05% |+13.64%| +47.36%| +112.73%|


## Config files used in tests
see the .json files
- e.g. `top50_SP500_period3_MACD_condition12.json` means the configs for period 3 by buy condition 1 + sell condition 2
  
## csv data

- see the .csv files (file name align with corresponding config file)
