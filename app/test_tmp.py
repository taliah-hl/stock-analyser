from stock_analyser import StockAnalyser
import pandas as pd
import numpy as np
import random
from loguru import logger
import sys



def stock_list1():
    return ['amd', 'sofi', 'pdd', 'nio']


stock = StockAnalyser()
result=stock.default_analyser(tickers='nvda', start='2022-09-01', end='2022-11-20',
                        method='close', 
                        window_size=5,
                        bp_trend_src='signal',
                            graph_showOption='save', graph_dir='../../test2_result'   )
print(stock.buypt_dates)
