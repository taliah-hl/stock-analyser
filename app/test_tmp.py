from stock_analyser import StockAnalyser
import pandas as pd
import numpy as np
import random
from loguru import logger
import sys



def stock_list1():
    return ['amd', 'sofi', 'pdd', 'nio']
logger.remove()     # remove deafult logger before adding custom logger
logger.add(
        sys.stderr,
        level='INFO'

)

stock = StockAnalyser()
result=stock.default_analyser(tickers='xlf', start='2009-01-01', end='2016-11-20',
                        method='ema', T=9, 
                        window_size=5,
                        bp_trend_src='signal',
                            graph_showOption='save', figsize=(100,60), graph_dir='../../xlf',annotfont=3 )
print(stock.buypt_dates)
