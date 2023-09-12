
from app.backtest import BackTest, BuyStrategy, runner, SellStrategy
from app.stock_analyser import BuyptFilter as bf
import pandas as pd
import numpy as np
import pytest
import random
from loguru import logger
import sys
import os
# adding root dir to sys.path
cwd=os.getcwd()
parent_wd = os.path.dirname(cwd)
abs_parent_wd  = os.path.abspath(parent_wd)
sys.path.insert(0, abs_parent_wd)

@pytest.fixture
def stock_list1():
    return ['amd', 'sofi', 'pdd', 'nio']

def set_logger():
    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='WARNING'

    )
def test_runner_one_ticker(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    revenue =runner(item, '2022-08-01', '2023-08-01',100000, SellStrategy.TRAILING_STOP,
           ts_percent=0.04, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           bp_filters={bf.CONVERGING_DROP, bf.IN_UPTREND, bf.RISING_PEAK}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return

def test_runner_one_ticker2(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    revenue =runner(item, '2022-08-01', '2023-08-01',100000, SellStrategy.HOLD_TIL_END,
           ts_percent=0.01, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           ma_short_list=[3, 9, 20], ma_long_list=[9, 30, 100],
           plot_ma=['ma10', 'ema20'], 
           bp_filters={bf.MA_SHORT_ABOVE_LONG, bf.IN_UPTREND}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return

def test_runner_list_of_stock_ticker3(stock_list1):
    set_logger()
    revenue =runner(stock_list1, '2023-05-01', '2023-08-01',100000, SellStrategy.PROFIT_TARGET,
           ts_percent=0.05, profit_target=0.1, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           trend_col_name='zigzag',
           bp_filters={bf.IN_UPTREND}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return

def test_runner_one_ticker4(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    revenue =runner(item, '2022-08-01', '2023-08-01',100000, SellStrategy.FIXED_STOP_LOSS,
           ts_percent=0.04, fixed_sl=0.03, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           bp_filters={bf.IN_UPTREND}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return

def test_runner_one_ticker5(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    revenue =runner(item, '2022-08-01', '2023-08-01',100000, SellStrategy.TRAILING_AND_FIXED_SL,
           ts_percent=0.04, fixed_sl=0.03, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           bp_filters={bf.IN_UPTREND}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return

def test_runner_one_ticker6(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    revenue =runner(item, '2022-08-01', '2023-08-01',100000, SellStrategy.TRAIL_FIX_SL_AND_PROFTARGET,
           ts_percent=0.04, fixed_sl=0.03, profit_target=0.1, buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           bp_filters={bf.IN_UPTREND}, graph_showOption="no")
    assert(isinstance(revenue, float))
    assert(revenue != 0)
    return