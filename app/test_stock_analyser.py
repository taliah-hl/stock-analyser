from stock_analyser import StockAnalyser
from stock_analyser import BuyptFilter
import pandas as pd
import numpy as np
import pytest
import random
from loguru import logger
import sys


@pytest.fixture
def stock_list1():
    return ['amd', 'sofi', 'pdd', 'nio']


def set_logger():
    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='WARNING'

    )


def test_runner_close(stock_list1):
    """
        run a short list of stock, DL data from 2023-01-01 to 2023-08-01
        get extrema from close price
        cal bp from zigzag
    """
    set_logger()
    
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    print(f"testing stock: {item}")
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2023-01-01', end='2023-08-01',
                           method='close', 
                           trend_col_name='zigzag',
                           graph_showOption='save', graph_dir='../../unit_test_result')

    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)
    #assert result['day of interest'].dtype==DayType

    return

def test_runner_ema9(stock_list1):
    set_logger()

    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    print(f"testing stock: {item}")
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2023-01-01', end='2023-08-01',
                            method='ema', T=9,
                            window_size=5,
                               graph_showOption='save', graph_dir='../../unit_test_result'   )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)
    #assert result['day of interest'].dtype==DayType

    return

def test_runner_ma5(stock_list1):
    set_logger()

    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    print(f"testing stock: {item}")
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2023-01-01', end='2023-08-01',
                            method='ma', T=5,
                            window_size=5,
                               graph_showOption='save', graph_dir='../../unit_test_result'   )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)
    #assert result['day of interest'].dtype==DayType

    return

def test_runner_close_macd(stock_list1):
    set_logger()

    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    print(f"testing stock: {item}")
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2023-05-01', end='2023-08-01',
                            method='close',
                            window_size=5,
                               graph_showOption='save', graph_dir='../../unit_test_result'   )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)
    #assert result['day of interest'].dtype==DayType

    return

def test_many_param(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2023-05-01', end='2023-08-01',
                            method='ema', T=1, 
                            window_size=5,smooth_ext=9,  smooth=True, zzupthres=0.08,
                            zzdownthres=0.08, trend_col_name='zigzag', 
                            bp_filters={BuyptFilter.IN_UPTREND, BuyptFilter.CONVERGING_DROP, BuyptFilter.MA_SHORT_ABOVE_LONG, BuyptFilter.RISING_PEAK},
                            ma_short_list=[3, 20, 100], ma_long_list=[9, 50, 200], plot_ma=['ma3', 'ema20'],
                            print_stock_data=True, csv_dir='../../unit_test_result',
                               graph_showOption='save', graph_dir='../../unit_test_result/'   )
    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)

def test_runner_3yr(stock_list1):
    set_logger()
    
    

    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    print(f"testing stock: {item}")
    stock = StockAnalyser()
    result=stock.default_analyser(tickers=item, start='2020-08-01', end='2023-08-01',
                            method='close',
                               graph_showOption='save', graph_dir='../../unit_test_result/'   )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >1
    assert 'close' in result
    assert 'type' in result
    assert 'zigzag' in result
    assert 'buy pt' in result
    assert 'day of interest' in result
    assert result['close'].dtypes==np.float64 or result['close'].dtypes==np.int64
    assert result['type'].dtypes==np.float64
    assert result['zigzag'].dtypes==np.int64
    assert isinstance(stock.extrema, pd.DataFrame)
    #assert result['day of interest'].dtype==DayType

    return


# def test_wrong_fn():
#     stock=StockAnalyser()
#     stock.wrong_fn()

def test_get_fns(stock_list1):
    set_logger()
    item=stock_list1[random.randint(0, len(stock_list1)-1)]
    stock=StockAnalyser()
    stock.download(item, start='2023-05-01', end='2023-08-01')
    res = stock.get_close_price()
    assert isinstance(res, pd.DataFrame)
    assert len(res)>1
    stock.get_stock_data()
    assert isinstance(res, pd.DataFrame)
    assert len(res)>1
    stock.print_stock_data()
    stock.add_column_ma(src_data=stock.stock_data['close'], mode='ema', period=5)
    stock.set_extrema(src_data=stock.stock_data['ema5'], close_price=stock.stock_data['close'])
    res = stock.get_extrema()
    assert isinstance(res, pd.DataFrame)
    assert len(res)>1
    res = stock.get_col('type')
    assert isinstance(res, pd.Series)
    assert len(res)>1




        


