# Demo py file for how to use StockAnalyser and Backtest
import sys
import os
# adding root dir to sys.path
cwd=os.getcwd()
parent_wd = os.path.dirname(cwd)
abs_parent_wd  = os.path.abspath(parent_wd)
sys.path.insert(0, abs_parent_wd)


from app import stock_analyser as sa
from app import backtest as bt

# set up stock account, specify stock ticker, start, end, capital
# set up stock account, specify stock ticker, start, end, capital
pdd_ac = bt.StockAccount('PDD', '2023-06-01', '2023-08-01', 10000)
vrtx_ac = bt.StockAccount('VRTX', '2023-06-01', '2023-08-01', 10000)

# initialize backtest
back_test = bt.BackTest()

# set buy, sell strategy of backtest

back_test.set_buy_strategy(strategy=bt.BuyStrategy.FOLLOW_BUYPT_FILTER,
                           buypt_filters={sa.BuyptFilter.MA_SHORT_ABOVE_LONG}) 
                            # here IN_UPTREND is forced to true no matter pass it in or not, because converging drop and rising peak is aimed at finding strinking "bowl" in uptrend
back_test.set_sell_strategy(strategy=bt.SellStrategy.TRAIL_FIX_SL_AND_PROFTARGET,fixed_sl=0.03, profit_target=0.3)
                    # if no ts_percent passed, no trail stop loss will be set

# put account in backtest and call roll
# pdd_ac.txn = back_test.roll(pdd_ac,
#                              graph_showOption='save'
#                         )
# if no trend_col_name passed, source of trend will set to default, that is MACD signal

# get revenue of ac
revenue = pdd_ac.cal_revenue()

vrtx_ac.txn = back_test.roll(pdd_ac,
                             ma_short_list=[3, 20],
                             ma_long_list=[9, 50],     # here will find point ma3> ma9 and ma20>ma50
                             
                             graph_showOption='save',
                             csv_dir='../folder_name'
                        )
vrtx_ac.no_of_trade = back_test.trade_tmie_of_ac

vrtx_ac.cal_revenue()
vrtx_ac.cal_buy_and_hold()
vrtx_ac.print_txn()




# save roll result to csv
# pdd_ac.txn_to_csv(save_path='../folder_name', textbox='text to print on csv')
vrtx_ac.txn_to_csv(save_path='../folder_name', textbox='text to print on csv')