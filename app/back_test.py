## import modules
from loguru import logger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import math
from tabulate import tabulate
import time
import argparse
import sys
import enum

## custom import

import stock_analyser as sa

class Action(enum.Enum):
    BUY=1
    SELL=2
    NAN=0

class BuyStrategy(enum.Enum):
    Default=0
    Uptrend_converging_bottom =1

class SellStrategy(enum.Enum):
    Default=0
    Stop_loss =1
    Buy_and_hold=2

class StockAccount():
    def __init__(self, ticker: str, start: str, end: str,initial_capital: int):
        self.ticker=ticker
        self.start=start
        self.end=end
        self.initial_capital=initial_capital
        self.stock_analyser = None
        self.txn: pd.DataFrame=None

    def roll(self, res):
        """
        TO BE IMPLEMENT
        """
        self.txn = res
        return self.txn
    
    def print_txn(self):
        logger.debug(tabulate(self.txn, headers='keys', tablefmt='psql', floatfmt=".2f"))




class BackTest():

    def __init__(self):
        pass
        self.buy_strategy: BuyStrategy=BuyStrategy.Uptrend_converging_bottom
        self.sell_strategy: SellStrategy=SellStrategy.Stop_loss
        self.sell_signal: str=''
        self.sell_signal: str=''
        self.stock_table=None
        self.stop_loss_percent: float=0.05
        #self.stock: str=''
        #self.account: StockAccount
        self.position_size: float=1     # portion of capital to use in each buy action
        self.start: str=''
        self.end: str=''
        self.buyDates: list     # list of str (date)
        self.sellDates: list     # list of str (date)
        self.actionDates: dict  # dict of list of str (date)


    def set_buy_strategy(self, strategy):
        self.buy_strategy = strategy


    def set_sell_strategy(self, strategy, percentage: float=0):
        self.sell_strategy = strategy
        if strategy==SellStrategy.Stop_loss:
            self.stop_loss_percent = percentage

    def set_buy_signal(self):
        pass

    def set_sell_signal(self):
        pass

    def set_stock(self, ticker: str, start: str, end: str, 
                  peak_btm_src: str='close', T:int=0,
                  bp_trend_src: str='signal',
                   extra_text_box:str='',
                    graph_showOption: str='no', 
                    graph_dir: str='../../stock.png', 
                    figsize: tuple=(36,24), 
                    annotfont: float=6
                  )->sa.StockAnalyser:
        """
            init stock using StockAnalyser.default_analyser
        """
        stock = sa.StockAnalyser()

        if self.buy_strategy == BuyStrategy.Uptrend_converging_bottom:

            self._stock_table = stock.default_analyser(
                tickers=ticker, start=start, end=end,
                method=peak_btm_src, T=T,
                bp_trend_src=bp_trend_src,
                bp_filter_conv_drop=True, bp_filter_rising_peak=True,
                extra_text_box=extra_text_box,
                graph_showOption=graph_showOption,
                graph_dir=graph_dir, 
                figsize=figsize, annotfont=annotfont
            )
            return stock
        
    def sell(self, prev_row: dict, cur_row: dict, portion: float=1)->dict:
        """
        - portion: portion of holding to sell

        return 

        updated row after sell
        """
        
        cur_row['action']=Action.SELL
        cur_row['deal price']=cur_row['close price']
        sell_share = math.ceil(prev_row['share']  * portion)
        cur_row['txn amt']=cur_row['deal price'] * sell_share
        cur_row['share'] = prev_row['share'] - sell_share
        cur_row['MV'] = cur_row['close price'] * cur_row['share']
        cur_row['cash'] = prev_row['cash'] + cur_row['txn amt']
        return cur_row
    
    def buy(self, prev_row, cur_row: dict, share: int)->dict:
        """
            input: prev row
            return 

            updated row after buy
        """
        cur_row['action']=Action.BUY
        cur_row['deal price']=cur_row['close price']
        cur_row['txn amt']=cur_row['deal price'] * share
        cur_row['share'] = prev_row['share'] + share
        cur_row['MV'] = cur_row['close price'] * cur_row['share']
        cur_row['cash'] = prev_row['cash'] - cur_row['txn amt']
        return cur_row

        

        
    def check_sell(self, strategy, prev_row: dict, cur_row: dict, latest_high: float, cur_price: float):
        """
            input: prev row
            return: cur row, bool: True=sold, False=not sold
        """
        
        if strategy== SellStrategy.Stop_loss:
            if cur_price < (latest_high * (1-self.stop_loss_percent)):
                # sell
                cur_row = self.sell(prev_row, cur_row)
                return (cur_row, True)
            
        cur_row['cash']=prev_row['cash']
        cur_row['share'] = prev_row['share']
        cur_row['action'] = np.nan
        cur_row['deal price'] = np.nan
        cur_row['txn amt'] = np.nan
        return (cur_row, False)



    def roll(self, acc: StockAccount)->pd.DataFrame:

        stock = self.set_stock(ticker=acc.ticker, start=acc.start, end=acc.end)

        
        # self.stock_table set after set fn
        txn_table=pd.DataFrame(index=self._stock_table.index, 
                               columns=['cash', 'share','close price','MV', 'action','deal price', 'txn amt'])
        
        cash_col=txn_table.columns.get_loc('cash')
        share_col=txn_table.columns.get_loc('share')
        close_col=txn_table.columns.get_loc('close price')
        mv_col=txn_table.columns.get_loc('MV')
        action_col=txn_table.columns.get_loc('action')
        dp_col=txn_table.columns.get_loc('deal price')
        amt_col = txn_table.columns.get_loc('txn amt')
        
        txn_table['cash'] = acc.initial_capital
        txn_table['share']=0 
        txn_table['close price']=self._stock_table['close']
        txn_table['MV']=0
        txn_table['action']=np.nan
        txn_table['deal price']=np.nan
        txn_table['txn amt']=np.nan

        print("len txn table")
        print(len(txn_table))
        
        
        is_holding=False   # is holding a stock currently or not
        latest_high=float('-inf')

        next_buypt=0
        if len(stock.buypt_dates) >0:
            idx=stock.buypt_dates[next_buypt]    #set initial idx to first buy point
            next_buypt +=1
        else:
            return txn_table
        
        
        while idx < len(self._stock_table):

           ## 1. Check sell
                
            if is_holding:
                latest_high = max(latest_high, txn_table['close price'][idx])
                (txn_table.iloc[idx], is_sold) = self.check_sell(strategy=self.sell_strategy, prev_row=txn_table.iloc[idx-1].to_dict(), cur_row=txn_table.iloc[idx].to_dict(),
                                        latest_high=latest_high, cur_price=txn_table['close price'][idx])
                is_holding = not is_sold
            
             ## 2. Check buy
            if (self._stock_table['day of interest'][idx]==sa.DayType.BUYPT) and (not is_holding):

                # buy with that day close price

                share_num=math.floor(txn_table['cash'][idx-1] / txn_table['close price'][idx])
                txn_table.iloc[idx] = self.buy(prev_row=txn_table.iloc[idx-1].to_dict(), 
                                        cur_row=txn_table.iloc[idx].to_dict(),
                                        share=share_num )
                is_holding = True
                latest_high=float('-inf') # reset latest high
                next_buypt +=1
                
            ## 3. if no action -> update row
            if txn_table['action'][idx] ==0: # no action this day
                txn_table.iloc[idx, cash_col]=txn_table['cash'][idx-1]
                txn_table.iloc[idx, share_col]=txn_table['share'][idx-1]
          
                # fast forward if !is_holding and no more buypoint onward
                if (idx >= stock.buypt_dates[-1]) and (not is_holding) and idx+1 >len(txn_table):
                    txn_table.iloc[idx+1:, cash_col] = txn_table.iloc[idx, cash_col]
                    txn_table.iloc[idx+1:, share_col] = txn_table.iloc[idx, share_col]
                    break
                if not is_holding:

                    if next_buypt < len(stock.buypt_dates) and idx+1 >len(txn_table):
                        txn_table.iloc[idx+1: stock.buypt_dates[next_buypt],   cash_col] = txn_table.iloc[idx,  cash_col]
                        txn_table.iloc[idx+1: stock.buypt_dates[next_buypt],   share_col] = txn_table.iloc[idx,  share_col]
                        idx = stock.buypt_dates[next_buypt] # jump to next buy point
                    

            if idx == (len(txn_table)-1) and is_holding:   # last row
                #  force sell
                txn_table.iloc[idx]  =self.sell(prev_row=txn_table.iloc[idx-1].to_dict(),
                                               cur_row=txn_table.iloc[idx].to_dict())
                break
            idx+=1

        return txn_table
        
def runner():
    ac = StockAccount('pdd', '2022-07-01', '2023-08-10', 10000)
    back_test = BackTest()
    back_test.set_buy_strategy(BuyStrategy.Uptrend_converging_bottom)
    back_test.set_sell_strategy(SellStrategy.Stop_loss, 0.05)
    ac.txn = back_test.roll(ac)
    ac.print_txn()


if __name__ == "__main__":
    runner()




