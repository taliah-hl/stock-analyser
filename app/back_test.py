## import modules
from loguru import logger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import math
from tabulate import tabulate
import time
from datetime import date
import argparse
import sys
import enum
import os

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
        self._revenue: float=None

    def roll(self, res):
        """
        TO BE IMPLEMENT
        """
        self.txn = res
        return self.txn
    
    def print_txn(self):
        logger.debug(f"----  Transaction Table of {self.ticker}  ----")
        logger.debug(tabulate(self.txn, headers='keys', tablefmt='psql', floatfmt=".2f"))

    def txn_to_csv(self, save_path:str=None):
        if save_path is None:
            save_path = f"../../back_test_result/{self.ticker}_{self.start}_{self.end}.csv"
        self.txn.to_csv(save_path)

        logger.info(f"csv saved to {save_path}")

    def cal_revenue(self):
        if self.txn is not None:
            return (self.txn['cash'][-1] - self.initial_capital)/self.initial_capital
        else:
            logger.warning("no transaction, cannot get revenue, you may want to roll the ACC first.")
            return None
        
    def set_revenue(self):
        if self.txn is not None:
            self._revenue = (self.txn['cash'][-1] - self.initial_capital)/self.initial_capital

    def get_revenue(self):
        if self._revenue is not None:
            return self._revenue
        elif (self._revenue is None) and (self.txn is not None):
            self.set_revenue()
            return self._revenue
        else:
            logger.warning("no transaction, cannot get revenue, you may want to roll the ACC first.")
            return None

    def cal_final_capial(self):
        if self.txn is not None:
            return self.txn['cash'][-1]
        else:
            logger.warning("no transaction, cannot get final capital, you may want to roll the ACC first.")
            return None


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
        
    def sell(self, prev_row: dict, cur_row: dict, trigger_price: float, portion: float=1, )->dict:
        """
        - portion: portion of holding to sell

        return 

        updated row after sell
        """
        
        cur_row['action']=Action.SELL
        cur_row['deal price']=trigger_price
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
                cur_row = self.sell(prev_row, cur_row, trigger_price=latest_high * (1-self.stop_loss_percent))
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
                               columns=['cash', 'share','close price','MV', 'action','deal price', 'txn amt', 'total asset', 'latest high','+-'])
        
        cash_col=txn_table.columns.get_loc('cash')
        share_col=txn_table.columns.get_loc('share')
        close_col=txn_table.columns.get_loc('close price')
        mv_col=txn_table.columns.get_loc('MV')
        action_col=txn_table.columns.get_loc('action')
        dp_col=txn_table.columns.get_loc('deal price')
        amt_col = txn_table.columns.get_loc('txn amt')
        ass_col = txn_table.columns.get_loc('total asset')
        lh_col = txn_table.columns.get_loc('latest high')
        change_col = txn_table.columns.get_loc('+-')
        
        txn_table['cash'] = acc.initial_capital
        txn_table['total asset'] = acc.initial_capital
        txn_table['share']=0 
        txn_table['close price']=self._stock_table['close']
        txn_table['MV']=0
        txn_table['action']=np.nan
        txn_table['deal price']=np.nan
        txn_table['txn amt']=np.nan
        txn_table['latest high']=np.nan
        txn_table['+-'] = np.nan

        #print("len txn table")
        #print(len(txn_table))
        
        
        is_holding=False   # is holding a stock currently or not
        latest_high=float('-inf')
        last_buy_date=0

        next_buypt=0
        if len(stock.buypt_dates) >0:
            idx=stock.buypt_dates[next_buypt]    #set initial idx to first buy point
        else:
            return txn_table
        
        #print(stock.buypt_dates)
        #print(stock.buypt_dates[-1])
        
        while idx < len(self._stock_table):

           # logger.debug(f"today: {txn_table.index[idx]}, is holding {is_holding}")


           ## 1. Check sell
                
            if is_holding:
                latest_high = max(latest_high, txn_table['close price'][idx])
                txn_table.iloc[idx, lh_col] = latest_high
                (txn_table.iloc[idx], is_sold) = self.check_sell(strategy=self.sell_strategy, prev_row=txn_table.iloc[idx-1].to_dict(), cur_row=txn_table.iloc[idx].to_dict(),
                                        latest_high=latest_high, cur_price=txn_table['close price'][idx])
                is_holding = not is_sold
                if is_sold:
                    txn_table.iloc[idx, change_col] = (txn_table['cash'][idx] - (txn_table['cash'][last_buy_date] + txn_table['txn amt'][last_buy_date]) ) / (txn_table['cash'][last_buy_date] + txn_table['txn amt'][last_buy_date])

                if not is_holding:
                    #print("no holding today")
                    #print(f"next_buypt: {next_buypt}, ")
                    for i in range(next_buypt, len(stock.buypt_dates)):
                        if stock.buypt_dates[next_buypt] > idx:
                            break
                        next_buypt+=1   # find next buy point date

                    if (next_buypt < len(stock.buypt_dates)) and ( idx+1 <len(txn_table)):
                        #print("fast forward")
                        txn_table.iloc[idx+1: stock.buypt_dates[next_buypt],   [cash_col, share_col]] = txn_table.iloc[idx,   [cash_col, share_col]]
                        idx = stock.buypt_dates[next_buypt] # jump to next buy point
                        # print("next idx: ", idx)
                        continue
            
             ## 2. Check buy
            if ((self._stock_table['day of interest'][idx]==sa.DayType.BUYPT) and (not is_holding)):
                #logger.debug("check buy triggered")
                # buy with that day close price

                share_num=math.floor(txn_table['cash'][idx-1] / txn_table['close price'][idx])
                txn_table.iloc[idx] = self.buy(prev_row=txn_table.iloc[idx-1].to_dict(), 
                                        cur_row=txn_table.iloc[idx].to_dict(),
                                        share=share_num )
                last_buy_date=idx
                is_holding = True
                latest_high=float('-inf') # reset latest high
                
                
            
            
            ## 3. if no action -> update row
            if pd.isna( txn_table.iloc[idx, action_col]) : # no action this day
                txn_table.iloc[idx, cash_col]=txn_table['cash'][idx-1]
                txn_table.iloc[idx, share_col]=txn_table['share'][idx-1]
                #print(f"updated: {txn_table['cash'][idx]}, {txn_table['cash'][idx]}")
          
                # fast forward if !is_holding and no more buypoint onward
                if (idx >= stock.buypt_dates[-1]) and (not is_holding) and idx+1 <len(txn_table):
                    txn_table.iloc[idx+1:, cash_col] = txn_table.iloc[idx, cash_col]
                    txn_table.iloc[idx+1:, share_col] = txn_table.iloc[idx, share_col]
                    break
                
                    

            if idx == (len(txn_table)-1) and is_holding:   # last row
                #  force sell
                txn_table.iloc[idx]  =self.sell(prev_row=txn_table.iloc[idx-1].to_dict(),
                                               cur_row=txn_table.iloc[idx].to_dict(),
                                               trigger_price=txn_table['close price'][idx])
                break
            #logger.debug(f"today action: {txn_table.iloc[idx, action_col]}")
            idx+=1

        for i in range(0, len(txn_table)):
            txn_table.iloc[i, mv_col] = txn_table['close price'][i] * txn_table['share'][i]
            txn_table.iloc[i, ass_col] = txn_table['cash'][i] + txn_table['MV'][i]

        return txn_table
        
    def print_revenue(self, ac_list:list, total_revenue):
        """
        input: list of acc
        all accs need to be roll first
        time range of all acc need to be same
        print revenue to csv file

        """
        
        table=[]
        

        for ac in ac_list:
            table.append({'stock': ac.ticker, 
                          'revenue': ac.get_revenue()})
        table.append({'stock': 'overall', 'revenue': total_revenue })

        revenue_table=pd.DataFrame( table, columns=['start', 'end', 
                                            'stock','buy strategy', 'sell strategy', 'revenue'])
        
        revenue_table['start']  = ac_list[0].start
        revenue_table['end']  = ac_list[0].end
        revenue_table['buy strategy']  = self.buy_strategy
        revenue_table['sell strategy']  = self.sell_strategy

        revenue_table.to_csv(f'../../back_test_result/all_revenue_.csv')




def runner(tickers, start:str, end:str, capital:float):

    if isinstance(tickers, str):
        ac = StockAccount(tickers, start, end, capital)
        logger.info(f'---- **** Back Test of {tickers} stated **** ---')
        back_test = BackTest()
        back_test.set_buy_strategy(BuyStrategy.Uptrend_converging_bottom)
        back_test.set_sell_strategy(SellStrategy.Stop_loss, 0.05)
        ac.txn = back_test.roll(ac)
        #ac.print_txn()
        ac.txn_to_csv()
        rev=ac.cal_revenue()
        logger.debug(f"revenue of {tickers}: {rev}")
        logger.info(f" Back Test of {tickers} done")
    
    
    elif isinstance(tickers, list):

        back_test = BackTest()
        back_test.set_buy_strategy(BuyStrategy.Uptrend_converging_bottom)
        back_test.set_sell_strategy(SellStrategy.Stop_loss, 0.05)

        acc_list=[]
        total_finl_cap=0
        for item in tickers:
            ac = StockAccount(item, start, end, capital)
            logger.info(f'---- **** Back Test of {item} started **** ---')
            
            ac.txn = back_test.roll(ac)
            ac.print_txn()
            ac.txn_to_csv()
            rev=ac.cal_revenue()
            total_finl_cap += ac.cal_final_capial()
            logger.info(f"revenue of {item}: {rev}")
            logger.info(f" Back Test of {item} done")
            acc_list.append(ac)

        final_rev = ( total_finl_cap - capital * len(tickers))/(capital * len(tickers))
        logger.info(f"total revenue of run: {final_rev}")
        back_test.print_revenue(acc_list, final_rev)



if __name__ == "__main__":
    start = time.time()
    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='INFO'

    )
    logger.add(
        f"../../BackTest_{date.today()}_log.log",
        level='INFO'

    )
    logger.info("-- ****  NEW RUN START **** --")


    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    
    short_list=['nvda', 'nio', 'pdd']

    stock_lst_file='../../hotstock25.txt'
    with open(stock_lst_file, 'r') as fio:
            lines = fio.readlines()

    for item in lines:
            item=item.strip()
    runner(lines, '2021-11-01', '2022-11-01', 50000)
    end=time.time()
    logger.info(f"time taken for whole run: {end-start}")
        




