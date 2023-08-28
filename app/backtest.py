## import modules
from loguru import logger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import math
from tabulate import tabulate
import time
from datetime import date, timedelta, datetime
import argparse, json
import sys
import enum
import os
import re

## custom import

import stock_analyser as sa


class Action(enum.Enum):
    BUY=1
    SELL=2
    NAN=0

class BuyStrategy(enum.Enum):
    DEFAULT=0
    FOLLOW_BUYPT_FILTER =1
    SOME_UNKNOWN_STRATEGY = 99

class SellStrategy(enum.Enum):
    DEFAULT=0   #default is trailing stop, change in set_sell_strategy() method of BackTest
    TRAILING_STOP  =1
    HOLD_TIL_END=2
    PROFIT_TARGET=3
    FIXED_STOP_LOSS=4
    TRAILING_AND_FIXED_SL  =5
    TRAIL_FIX_SL_AND_PROFTARGET=6
    MIX=10  # did not set yet

class StockAccount():
    def __init__(self, ticker: str, start: str, end: str,initial_capital: int):
        self.ticker=ticker
        self.start=start
        self.end=end
        self.initial_capital=initial_capital
        self.stock_analyser = None
        self.txn: pd.DataFrame=None
        self.revenue: float=None
        self.revenue_buy_and_hold: float=None
        self.no_of_trade: int=0

    
    def print_txn(self):
        logger.debug(f"----  Transaction Table of {self.ticker}  ----")
        logger.debug(tabulate(self.txn, headers='keys', tablefmt='psql', floatfmt=".2f"))

    def txn_to_csv(self, save_path:str=None, textbox: str=None):
        if save_path is None:
            save_path='../../back_test_result/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        add_num =1
        save_path += f"/roll_result_{self.ticker}_{self.start}_{self.end}"
        save_path_norepeat = save_path
        while os.path.isfile(f'{save_path_norepeat}.csv'):
            save_path_norepeat = save_path + f'_{add_num}'
            add_num +=1

        save_path_norepeat += '.csv'

        self.cal_revenue()
        self.txn.to_csv(save_path_norepeat)
        with open(save_path_norepeat, 'a') as fio:
            fio.write(textbox)
            fio.write('\n')
            fio.write('revenue: ,'+'{:.2%}'.format(self.revenue) + ', {:.2f}\n'.format(self.initial_capital * (self.revenue) ))
            fio.write('revenue if buy and hold: ,')
            fio.write('{:.2%}'.format(self.revenue_buy_and_hold) +', {:.2f}\n'.format(self.initial_capital * (self.revenue_buy_and_hold) ))
            fio.write('no. of trade: ,' + str(self.no_of_trade))
            

        logger.info(f"csv saved to {save_path_norepeat}")

        
    def cal_revenue(self):
        if self.txn is not None:
            self.revenue = (self.txn['cash'][-1] - self.initial_capital)/self.initial_capital
            return self.revenue
        else:
            logger.warning("no transaction, cannot get revenue, you may want to roll the ACC first.")
            return None

    def get_revenue(self):
        if self.revenue is not None:
            return self.revenue
        elif (self.revenue is None) and (self.txn is not None):
            self.set_revenue()
            return self.revenue
        else:
            logger.warning("no transaction, cannot get revenue, you may want to roll the ACC first.")
            return None

    def cal_final_capital(self):
        if self.txn is not None:
            return self.txn['cash'][-1]
        else:
            logger.warning("no transaction, cannot get final capital, you may want to roll the ACC first.")
            return None
    
    def cal_buy_and_hold(self)->float:
        """
        calculate revenue of buying at first day and selling at last day as control group
        set function of self.revenue_buy_and_hold
        """
        if self.txn is not None:
            buy_share = math.floor(self.initial_capital/ self.txn['close price'][0])
            money_left = self.initial_capital- buy_share * self.txn['close price'][0]
            sold = buy_share * self.txn['close price'][-1]
            money_left = money_left+sold
            self.revenue_buy_and_hold = (money_left - self.initial_capital) /self.initial_capital
            return self.revenue_buy_and_hold
        else:
            logger.warning("no transaction, cannot calculate buy and hold, you may want to roll the ACC first.")
            return



class BackTest():

    def __init__(self):
        pass
        self.buy_strategy: BuyStrategy=BuyStrategy.FOLLOW_BUYPT_FILTER
        self.sell_strategy: SellStrategy=SellStrategy.TRAILING_STOP 
        self.sell_signal: str=''
        self.sell_signal: str=''
        self.stock_table=None
        self.trail_loss_percent: float=None
        self.fixed_st: float=None
        self.profit_target: float=None
        #self.stock: str=''
        #self.account: StockAccount
        self.position_size: float=1     # portion of capital to use in each buy action
        self.start: str=''
        self.end: str=''
        self.buyDates: list     # list of str (date)
        self.sellDates: list     # list of str (date)
        self.actionDates: dict  # dict of list of str (date)
        self.profit_target=float('inf')
        self.bp_filters=set()
        self.trade_tmie_of_ac: int=0


    def set_buy_strategy(self, strategy=None, buypt_filters: set=set()):
        if strategy is not None:
            self.buy_strategy = strategy

        self.bp_filters = buypt_filters


    def set_sell_strategy(self, strategy, ts_percent: float=None, 
                          profit_target: float=None, fixed_sl: float=None):
        self.sell_strategy = strategy
        if self.sell_strategy == SellStrategy.DEFAULT:
            self.sell_strategy = SellStrategy.TRAILING_STOP
        if not pd.isna(ts_percent):
            self.trail_loss_percent = ts_percent
        if not pd.isna(profit_target):
            self.profit_target=profit_target
        if not pd.isna(fixed_sl):
            self.fixed_st=fixed_sl


    def set_buy_signal(self):
        pass

    def set_sell_signal(self):
        pass

    def set_stock(self, ticker: str, start: str, end: str, 
                  peak_btm_src: str='close', T:int=0,
                  window_size=10, smooth_ext=10,
                  trend_col_name: str='slope signal',
                  bp_filters:set=set(),
                  ma_short_list: list=[], ma_long_list=[],
                  plot_ma: list=[],
                   extra_text_box:str='',
                    graph_showOption: str='save', 
                    graph_dir: str=None, 
                    figsize: tuple=(36,24), 
                    annotfont: float=4,
                    csv_dir: str=None, to_print_stock_data: bool=True 
                  )->sa.StockAnalyser:
        """
            init stock using StockAnalyser.default_analyser
        """
        stock = sa.StockAnalyser()
        if not bp_filters:
            bp_filters = self.bp_filters


        if self.buy_strategy == BuyStrategy.FOLLOW_BUYPT_FILTER:

            self._stock_table = stock.default_analyser(
                tickers=ticker, start=start, end=end,
                method=peak_btm_src, T=T,
                trend_col_name=trend_col_name,
                ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                plot_ma=plot_ma,
                bp_filters=bp_filters,
                extra_text_box=extra_text_box,
                graph_showOption=graph_showOption,
                graph_dir=graph_dir, 
                figsize=figsize, annotfont=annotfont,
                csv_dir=csv_dir, print_stock_data=to_print_stock_data 
            )
            return stock
        else:
            
            logger.warning(f"buy strategy receive: {self.buy_strategy}, which is not configurated, program exit.")
            exit(0)

    def sell(self, prev_row: dict, cur_row: dict, trigger_price: float, portion: float=1, last_buy_date=None, trigger: str=None)->dict:
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
        cur_row['trigger'] = trigger
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
        self.trade_tmie_of_ac +=1
        return cur_row

        

        
    def check_sell(self, strategy, prev_row: dict, cur_row: dict, latest_high: float, cur_price: float, last_buy_date=None):
        """
            input: prev row
            return: cur row, bool: True=sold, False=not sold
        """

        
        if self.sell_strategy== SellStrategy.TRAILING_STOP :
            if cur_price < (latest_high * (1-self.trail_loss_percent)):
                # sell
                cur_row = self.sell(prev_row, cur_row, trigger_price= math.floor(latest_high * (1-self.trail_loss_percent)*100)/100, trigger='trail stop')
                return (cur_row, True)
        elif (self.sell_strategy==SellStrategy.TRAIL_FIX_SL_AND_PROFTARGET) or (self.sell_strategy==SellStrategy.TRAILING_AND_FIXED_SL) :
            if (self.trail_loss_percent is not None 
                and cur_price < (latest_high * (1-self.trail_loss_percent)) ):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.floor(latest_high*(1-self.trail_loss_percent)*100)/100, trigger='trail stop')
                return (cur_row, True)
                
            elif (self.fixed_st is not None 
                  and cur_price <( (self._stock_table['close'][last_buy_date]) * (1-self.fixed_st))):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.floor(self._stock_table['close'][last_buy_date] * (1-self.fixed_st)*100)/100, trigger='fixed SL')
                return (cur_row, True)

            elif ( self.profit_target is not None
                  and cur_price >=  self._stock_table['close'][last_buy_date] * (1+self.profit_target)):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.ceil(self._stock_table['close'][last_buy_date] * (1+self.profit_target)*100)/100, trigger='profit target')
                return (cur_row, True)

            
            
        cur_row['cash']=prev_row['cash']
        cur_row['share'] = prev_row['share']
        cur_row['action'] = np.nan
        cur_row['deal price'] = np.nan
        cur_row['txn amt'] = np.nan
        return (cur_row, False)



    def roll(self, acc: StockAccount,
              method: str='', T: int=0, 
            window_size=10, smooth_ext=10, zzupthres: float=0.09, zzdownthres: float=0.09,
            trend_col_name: str='slope signal',
           bp_filters:set=set(),
           ma_short_list: list=[], ma_long_list=[],
           plot_ma: list=[],
           extra_text_box:str='',
           graph_showOption: str='save', graph_dir: str=None, figsize: tuple=(36,24), annotfont: float=6,
           csv_dir: str=None, to_print_stock_data: bool=True )->pd.DataFrame:

        try:
            if not bp_filters:
                bp_filters = self.bp_filters
            stock = self.set_stock(ticker=acc.ticker, start=acc.start, end=acc.end,
                                   trend_col_name=trend_col_name,
                                   bp_filters=bp_filters,
                                   ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                   plot_ma=plot_ma,
                                   extra_text_box=extra_text_box,
                                   graph_showOption=graph_showOption, graph_dir=graph_dir,
                                   figsize=figsize, annotfont=annotfont, 
                                   csv_dir=csv_dir, to_print_stock_data=to_print_stock_data
                                   )
        except Exception as err:
            logger.error(err)
            return None
        self.trade_tmie_of_ac = 0

        
        # self.stock_table set after set fn
        txn_table=pd.DataFrame(index=self._stock_table.index, 
                               columns=['cash', 'share','close price','MV', 'action',
                                        'deal price', 'txn amt', 'total asset', 'latest high','+-', 'trigger'])
        
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
        trig_col = txn_table.columns.get_loc('trigger')
        
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
        txn_table['trigger'] = np.nan

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
                                        latest_high=latest_high, cur_price=txn_table['close price'][idx], last_buy_date=last_buy_date)
                is_holding = not is_sold
                if is_sold:
                    txn_table.iloc[idx, change_col] = (txn_table['deal price'][idx] - txn_table['deal price'][last_buy_date] ) / txn_table['deal price'][last_buy_date] 

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
                    #print(f"sold, is holding={is_holding}, today: {txn_table.index[idx]}")
            
             ## 2. Check buy
            if ((self._stock_table['day of interest'][idx]==sa.DayType.BUYPT) and (not is_holding) and idx != len(txn_table)-1):
                #logger.debug("check buy triggered")
                # buy with that day close price
                #print(f"in checkbuy\n, is holding={is_holding}, today: {txn_table.index[idx]}, cash={txn_table['cash'][idx-1]}")
                try:
                    share_num=math.floor(txn_table['cash'][idx-1] / txn_table['close price'][idx])
                    if share_num >=1:
                        txn_table.iloc[idx] = self.buy(prev_row=txn_table.iloc[idx-1].to_dict(), 
                                                cur_row=txn_table.iloc[idx].to_dict(),
                                                share=share_num )
                        last_buy_date=idx
                        is_holding = True
                        latest_high=txn_table['close price'][idx] # reset latest high
                except OverflowError as err:
                    logger.error(err)
                    logger.error(f"at date={txn_table.index[idx]}, cash/close=infinity")

                
            
            
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
                                               trigger_price=txn_table['close price'][idx],
                                               trigger='last day')
                txn_table.iloc[idx, change_col] = (txn_table['deal price'][idx] - txn_table['deal price'][last_buy_date] ) / txn_table['deal price'][last_buy_date] 
                
                break
            #logger.debug(f"today action: {txn_table.iloc[idx, action_col]}")
            idx+=1

        for i in range(0, len(txn_table)):
            txn_table.iloc[i, mv_col] = txn_table['close price'][i] * txn_table['share'][i]
            txn_table.iloc[i, ass_col] = txn_table['cash'][i] + txn_table['MV'][i]

        return txn_table
        
    def print_revenue(self, ac_list:list, total_revenue, average_rev_buy_hold: float, save_path:str=None, textbox: str=None):
        """
        input: list of acc
        all accs need to be roll first
        time range of all acc need to be same
        print revenue to csv file

        """
        if save_path is None:
            save_path='../../back_test_result/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        add_num =1
        save_path += f"/all_revenue"

        save_path_norepeat = save_path
        while os.path.isfile(f'{save_path_norepeat}.csv'):
            save_path_norepeat = save_path + f'_{add_num}'
            add_num +=1

        save_path_norepeat += '.csv'
        
        table=[]


        for ac in ac_list:
            table.append({'stock': ac.ticker, 
                          'revenue': ac.get_revenue(),
                          "trade times": ac.no_of_trade,
                          'revenue if buy and hold': ac.revenue_buy_and_hold,
                          
                          })
        table.append({'stock': 'overall', 'revenue': total_revenue , "revenue if buy and hold": average_rev_buy_hold})

        revenue_table=pd.DataFrame( table, columns=['start', 'end', 
                                            'stock','buy point filter', 'sell strategy', 'revenue', 'trade times', 'revenue if buy and hold'])
        
        revenue_table['start']  = ac_list[0].start
        revenue_table['end']  = ac_list[0].end
        filter_str = ''
        for item in self.bp_filters:
            filter_str += (item.name+', ')
        revenue_table['buy point filter']  = filter_str
        revenue_table['sell strategy']  = self.sell_strategy

        revenue_table.to_csv(save_path_norepeat)
        with open(save_path_norepeat, 'a') as fio:
            fio.write(textbox)
        logger.info(f"overall revenue csv saved to {save_path_norepeat}")




def runner(tickers, start:str, end:str, capital:float, 
           sell_strategy, ts_percent: float=None, fixed_sl: float=None, profit_target:  float=None,
           buy_strategy=BuyStrategy.FOLLOW_BUYPT_FILTER,
           trend_col_name: str='slope signal',
           bp_filters: set=set(),
           ma_short_list: list=[], ma_long_list=[],
           plot_ma: list=[],
           graph_showOption: str='save', graph_dir: str=None, figsize: tuple=(36,24), annotfont: float=4,
           csv_dir:str='../../', print_all_ac:bool=True)->float:
    """
    return 

    overall revenue
    """

    if isinstance(tickers, str):
        ac = StockAccount(tickers, start, end, capital)
        logger.info(f'---- **** Back Test of {tickers} stated **** ---')
        back_test = BackTest()
        back_test.set_buy_strategy(buy_strategy, bp_filters)
        back_test.set_sell_strategy(strategy=sell_strategy, ts_percent=ts_percent, fixed_sl=fixed_sl, profit_target=profit_target)



        ac.txn = back_test.roll(ac,
                               trend_col_name=trend_col_name,
                                ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                plot_ma=plot_ma,
                                graph_showOption=graph_showOption,
                                graph_dir=graph_dir, figsize=figsize, annotfont=annotfont,
                                csv_dir=csv_dir

                                  )
        ac.no_of_trade = back_test.trade_tmie_of_ac
        
        rev=ac.cal_revenue()
        ac.cal_buy_and_hold()
       
        ac.print_txn()
        str_to_print = f'{tickers}: trail stop={ts_percent}, fixed stop loss={fixed_sl}, profit target={profit_target}\n'
        for item in back_test.bp_filters:
            str_to_print+= str(item)+', '

        ac.txn_to_csv(save_path=csv_dir, textbox=str_to_print)
        
        logger.info(f"revenue of {tickers}: {rev}")
        logger.info(f" Back Test of {tickers} done")
        return rev
    
    
    elif isinstance(tickers, list):

        back_test = BackTest()
        back_test.set_buy_strategy(BuyStrategy.FOLLOW_BUYPT_FILTER, bp_filters)
        back_test.set_sell_strategy(strategy=sell_strategy, ts_percent=ts_percent, fixed_sl=fixed_sl, profit_target=profit_target)

        acc_list=[]
        rev_bh_list =[]
        total_finl_cap=0
        for item in tickers:
            ac = StockAccount(item, start, end, capital)
            logger.info(f'---- **** Back Test of {item} started **** ---')
            
            try:
                ac.txn = back_test.roll(ac,
                               trend_col_name=trend_col_name,
                                ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                plot_ma=plot_ma,
                                graph_showOption=graph_showOption,
                                graph_dir=graph_dir, figsize=figsize, annotfont=annotfont,
                                csv_dir=csv_dir,
                                to_print_stock_data = print_all_ac

                                  )
            except Exception as err:
                logger.error(err)
                continue
            rev=ac.cal_revenue()
            rev_buy_hold = ac.cal_buy_and_hold()
            ac.no_of_trade = back_test.trade_tmie_of_ac
            if print_all_ac and ac.txn is not None:
                ac.print_txn()
                str_to_print = f'{item}: trail stop={ts_percent}, fixed stop loss={fixed_sl}, profit target={profit_target}'
                for b in back_test.bp_filters:
                    str_to_print+= str(b)+', '
                ac.txn_to_csv(save_path=csv_dir, textbox=str_to_print)
            
            try:
                total_finl_cap += ac.cal_final_capital()
                rev_bh_list.append(rev_buy_hold)
                logger.info(f"revenue of {item}: {rev}")
                logger.info(f" Back Test of {item} done")
            except Exception as err:
                logger.error(err)
                continue
            acc_list.append(ac)

        avg_rev_bh = sum(rev_bh_list) / len(rev_bh_list)
        final_rev = ( total_finl_cap - capital * len(tickers))/(capital * len(tickers))
        logger.info(f"total revenue of run: {final_rev}")
        back_test.print_revenue(acc_list, final_rev, avg_rev_bh, save_path=csv_dir, textbox=f'trail stop={ts_percent}, fixed stop loss={fixed_sl}, profit target={profit_target}')
        return final_rev

def yearly_test():
    
    #stock_lst_file='../../hotstock100.txt'
    res_save_dir='../../back_test_result/yearly_trail5%_100result.csv'

    start_date = pd.Timestamp('2007-01-01')
    end_date = pd.Timestamp('2023-07-02')
    date_offset = pd.DateOffset(months=6)
    
    date_list = []
    current_date = start_date

    while current_date < end_date:
        date_list.append(current_date)
        current_date += date_offset

    print(date_list)

    
    ts_percent=0.05


    logger.info(f"stock list file got: {stock_lst_file}")
    logger.info(f"yearly test, gap ={date_offset}, earliest={start_date}, latest={end_date}")
    with open(stock_lst_file, 'r') as fio:
        lines = fio.readlines()
    
    fiores = open(res_save_dir, 'w')
    fiores.write(f'yearly test, gap ={date_offset}, earliest={start_date}, latest={end_date}\n')
    fiores.write(f'param: trailing stop: {ts_percent}')
    fiores.write('stock in test: \n')
    fiores.write(f'{lines}\n')
    # write header
    fiores.write(f"start month,end month,overall revenue\n")
    for item in lines:
        item=re.sub(r'\W+', "", item)

        

    to_write=''

    for dt in date_list:
        
        enddt=dt+pd.DateOffset(months=12)
        logger.info(f'processing start date: {dt}, end date: {enddt}')
        
        revenue = runner(lines, dt, enddt, 10000, SellStrategy.TRAILING_STOP , ts_percent=ts_percent,  
                print_all_ac=True, csv_dir=csv_dir)
        fiores.write(f'{dt},{enddt},{revenue}\n')

        
    fiores.close()

    logger.info(f"csv result saved to {res_save_dir}")



    

if __name__ == "__main__":
    start = time.time()
    logger.remove()     # remove deafult logger before adding custom logger
    logger.add(
        sys.stderr,
        level='INFO'

    )
    logger.add(
        f"../log/BackTest_{date.today()}_log.log",
        level='DEBUG'

    )
    logger.info("-- ****  NEW RUN START **** --")


    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    
    short_list=['nvda', 'nio', 'pdd']

    


    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker','-t', help='stock symbol',type=str, default='')
    parser.add_argument('--start','-s', help='start date',  type=str, default='')
    parser.add_argument('--end', '-e', help='end date', type=str, default='')
    parser.add_argument('--capital', '-c', help='initial captial', type=float, default=10000)


    parser.add_argument('--stocklist_file', '-f', help='stock list file dir', type=str, default=None)
    parser.add_argument('--csv_dir', '-v', help='csv folder dir (file name is pre-set), default=../../', type=str, default='../../result/')
    parser.add_argument('--graph_dir', '-g',type=str, default='../../result/')  # no .png
    parser.add_argument('--figsize', type=tuple, default=(40,20))
    parser.add_argument('--figdpi', type=int, default=200)
    parser.add_argument('--showopt', '-o', help='graph show option: \'save\', \'show\', \'no\'', type=str, default='save')
    parser.add_argument('--configfile', '-j', 
                        help='json config file for argument to pass to backtest.runner, if set, will ignore other arguments in command line', 
                        type=str, default=None)
    args=parser.parse_args()

    stockticker=args.ticker
    try:
        assert isinstance(stockticker, str)
    except Exception:
        pass
    else:
        stockticker=stockticker.upper()
        logger.info(f"stock given in cmd prompt: {stockticker}")

    
    configfile = args.configfile

    #yearly_test()
    #end=time.time()
    #logger.info(f"time taken for whole run: {end-start}")
    #exit(0)

    if configfile is None:
        stockstart = args.start
        stockend = args.end
        capital= args.capital
        stock_lst_file = args.stocklist_file
        graph_file_dir = args.graph_dir
        graph_figsize=args.figsize
        graph_dpi=args.figdpi
        graph_show_opt=args.showopt
        csv_dir=args.csv_dir

        if stock_lst_file != None:
            logger.info(f"stock list file got: {stock_lst_file}")
            with open(stock_lst_file, 'r') as fio:
                lines = fio.readlines()
            
            for item in lines:
                item=re.sub(r'\W+', '', item)
                
            runner(lines, stockstart, stockend, capital, 
                SellStrategy.TRAILING_STOP , ts_percent=0.05,  
                bp_filters={sa.BuyptFilter.CONVERGING_DROP, sa.BuyptFilter.IN_UPTREND, sa.BuyptFilter.RISING_PEAK, sa.BuyptFilter.MA_SHORT_ABOVE_LONG},
                    ma_short_list=[50,3],
                    ma_long_list=[200,15],
                    graph_showOption=graph_show_opt,
                print_all_ac=True, csv_dir=csv_dir)

        ## run one stock from cmd
        else:
            # filter: peak bottom and ma above
            # runner(stockticker, stockstart, stockend, capital, 
            #        SellStrategy.Trailing_and_fixed_stoploss, ts_percent=0.05,
            #        bp_filters={sa.BuyptFilter.CONVERGING_DROP, sa.BuyptFilter.IN_UPTREND, sa.BuyptFilter.RISING_PEAK, sa.BuyptFilter.MA_SHORT_ABOVE_LONG},
            #         ma_short_list=[3, 50],
            #         ma_long_list=[13, 150],
            #         graph_showOption=graph_show_opt,
            #         graph_dir=graph_file_dir,
            #        print_all_ac=True, csv_dir=csv_dir)
            
            # filter:sma cross only
            runner(stockticker, stockstart, stockend, capital, 
                SellStrategy.TRAILING_AND_FIXED_SL , ts_percent=0.05,
                bp_filters={sa.BuyptFilter.RISING_PEAK, sa.BuyptFilter.CONVERGING_DROP},
                    ma_short_list=[3],
                    ma_long_list=[9],
                    
                    graph_showOption=graph_show_opt,
                    graph_dir=graph_file_dir,
                print_all_ac=True, csv_dir=csv_dir)
            
            ### ADD FILTER EXAMPLE
            # runner(stockticker, stockstart, stockend, capital, 
            #        SellStrategy.Trailing_and_fixed_stoploss, ts_percent=0.05,
            #        bp_filters={sa.BuyptFilter.Some_filter},     # pass filter here
            #         graph_showOption=graph_show_opt,
            #         graph_dir=graph_file_dir,
            #        print_all_ac=True, csv_dir=csv_dir)
    else:
        with open(configfile, encoding = 'utf-8') as fio:
            confjson = json.load(fio)
        
        stockticker= confjson['ticker']
        stockstart = confjson['start']
        stockend= confjson['end']
        capital = confjson.get("capital", 10000)
        ts_percent = confjson.get("stop loss percent", 0.05)
        fsl_percent = confjson.get("fixed stop loss percent", None)
        profit_target = confjson.get("profit target", None)
        bp_filters_list = confjson.get("buy point filters", [])
        sell_strategy_str = confjson.get("sell strategy", None)
        buy_strategy_str = confjson.get("buy strategy", "FOLLOW_BUYPT_FILTER")
        
        ma_short_list = confjson.get("ma short", [])
        ma_long_list = confjson.get("ma long", [])
        plot_ma = confjson.get("plot ma", [])
        graph_show_opt =confjson.get("graph show option", 'no')
        graph_file_dir = confjson.get("graph dir", '../../result')
        csv_dir = confjson.get("csv dir", '../../result')
        print_all_ac =confjson.get("print all ac", False)
        annotfont = confjson.get("graph font size", 4)
        
        print(type(stockticker))

        bp_filter = set()
        sell_stra = None
        buy_strategy = None

        bp_filters_list = [item.upper() for item in bp_filters_list]
        if isinstance(sell_strategy_str, str):
            sell_strategy_str = sell_strategy_str.upper()


        for data in sa.BuyptFilter:
            if data.name in bp_filters_list:
                bp_filter.add(data)

        for data in SellStrategy:
            if data.name == sell_strategy_str:
                sell_stra=data
                break
        logger.debug(f"sell strategy got: {sell_stra}")

        if isinstance(buy_strategy_str, str):
            buy_strategy_str.upper()
        for data in BuyStrategy:
            if data.name == buy_strategy_str:
                buy_strategy = data

        logger.debug("config received from config file json:")
        logger.debug(confjson)

        runner(stockticker, stockstart, stockend, capital, 
               buy_strategy=buy_strategy,
              sell_strategy=sell_stra, 
              ts_percent=ts_percent,
              fixed_sl=fsl_percent,
              profit_target=profit_target,
                bp_filters=bp_filter,
                ma_short_list=ma_short_list,
                ma_long_list=ma_long_list,
                plot_ma= plot_ma,
                graph_showOption=graph_show_opt,
                graph_dir=graph_file_dir,
                print_all_ac=print_all_ac, csv_dir=csv_dir,
                annotfont=annotfont)








    end=time.time()
    logger.info(f"time taken for whole run: {end-start}")


## EXample 
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=tsm -c=10000 -v='../../back_test_result/'

## save graph and save roll result csv
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=pdd -c=10000 -o=no -v=../../back_test_result/
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=pdd -c=10000 -o=save -v=../../back_test_result/ -g=../../back_test_result/PDD_0816
    




