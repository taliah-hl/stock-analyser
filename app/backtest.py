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
import argparse
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
    Default=0
    Uptrend_converging_bottom =1

class SellStrategy(enum.Enum):
    Default=0
    Trailing_stop =1
    Buy_and_hold=2
    Profit_target=3
    Fixed_stop_loss=4
    Trailing_and_fixed_stoploss =5
    TS_FSL_PT=5
    Mix=10

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
        #self.txn = res
        return self.txn
    
    def print_txn(self):
        logger.debug(f"----  Transaction Table of {self.ticker}  ----")
        logger.debug(tabulate(self.txn, headers='keys', tablefmt='psql', floatfmt=".2f"))

    def txn_to_csv(self, save_path:str=None, textbox: str=None):
        if save_path is None:
            save_path = f"../../back_test_result/roll_result_{self.ticker}_{self.start}_{self.end}.csv"
        else:
            save_path = save_path+f"roll_result_{self.ticker}_{self.start}_{self.end}.csv"
        self.txn.to_csv(save_path)
        with open(save_path, 'a') as fio:
            fio.write(textbox)

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

    def cal_final_capital(self):
        if self.txn is not None:
            return self.txn['cash'][-1]
        else:
            logger.warning("no transaction, cannot get final capital, you may want to roll the ACC first.")
            return None


class BackTest():

    def __init__(self):
        pass
        self.buy_strategy: BuyStrategy=BuyStrategy.Uptrend_converging_bottom
        self.sell_strategy: SellStrategy=SellStrategy.Trailing_stop
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


    def set_buy_strategy(self, strategy, buypt_filters: set=set()):
        self.buy_strategy = strategy
        self.bp_filters = buypt_filters


    def set_sell_strategy(self, strategy, ts_percent: float=None, 
                          profit_target: float=None, fixed_st: float=None):
        self.sell_strategy = strategy

        if strategy==SellStrategy.Trailing_stop:
            
            if pd.isna(ts_percent):
                raise Exception("sell strategy selected as trailing stop, but trail stop percentage not provided!\n provide trail stop percentage in ts_percent in BackTest.set_sell_strategy()\nprogram exit.")
                
            self.trail_loss_percent = ts_percent
        elif strategy==SellStrategy.TS_FSL_PT or strategy==SellStrategy.Trailing_and_fixed_stoploss:
            self.trail_loss_percent = ts_percent
            self.profit_target=profit_target
            self.fixed_st=fixed_st

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
                    csv_dir: str=None
                  )->sa.StockAnalyser:
        """
            init stock using StockAnalyser.default_analyser
        """
        stock = sa.StockAnalyser()
        if not bp_filters:
            bp_filters = self.bp_filters
        if graph_dir is None:
            graph_dir=f'../../{ticker}_{start}_{end}'

        if self.buy_strategy == BuyStrategy.Uptrend_converging_bottom:

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
                csv_dir=csv_dir
            )
            return stock
        
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
        return cur_row

        

        
    def check_sell(self, strategy, prev_row: dict, cur_row: dict, latest_high: float, cur_price: float, last_buy_date=None):
        """
            input: prev row
            return: cur row, bool: True=sold, False=not sold
        """
        
        if strategy== SellStrategy.Trailing_stop:
            if cur_price < (latest_high * (1-self.trail_loss_percent)):
                # sell
                cur_row = self.sell(prev_row, cur_row, trigger_price= math.floor(latest_high * (1-self.trail_loss_percent)*100)/100, trigger='trail stop')
                return (cur_row, True)
        elif strategy==SellStrategy.TS_FSL_PT or strategy==SellStrategy.Trailing_and_fixed_stoploss:
            if (self.trail_loss_percent is not None 
                and cur_price < (latest_high * (1-self.trail_loss_percent)) ):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.floor(latest_high*(1-self.trail_loss_percent)*100)/100, trigger='trail stop')
                return (cur_row, True)
                
            elif (self.fixed_st is not None 
                  and cur_price <( (self._stock_table['close'][last_buy_date]) * (1-self.fixed_st))):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.floor(self._stock_table['close'][last_buy_date] * (1-self.fixed_st)*100)/100, trigger='fixed ST')
                return (cur_row, True)

            elif ( self.profit_target is not None
                  and cur_price >=  self._stock_table['close'][last_buy_date] * (1+self.profit_target)):
                cur_row = self.sell(prev_row, cur_row, trigger_price=math.floor(self._stock_table['close'][last_buy_date] * (1+self.profit_target)*100)/100, trigger='profit target')
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
           csv_dir: str=None )->pd.DataFrame:

        try:
            stock = self.set_stock(ticker=acc.ticker, start=acc.start, end=acc.end,
                                   trend_col_name=trend_col_name,
                                   bp_filters=bp_filters,
                                   ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                   plot_ma=plot_ma,
                                   extra_text_box=extra_text_box,
                                   graph_showOption=graph_showOption, graph_dir=graph_dir,
                                   figsize=figsize, annotfont=annotfont, 
                                   csv_dir=csv_dir
                                   )
        except Exception as err:
            logger.error(err)
            return None

        
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
            
             ## 2. Check buy
            if ((self._stock_table['day of interest'][idx]==sa.DayType.BUYPT) and (not is_holding)):
                #logger.debug("check buy triggered")
                # buy with that day close price

                try:
                    share_num=math.floor(txn_table['cash'][idx-1] / txn_table['close price'][idx])
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
                                               trigger_price=txn_table['close price'][idx])
                break
            #logger.debug(f"today action: {txn_table.iloc[idx, action_col]}")
            idx+=1

        for i in range(0, len(txn_table)):
            txn_table.iloc[i, mv_col] = txn_table['close price'][i] * txn_table['share'][i]
            txn_table.iloc[i, ass_col] = txn_table['cash'][i] + txn_table['MV'][i]

        return txn_table
        
    def print_revenue(self, ac_list:list, total_revenue, save_path:str=None, textbox: str=None):
        """
        input: list of acc
        all accs need to be roll first
        time range of all acc need to be same
        print revenue to csv file

        """
        
        table=[]
        if save_path is None:
            save_path = '../../all_revenue_.csv'
        else:
            save_path = save_path + 'all_revenue_.csv'

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

        revenue_table.to_csv(save_path)
        with open(save_path, 'a') as fio:
            fio.write(textbox)
        logger.info(f"overall revenue csv saved to {save_path}")




def runner(tickers, start:str, end:str, capital:float, 
           sell_strategy, ts_percent: float=None, fixed_st: float=None, profit_target:  float=None,
           buy_strategy=None,
           trend_col_name: str='slope signal',
           bp_filters: set=set(),
           ma_short_list: list=[], ma_long_list=[],
           plot_ma: list=[],
           graph_showOption: str='save', graph_dir: str=None, figsize: tuple=(36,24), annotfont: float=6,
           csv_dir:str='../../', print_all_ac:bool=True)->float:
    """
    return 

    overall revenue
    """

    if isinstance(tickers, str):
        ac = StockAccount(tickers, start, end, capital)
        logger.info(f'---- **** Back Test of {tickers} stated **** ---')
        back_test = BackTest()
        back_test.set_buy_strategy(BuyStrategy.Uptrend_converging_bottom, bp_filters)
        back_test.set_sell_strategy(strategy=sell_strategy, ts_percent=ts_percent, fixed_st=fixed_st, profit_target=profit_target)
        ac.txn = back_test.roll(ac,
                               trend_col_name=trend_col_name,
                                bp_filters=bp_filters,
                                ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                plot_ma=plot_ma,
                                graph_showOption=graph_showOption,
                                graph_dir=graph_dir, figsize=figsize, annotfont=annotfont,
                                csv_dir=csv_dir

                                  )
        
        rev=ac.cal_revenue()
        if print_all_ac:
            ac.print_txn()
            ac.txn_to_csv(save_path=csv_dir, textbox=f'{tickers}: trail stop={ts_percent}, fixed stop loss={fixed_st}, profit target={profit_target}\nrevenue: {rev}')
        
        logger.debug(f"revenue of {tickers}: {rev}")
        logger.info(f" Back Test of {tickers} done")
        return rev
    
    
    elif isinstance(tickers, list):

        back_test = BackTest()
        back_test.set_buy_strategy(BuyStrategy.Uptrend_converging_bottom, bp_filters)
        back_test.set_sell_strategy(strategy=sell_strategy, ts_percent=ts_percent, fixed_st=fixed_st, profit_target=profit_target)

        acc_list=[]
        total_finl_cap=0
        for item in tickers:
            ac = StockAccount(item, start, end, capital)
            logger.info(f'---- **** Back Test of {item} started **** ---')
            
            try:
                ac.txn = back_test.roll(ac,
                               trend_col_name=trend_col_name,
                                bp_filters=bp_filters,
                                ma_short_list=ma_short_list, ma_long_list=ma_long_list,
                                plot_ma=plot_ma,
                                graph_showOption=graph_showOption,
                                graph_dir=graph_dir, figsize=figsize, annotfont=annotfont,
                                csv_dir=csv_dir

                                  )
            except Exception as err:
                logger.error(err)
                continue
            rev=ac.cal_revenue()
            if print_all_ac and ac.txn is not None:
                ac.print_txn()
                ac.txn_to_csv(save_path=csv_dir, textbox=f'{item}: trail stop={ts_percent}, fixed stop loss={fixed_st}, profit target={profit_target}\nrevenue: {rev}')
            
            try:
                total_finl_cap += ac.cal_final_capital()
                logger.info(f"revenue of {item}: {rev}")
                logger.info(f" Back Test of {item} done")
            except Exception as err:
                logger.error(err)
                continue
            acc_list.append(ac)

        final_rev = ( total_finl_cap - capital * len(tickers))/(capital * len(tickers))
        logger.info(f"total revenue of run: {final_rev}")
        back_test.print_revenue(acc_list, final_rev, save_path=csv_dir, textbox=f'trail stop={ts_percent}, fixed stop loss={fixed_st}, profit target={profit_target}')
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
        
        revenue = runner(lines, dt, enddt, 10000, SellStrategy.Trailing_stop, ts_percent=ts_percent,  
                print_all_ac=False, csv_dir=csv_dir)
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
        f"../../BackTest_{date.today()}_log.log",
        level='INFO'

    )
    logger.info("-- ****  NEW RUN START **** --")


    watch_list = ['amd', 'sofi', 'intc', 'nio', 
                  'nvda', 'pdd', 'pltr', 'roku',
                  'snap', 'tsla', 'uber', 'vrtx',
                  'xpev']
    
    short_list=['nvda', 'nio', 'pdd']

    


    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker','-t', help='stock symbol',type=str, default='NVDA')
    parser.add_argument('--start','-s', help='start date',  type=str, default='2022-07-20')
    parser.add_argument('--end', '-e', help='end date', type=str, default='2023-08-03')
    parser.add_argument('--capital', '-c', help='initial captial', type=float, default=10000)

    parser.add_argument('--buy', '-b', help='buy strategy, 1=conv drop', type=int, default=1)
    parser.add_argument('--sell', '-p', help='sell strategy, number code same as class SellStrategy', type=int, default=5)

    parser.add_argument('--stocklist_file', '-f', help='stock list file dir', type=str, default=None)
    parser.add_argument('--csv_dir', '-v', help='csv folder dir (file name is pre-set), default=../../', type=str, default='../../')
    parser.add_argument('--graph_dir', '-g',type=str, default='../../untitled')  # no .png
    parser.add_argument('--figsize', type=tuple, default=(40,20))
    parser.add_argument('--figdpi', type=int, default=200)
    parser.add_argument('--showopt', '-o', help='graph show option: \'save\', \'show\', \'no\'', type=str, default='save')
    args=parser.parse_args()

    stockticker=args.ticker
    try:
        assert isinstance(stockticker, str)
    except Exception:
        pass
    else:
        stockticker=stockticker.upper()
        logger.info(f"stock given in cmd prompt: {stockticker}")

    stockstart = args.start
    stockend = args.end
    capital= args.capital
    sells=args.sell
    buys=args.buy
    stock_lst_file = args.stocklist_file
    graph_file_dir = args.graph_dir
    graph_figsize=args.figsize
    graph_dpi=args.figdpi
    graph_show_opt=args.showopt
    csv_dir=args.csv_dir

    #yearly_test()
    #end=time.time()
    #logger.info(f"time taken for whole run: {end-start}")
    #exit(0)

    if stock_lst_file != None:
        logger.info(f"stock list file got: {stock_lst_file}")
        with open(stock_lst_file, 'r') as fio:
            lines = fio.readlines()
        
        for item in lines:
            item=re.sub(r'\W+', '', item)
            
        runner(lines, stockstart, stockend, capital, 
               SellStrategy.Trailing_stop, ts_percent=0.05,  
               bp_filters={sa.BuyptFilter.Converging_drop, sa.BuyptFilter.In_uptrend, sa.BuyptFilter.Rising_peak, sa.BuyptFilter.SMA_short_above_long},
                ma_short_list=[50,3],
                ma_long_list=[200,15],
                graph_showOption='no',
               print_all_ac=False, csv_dir=csv_dir)

    ## run one stock from cmd
    else:
        runner(stockticker, stockstart, stockend, capital, 
               SellStrategy.Trailing_and_fixed_stoploss, ts_percent=0.05,
               bp_filters={sa.BuyptFilter.Converging_drop, sa.BuyptFilter.In_uptrend, sa.BuyptFilter.Rising_peak, sa.BuyptFilter.SMA_short_above_long},
                ma_short_list=[3],
                ma_long_list=[9],
                graph_showOption='save',
                graph_dir=graph_file_dir,
               print_all_ac=True, csv_dir=csv_dir)

    end=time.time()
    logger.info(f"time taken for whole run: {end-start}")


## EXample 
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=tsm -c=10000 -v='../../back_test_result/'

## save graph and save roll result csv
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=pdd -c=10000 -o=no -v=../../back_test_result/
# python backtest.py -s=2022-08-01 -e=2023-08-16 -t=pdd -c=10000 -o=save -v=../../back_test_result/ -g=../../PDD
    




