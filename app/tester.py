from stock_analyser import StockAnalyser
from stock_analyser import trial_runner
import configparser
import argparse, json

config = configparser.ConfigParser()
config.read('test_config.ini')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', '-j', help='config file in json format', type=str)
    args=parser.parse_args()
    configfile = args.configfile

    

    with open('configs/backtest_config.json', encoding = 'utf-8') as f:
        json_config = json.load(f)
    print(json_config['ticker'])
    print(json_config['start'])
    print(json_config['ma_short_list'])
    print(type(json_config['ma_short_list']))
    print(type(json_config['ma_short_list'][0]))
    print(type(json_config['print_all_ac']))
    print(json_config['print_all_ac'])
  