#!/bin/bash

#Run the first command
export PYTHONPATH="$PWD"
echo "Testing stock_analyser.py ..."
echo "Running: python app/stock.py -t=pdd -s=2023-05-01 -e=2023-08-20 -g=../unit_test_result/ -v=../unit_test_result"
python app/stock_analyser.py -t=pdd -s=2023-05-01 -e=2023-08-20 -g=../unit_test_result/ -v=../unit_test_result

# Check the exit status of the first command
if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test of stock_analyser"
    exit 1
fi




echo "Running: python app/stock_analyser.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../unit_test_result/ -v=../unit_test_result ..."
python app/stock_analyser.py -f=./app/configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../unit_test_result/ -v=../unit_test_result

if [ $? -ne 0 ]; then
    echo "Error occurred in 2nd test of stock_analyser"
    exit 1
fi

echo "Testing backtest.py ..."



echo "Running backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../unit_test_result -g=../unit_test_result ..."
python app/backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../unit_test_result -g=../unit_test_result

if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test of backtest"
    exit 1
fi

echo "Running: python backtest.py -j=./app/configs/backtest_config_example.json ..."
python app/backtest.py -j=./app/configs/backtest_config_example.json
if [ $? -ne 0 ]; then
    echo "of backtest"
    exit 1
fi

echo "Running: python backtest.py -j=./app/configs/list_of_stock_1yr_config.json ..."
python app/backtest.py -j=./app/configs/list_of_stock_1yr_config.json


if [ $? -ne 0 ]; then
    echo "Error occurred in 3rd test of backtest"
    exit 1
fi

echo "All commands completed successfully."
