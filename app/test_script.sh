#!/bin/bash

# Run the first command

echo "Testing stock_analyser.py ..."
echo "Running: python stock.py -t=pdd -s=2023-05-01 -e=2023-08-20"
python stock_analyser.py -t=pdd -s=2023-05-01 -e=2023-08-20

# Check the exit status of the first command
if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test"
    exit 1
fi




echo "Running: python stock_analyser.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../../unit_test_result/ -v=../../unit_test_result"
python stock_analyser.py -f=./configs/2stocks.txt -s=2022-08-01 -e=2023-08-01 -g=../../unit_test_result/ -v=../../unit_test_result

if [ $? -ne 0 ]; then
    echo "Error occurred in 2nd test"
    exit 1
fi

echo "Testing backtest.py ..."



echo "Running backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../../unit_test_result -g=../../unit_test_result"
python backtest.py -t=pdd -s=2022-08-01 -e=2023-08-16 -c=10000 -o=no -v=../../unit_test_result -g=../../unit_test_result

if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test"
    exit 1
fi

python backtest.py -j=./configs/backtest_config_example.json
echo "Running: python backtest.py -j=./configs/backtest_config_example.json"
if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test"
    exit 1
fi

python backtest.py -j=./configs/list_of_stock_1yr_config.json
echo "Running: python backtest.py -j=./configs/list_of_stock_1yr_config.json"

if [ $? -ne 0 ]; then
    echo "Error occurred in 1st test"
    exit 1
fi

echo "All commands completed successfully."