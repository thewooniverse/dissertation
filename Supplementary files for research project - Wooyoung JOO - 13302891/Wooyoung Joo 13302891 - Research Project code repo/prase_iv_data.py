# script to parse data from merged csv dump
import os
import csv
import pandas as pd
import datetime





underlying_symbols = ['AAPL', 'AMZN', 'FB', 'NFLX']

# change directory to the dumped file
os.chdir('./raw_iv_timeseries/IV')
filenames = os.listdir('.')
print(filenames)

# remove DS store (unique to mac)
try:
    filenames.remove('.DS_Store')
except:
    pass





for symbol in underlying_symbols:
    ticker = f"${symbol}"
    iv_timeseries = open("../../iv_timeseries/" + ticker + "-"'timeseries.csv', 'a')
    fields = ['time', 'iv_30']
    output_writer = csv.DictWriter(iv_timeseries, fieldnames=fields)
    output_writer.writeheader()

    for file in filenames:

        date_part = file.split('_')[1]
        date = date_part[:-4]
        quote_date = ''
        iv_30 = ''
        row_obj = {'time': date, 'iv_30': 0}


        with open(file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                if row[0] == symbol:
                    row_obj['iv_30'] = row[2] #third column, row[i] == 2 is the iv30 column in the dumped file

        output_writer.writerow(row_obj)










