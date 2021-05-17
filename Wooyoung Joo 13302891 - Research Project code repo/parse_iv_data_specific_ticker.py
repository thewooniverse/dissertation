# this script was used to parse $GOOG data specifically

import os
import csv
import datetime
import pandas as pd


ticker = '$GOOG'


os.chdir('./raw_iv_timeseries')
filenames = os.listdir('./' + ticker)

try:
    filenames.remove('.DS_Store')
except:
    pass






# create the new single timeseries file - 2 columns,
iv_timeseries = open("../iv_timeseries/" + str(ticker)+ "-"'timeseries.csv', 'a')
fields = ['time', 'iv_30']

output_writer = csv.DictWriter(iv_timeseries, fieldnames=fields)
output_writer.writeheader()



for file in filenames:

    date_part = file.split('_')[1]
    date = date_part[:-4]
    quote_date = ''
    iv_30 = ''
    row_obj = {'time': date, 'iv_30': 0}


    with open("./" + ticker + "/" + file, newline='') as csvfile:
        print(csvfile)
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            if row[1] == 'quote_datetime':
                pass
            else:
                row_obj['iv_30'] = row[2]

    # now that I have the date and the iv_30 values extracted from these timeseries, I can now wrie them into the singular timeseries fiel
    output_writer.writerow(row_obj)














