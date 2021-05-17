
# import relevant modules
import os
import pandas as pd
import csv
import datetime




# define parameters - target filename / directory and the model in which I calculate the "noise"
ticker = "$NFLX"
os.chdir("./raw_tweet_dumps/" + ticker + "/")

# 100 rt = 1 tweet
rt_point = 0.01
# 100 replies = 1 tweet
reply_point = 0.01
# 1000 likes = 1 tweet
like_point = 0.001




# get all the csv filenames in a given dump folder, and populate a list with all of the filenames
file_list = os.listdir()
try:
    file_list.remove('.DS_Store')

except:
    pass

file_list = sorted(file_list)




# open a new csv file to store the timeseries for ALL of the csv files in the directory abov
timeseries_csv = open("../" + str(ticker)+ "-"'timeseries.csv', 'a')
data_container = []

# prepare a list to be populated with dictionary items
fields = ['time', 'noise']
output_writer = csv.DictWriter(timeseries_csv, fieldnames=fields)
output_writer.writeheader()

# for each csv file title in the list
for file in file_list:

    # parse the filename to get the datetime
    new_name = '-'.join(file.split('-')[2:])
    date = new_name[:-4]
    date = date.split('-')
    date = [i.zfill(2) for i in date]
    date = '-'.join(date)

    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")

    row_obj = {'time': date, 'noise': 0}
    data_container.append(row_obj)

    # read csv with pd and load the data - now we have the dataframe
    df = pd.read_csv(file)
    index = df.index
    num_rows = range(len(index))


    # for each row in the given csv file (which represents each tweet in a given day's .csv file)
    for row_idx in num_rows:

        tweet = df.iloc[row_idx, 10]
        print(tweet)
        replies_count = df.iloc[row_idx, 15]
        rt_count = df.iloc[row_idx, 16]
        likes_count = df.iloc[row_idx, 17]
        cashtags = set(df.iloc[row_idx, 19].strip('[').strip(']').strip('\'').split(','))

        # spam filter for tweets with too many /
        # unspecific tags such that any tweets with other cashtags other than the exact ticker
        # (e.g. $AAPL) will not count towards a noise on the topic of $AAPL
        if len(cashtags) > 1:
            tweet_noise_level = 1
            tweet_noise_level += (rt_count * rt_point)+(replies_count * reply_point)+(likes_count * like_point)
            # print(tweet, tweet_noise_level)
            # print(tweet, cashtags, len(cashtags), tweet_noise_level)
            row_obj['noise'] += tweet_noise_level


        else:
            pass
    print(row_obj)
    output_writer.writerow(row_obj)

print(data_container)

timeseries_csv.close()




