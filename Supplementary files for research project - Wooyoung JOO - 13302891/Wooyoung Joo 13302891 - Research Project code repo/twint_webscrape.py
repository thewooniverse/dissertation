import twint
import datetime
import os
from pytz import timezone

# This script is used to scrape Twitter using the Twint library available at (https://github.com/twintproject)
# define when I want to start observing and for how long
start_datetime = datetime.datetime(2019, 12, 30)
start_datetime = start_datetime.replace(tzinfo=timezone('UTC'))
time_window = 370
search_term = "$GOOG"


try:
    os.mkdir('./raw_tweet_dump')
except:
    pass

# define a Twint search function that outputs the correct csv files
# passes a datetime object, and a int time window, and the term that you want to search (string)

def twint_search(dt, window, term):

    datetime_list = []

    for day in range(window):
        tdelta = datetime.timedelta(days=day)
        new_date = dt + tdelta
        new_date = new_date.strftime('%Y-%-m-%-d')
        datetime_list.append(new_date)


    for idx in list(range(len(datetime_list))):
        c = twint.Config()
        c.Search = term
        c.Store_csv = True
        c.Since = datetime_list[idx]
        c.Until = datetime_list[idx+1]
        c.Output = "./raw_tweet_dump/" + term + "--" + datetime_list[idx] + ".csv"
        twint.run.Search(c)


twint_search(start_datetime, time_window, search_term)














