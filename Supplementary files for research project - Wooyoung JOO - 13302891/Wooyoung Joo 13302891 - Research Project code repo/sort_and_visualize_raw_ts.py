

# import modules
import os
import pandas as pd
from pandas.plotting import lag_plot
from matplotlib import pyplot as plt
import numpy as np


# sort and visualize raw noise timeseries data
os.chdir("./tweet_noise_timeseries/")
noise_data_file_names = os.listdir(os.getcwd())
try:
    noise_data_file_names.remove('.DS_Store')

except:
    pass




# this list will contain all ticker: [noise_df, iv_df] pairs to be finally outer joined
ticker_noise_dataframes = {}

# load the noise level dataframes - sort and visualize them
for file in noise_data_file_names:
    df = pd.read_csv(file, index_col='time')
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    df.sort_values(by=['time'], inplace=True)


    ticker = file.split('-')[0]
    ticker_noise_dataframes.update({ticker: df})

    # container = {'ticker': file.split('-')[0], 'noise_dataframe': df}
    # noise_ts_dataframes.append(container)




# sort add the IV30 timeseries dataframes
ticker_iv30_dataframes = {}


os.chdir('../iv_timeseries/')
iv_data_file_names = os.listdir(os.getcwd())

try:
    iv_data_file_names.remove('.DS_Store')
except:
    pass




# sort and load
for file in iv_data_file_names:
    iv_df = pd.read_csv(file, index_col='time')
    iv_df.index = pd.to_datetime(iv_df.index)
    iv_df.sort_values(by=['time'], inplace=True)

    time_x = iv_df.index
    noise_y = iv_df['iv_30']

    ticker = file.split('-')[0]
    ticker_iv30_dataframes.update({ticker: iv_df})








# fianlly - combine the two dataframe to produce final csv dataset, intrapolate the missing (or null iv30 data)
os.chdir('../final_dataset/')

for key in ticker_iv30_dataframes.keys():

    ticker_iv_df = ticker_iv30_dataframes[key]
    ticker_noise_df = ticker_noise_dataframes[key]
    right_merged = pd.merge(ticker_iv_df, ticker_noise_df, how="right", on=['time'])


    iv_series = right_merged['iv_30']
    # replace all 0s and and nan with correct np NaN
    iv_series.replace(iv_series['2020-01-01'], np.nan, inplace=True)
    iv_series.replace(0, np.nan, inplace=True)
    right_merged['iv_30'] = iv_series

    # intrapolate data
    intrapolated_data = right_merged.interpolate(method='linear', limit_direct='both', axis=0)

    intrapolated_data.to_csv(f'{key}.csv')

    time_x = intrapolated_data.index
    noise_y = intrapolated_data['noise']
    iv30_y = intrapolated_data['iv_30']
    plt.plot(time_x, noise_y, label=key + ' raw noise timeseries')
    plt.plot(time_x, iv30_y, label=key + ' IV30(introapolated)_timeseries')
    plt.tight_layout()
    plt.legend()
    plt.show() # comment this out when we no longer need to show plot.
















