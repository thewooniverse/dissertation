The code contained in this repository is mostly written by Wooyoung Joo for his undergraduate Research Project at Birkbeck College, Unviersity of London.
Some code snippets were adapted from toher sources (cited below) for the final granger causality test scripts in the final stage of the research.


The code repository contains a few scripts and folders used to gather, process and analyze the data:

- twint_webscrapte.py is a script that uses the Twint library (https://github.com/twintproject/twint/tree/master/twint) to scape Twitter for Twitter relating to the research topic at hand (stock tickers) for specific dates
- based on the values of my own noise valuing crierion developed for calculating noise levels.
- raw_tweet_dumps contains the result of this script

- calculate_noise.py is a script that runs through all of the tweets scraped for a given topic, to produce a raw timeseries dataset
- prase_iv_data.py and parse_iv_data_specific_ticker.py both achieve the same goal of parsing the raw_iv_timeseries data dumped from CBOE datashop to fit the purpose

- sort_and_visualize_raw_ts.py firstly inrapolates the iv_data for missing values then takes the two raw, intrapolated timeseries datasets for both iv30 and Twitter noise level and combines it into the final dataset. This script also helps visualize the raw dataset.


- granger_causality_test.py is a script inspired and adapted from the tuorials below. This script takes the final dataset for a given $ticker in the final dataset folder, processes the data and runs a granger causality test on them
- to determine whether a lagged version of X is helpful in increasing the predictiveness of vector autoregressive models using lagged versions of Y alone.
- finally in the scrip the normalized version of the dataset is visualized.

Tutorial and guide for Granger Causality testing:
- https://rishi-a.github.io/2020/05/25/granger-causality.html
- https://towardsdatascience.com/granger-causality-and-vector-auto-regressive-model-for-time-series-forecasting-3226a64889a6
- https://medium.com/swlh/using-granger-causality-test-to-know-if-one-time-series-is-impacting-in-predicting-another-6285b9fd2d1c



The readme may be incomplete; for any inquiries regarding the code written or the full tweet dump dataset please email me at:
wjoo01@mail.bbk.ac.uk

Thank you



