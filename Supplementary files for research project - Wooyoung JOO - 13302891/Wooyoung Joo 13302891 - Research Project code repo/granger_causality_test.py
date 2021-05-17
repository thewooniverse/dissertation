# Granger Causality test as Toda, H. Y and T. Yamamoto (1995). Statistical inferences in vector autoregressions with possibly integrated processes. Journal of Econometrics, 66, 225-250.
# add export functionality
# for each ticker



# import all relevant models
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.stattools import durbin_watson
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from math import sqrt

import os
import pickle
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats




# import the correct dataframes from sor and visualize raw_ts.
ticker = "$GOOG"
os.chdir('./final_dataset')



# ADF and KPSS tests to test for stationarity
data = pd.read_csv(ticker +'.csv', index_col='time')
data.index = pd.to_datetime(data.index)
data = data.dropna()
print(data)


# CHECK FOR STATIONARITY
from pandas.plotting import lag_plot

f2, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 5))
f2.tight_layout()

lag_plot(data['noise'], ax=ax4)
ax4.set_title('noise level on Twiter');

lag_plot(data['iv_30'], ax=ax5)
ax5.set_title('iv_30(intrapolated)');

# plt.show()

# maintain a duplicate copy
rawData = data.copy(deep=True)





# splitting the data into a training and test set, which will be necessary for VAR analyses
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]




# Augmented Dickey-Fuller tests to test for unit roots for existing datasets
X1 = np.array(data['noise'])
X1 = X1[~np.isnan(X1)]
result = adfuller(X1)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


X2 = np.array(data['iv_30'])
X2 = X2[~np.isnan(X2)]
result = adfuller(X2)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# KPSS test for stationarity
def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(X1)
kpss_test(X2)



### we find that the noise level array is NOT stationary ###




# differencing the data to achieve stationarity
data['noise'] = data['noise'] - data['noise'].shift(1)
data['iv_30'] = data['iv_30'] - data['iv_30'].shift(1)
data = data.dropna()
print(data)

X1 = np.array(data['noise'])
X1 = X1[~np.isnan(X1)]
result = adfuller(X1)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


X2 = np.array(data['iv_30'])
X2 = X2[~np.isnan(X2)]
result = adfuller(X2)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# KPSS test for stationarity
def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(X1)
kpss_test(X2)


# we can confirm in both the plots and the statistical tests that the series is stationary.

f2, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 5))
f2.tight_layout()

lag_plot(data['noise'], ax=ax4)
ax4.set_title('noise level on Twiter');

lag_plot(data['iv_30'], ax=ax5)
ax5.set_title('iv_30(intrapolated)');

# plt.show()


# m = mximum order of integration for the group of timeseries.
# for us m=1


######## normalizing a timeseries to confine it within 1 - for better visualization

time_x = data.index

new_data = data.copy(deep=True)
normal_noise_y = new_data['noise']
normal_iv30_y = new_data['iv_30']

normal_noise_y = (normal_noise_y - normal_noise_y.min()) / (normal_noise_y.max() - normal_noise_y.min())
normal_iv30_y = (normal_iv30_y - normal_iv30_y.min()) / (normal_iv30_y.max() - normal_iv30_y.min())

plt.plot(time_x, normal_noise_y, label="normalized " + ticker + ' change in noise timeseries')
plt.plot(time_x, normal_iv30_y, label="normalized " + ticker + ' change in IV30(introapolated) timeseries')


plt.tight_layout()
plt.legend()
# plt.show()






### Setting up and running the VAR (Vector Auto Regressive) models
# obtaining the lag P for VAR
rawData = rawData.dropna()

model = VAR(rawData) #recall that rawData is w/o difference operation
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    result = model.fit(i)
    try:
        print('Lag Order =', i)
        print('AIC : ', result.aic)
        print('BIC : ', result.bic)
        print('FPE : ', result.fpe)
        print('HQIC: ', result.hqic, '\n')
    except:
        continue



# Select the p lag value according to the Aikake Information Criterion, AIC
model = VAR(train)
model_fitted = model.fit(10)



# check for serial correlation of residuals (errors) using the Durbain Wtson statistics test
# close the value is to 2, there is no significant serial correlation. closer to 0 indicates positive serial correlation and 4 implied negative serial correlation
out = durbin_watson(model_fitted.resid)

for col, val in zip(data.columns, out):
    print(col, ':', round(val, 2))

# check for cointegration
result=ts.coint(data['noise'], data['iv_30'])
print(result)
# p value is < 0.05 therefore, the null hypothesis is rejected.


model = VAR(train)
model_fitted = model.fit(10)
#get the lag order
lag_order = model_fitted.k_ar









### test for granger non-causality

maxlag=lag_order #becuase we got this value before. We are not suppose to add 1 to it

test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.


    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

train_set = grangers_causation_matrix(train, variables = train.columns)


data_result = grangers_causation_matrix(data, data.columns)
data_result.to_csv(f"{ticker}_result.csv")


print(data_result)
# for google, there is a two-way granger causality, and noise granger causes iv_30 much more significantly than the oher way around










