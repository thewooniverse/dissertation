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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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



# exploring the data and checking for stationarity
data = pd.read_csv(ticker +'.csv', index_col='time')
data.index = pd.to_datetime(data.index)
data = data.dropna()
# print(data)

# CHECK FOR STATIONARITY in visualized graph
from pandas.plotting import lag_plot

f2, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 5))
f2.tight_layout()

lag_plot(data['noise'], ax=ax4)
ax4.set_title('noise level on Twiter');

lag_plot(data['iv_30'], ax=ax5)
ax5.set_title('iv_30(intrapolated)');

# plt.show()
# we can see that the data is NOT stationary




# maintain a duplicate copy of the raw data
raw_data = data.copy(deep=True)


print(raw_data.iv_30.describe().T)
print(raw_data.noise.describe().T)



# splitting the data into a training and test set, which will be necessary for VAR analyses and prediction model bulding
n_obs = 30
X_train, X_test = data[0:-n_obs], data[-n_obs:]
print(X_train.shape, X_test.shape) # correctly produces the two datasets


# we transform the dataset by differencing it in order to achieve stationarity
transformed_data = X_train.diff().dropna()
# print(transformed_data.head())




# Augmented Dickey-Fuller tests to test for unit roots for the dataset to est for stationarity
X1 = np.array(transformed_data['noise'])
X1 = X1[~np.isnan(X1)]
result = adfuller(X1)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

X2 = np.array(transformed_data['iv_30'])
X2 = X2[~np.isnan(X2)]
result = adfuller(X2)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# if p value is less than 0.05 the null hypothesis is rejected and the data is stationry.







# confirmation through a scatterplot of the differenced / transformed data
f2, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 5))
f2.tight_layout()

lag_plot(transformed_data['noise'], ax=ax4)
ax4.set_title('noise level on Twiter');

lag_plot(transformed_data['iv_30'], ax=ax5)
ax5.set_title('iv_30(intrapolated)');

plt.show()

# normalizing a timeseries to confine it within 1 - for better visualization of data

time_x = transformed_data.index

new_data = data.copy(deep=True)
normal_noise_y = transformed_data['noise']
normal_iv30_y = transformed_data['iv_30']

normal_noise_y = (normal_noise_y - normal_noise_y.min()) / (normal_noise_y.max() - normal_noise_y.min())
normal_iv30_y = (normal_iv30_y - normal_iv30_y.min()) / (normal_iv30_y.max() - normal_iv30_y.min())

plt.plot(time_x, normal_noise_y, label="normalized " + ticker + ' change in noise timeseries')
plt.plot(time_x, normal_iv30_y, label="normalized " + ticker + ' change in IV30(introapolated) timeseries')


plt.tight_layout()
plt.legend()
plt.show()







# obtain the lag P for VAR
raw_data = raw_data.dropna()
model = VAR(raw_data) #recall that rawData is w/o difference operation
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

# depending on lowest value of AIC


model = VAR(transformed_data)
model_fitted = model.fit(9)
#get the lag order
lag_order = model_fitted.k_ar
print(lag_order)



maxlag=lag_order
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):

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


raw_data_result = grangers_causation_matrix(raw_data, raw_data.columns)
print(raw_data_result)

data_result = grangers_causation_matrix(transformed_data, transformed_data.columns)
print(data_result)

### Granger causality test tests the null hypotehsis that the coefficient of past values in the regression equation is zero.
### tha past values of timeseries x does no cause the other timeseries y
### such that if the p value obained is less than the significance level of 0.05, you can reject the null hypothesis.

# data_result.to_csv(f"{ticker}_granger_result.csv")




### now that we have some extent of granger causality, we can train the model to now try and predict future values.
mod = VAR(transformed_data)
res = mod.fit(maxlags=lag_order, ic='aic')
# print(res.summary())

lag_order = res.k_ar
print(lag_order)

# durbin watson test for autocorrelation, test statistic values of 1.5 to 2.5 ar considered normal.
# 2 is no autocorrelation, 0 is abs positive autocorrelation and 4 is negative autocorrelation.
out = durbin_watson(res.resid)

for col, val in zip(transformed_data.columns, out):
    print(col, ':', round(val, 2))



# Input data for forecasting
input_data = transformed_data.values[-lag_order:]
print(input_data.shape)

# forecasting
pred = res.forecast(y=input_data, steps=n_obs)

pred = (pd.DataFrame(pred, index=X_test.index, columns=X_test.columns + '_pred'))
# print(pred.shape)





# inverting transformation
def invert_transformation(X_train, pred):

    forecast = pred.copy()
    columns = X_train.columns
    for col in columns:
        forecast[str(col)+'_pred'] = X_train[col].iloc[-1] + forecast[str(col)+'_pred'].cumsum()
    return forecast

output = invert_transformation(X_train, pred)




#combining predicted and real data set
combine = pd.concat([output['iv_30_pred'], X_test['iv_30']], axis=1)
combine['accuracy'] = round(combine.apply(lambda row: row.iv_30_pred /row.iv_30 *100, axis = 1),2)
combine['accuracy'] = pd.Series(["{0:.2f}%".format(val) for val in combine['accuracy']],index = combine.index)
combine = combine.round(decimals=2)
combine = combine.reset_index()
combine = combine.sort_values(by='time', ascending=False)

print(combine)
combine.to_csv(f'{ticker}prediction_result.csv')




#Forecast bias
forecast_errors = [combine['iv_30'][i]- combine['iv_30_pred'][i] for i in range(len(combine['iv_30']))]
bias = sum(forecast_errors) * 1.0/len(combine['iv_30'])
print('Bias: %f' % bias)
print('MAPE:', mean_absolute_percentage_error(combine['iv_30'].values, combine['iv_30_pred'].values))
print('MAE:', mean_absolute_error(combine['iv_30'].values, combine['iv_30_pred'].values))
print('MSE:', mean_squared_error(combine['iv_30'].values, combine['iv_30_pred'].values))
print('RMSE:', sqrt(mean_squared_error(combine['iv_30'].values, combine['iv_30_pred'].values)))












