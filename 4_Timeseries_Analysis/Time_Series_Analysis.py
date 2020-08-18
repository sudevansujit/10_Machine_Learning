# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 07:47:32 2020

@author: Sujit

Time Series Analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('fivethirtyeight')

df = pd.read_csv('international-airline-passengers.csv', 
                 header = None, 
                 names = ['date', 'pass'], 
                 parse_dates = ['date'], 
                 index_col = ['date'])
df.head()

# Stastical Description
df.describe()

### Plotting

# Plotting
df.plot( figsize = (12, 4))
plt.show()

### Boxplot

# Boxplot
plt.figure(figsize = (12, 4))
sns.boxplot(y = df['pass'], x = df.index.year)
plt.show()

### Additive Decomposition

# Additive Decomposition
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
rcParams['figure.figsize'] = 18, 8
decomposition = seasonal_decompose(df['pass'], model='additive') # additive seasonal index
fig = decomposition.plot()
plt.show()

### Multiplicative Decomposition

# Multiplicative Decomposition
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
rcParams['figure.figsize'] = 18, 8
decomposition = seasonal_decompose(df['pass'], model = 'multiplicative')
fig = decomposition.plot()
plt.show()

### Stationarity Test

#### ADF Test

# Stationarity Test
from statsmodels.tsa.stattools import adfuller, kpss
adf_test = adfuller(df['pass'], )
print("p value ADF Test", adf_test[1])
"p value > 0.05 so fail to reject null hypothesis, so series is not stationary"

#### KPSS Test

kpss_test = kpss(df['pass'])
print("p value KPSS Test", kpss_test[1])
'p value < 0.05 so we reject null hypothesis which states series is stationary'

# Box cox transformation
from scipy.stats import boxcox
df['box_pass'] = boxcox(df['pass'], lmbda = 0)
df.head()

plt.plot(df['box_pass'])
plt.title("Box Transform Plot")
plt.show()

# 1st Differencing 
df['1st_box_pass'] = df['box_pass'].diff()


# 1st Testing
adf_test = adfuller(df['1st_box_pass'].dropna())
print("p Value ADF test", adf_test[1])
df.head()

# 2nd differencing
df['2nd_box_pass'] = df['box_pass'].diff().diff()


# 2nd Testing
adf_test = adfuller(df['2nd_box_pass'].dropna())
print("p Value ADF test",adf_test[1])
df.head()

### ACF and PACF Plots

# ACF and PACF Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize = (12, 4))
plot_acf(df['box_pass'], ax = plt.gca(), lags = 30)
plt.show()

# 1st ACF Plot
plt.figure(figsize = (12, 4))
plot_acf(df['1st_box_pass'].dropna(), ax = plt.gca(), lags = 30)
plt.show()

plt.figure(figsize = (12, 4))
plot_pacf(df['1st_box_pass'].dropna(), ax = plt.gca(), lags = 30)
plt.show()

# 2nd ACF Plot
plt.figure(figsize = (12, 4))
plot_acf(df['2nd_box_pass'].dropna(), ax = plt.gca(), lags = 30)
plt.show()

plt.figure(figsize = (12, 4))
plot_pacf(df['2nd_box_pass'].dropna(), ax = plt.gca(), lags = 30)
plt.show()

### Plotting Rolling Statistics

#Plotting Rolling Statistics
plt.style.use('fivethirtyeight')
rolmean = df['pass'].rolling(window  = 12).mean()
rolstd = df['pass'].rolling(window  = 12).std()
plt.plot(df['pass'])
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolstd, c = 'r', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

"""Making Time Series Stationary
There are 2 major reasons behind non-stationaruty of a TS:

Trend – varying mean over time. For eg, in this case we saw that on average,
 the number of passengers was growing over time.
 
Seasonality – variations at specific time-frames. eg people might have a 
tendency to buy cars in a particular month because of pay increment or 
festivals."""

df['pass_ma'] = rolmean
df.head(14)

box_rolmean = df['box_pass'].rolling(window = 12).mean()
df['box_pass_ma'] = box_rolmean
df.head(14)

df['ts_box_diff'] = df['box_pass'] - df['box_pass_ma']
df.head(14)

def test_stationarity(timeseries):
    timeseries = timeseries.dropna()
    plt.style.use('fivethirtyeight')
    avg_box_rolmean = timeseries.rolling(window  = 12).mean()
    std_box_rolstd = timeseries.rolling(window  = 12).std()
    plt.plot(timeseries)
    plt.plot(avg_box_rolmean, label='Rolling Mean')
    plt.plot(std_box_rolstd, c = 'r', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#Plotting Rolling Statistics and Testing ADF
test_stationarity(df['ts_box_diff'])

"""Exponentially weighted moving average:
To overcome the problem of choosing a defined window in moving average, we can 
use exponential weighted moving average
We take a ‘weighted moving average’ where more recent values are given a higher 
weight.

There can be many technique for assigning weights. A popular one is exponentially
weighted moving average where weights are assigned to all the previous values
with a decay factor."""
 
# Exponential Moving Average
expwighted_avg = df['box_pass'].ewm( halflife=12).mean()
plt.plot(df['box_pass'])
plt.plot(expwighted_avg, color='red') 
plt.show()

df['ewm_box_pass'] = expwighted_avg
df.head()

df.tail()

df['ewm_box_pass_diff']  = df['box_pass'] - df['ewm_box_pass']
df.head()

#Plotting Rolling Statistics and Testing ADF
test_stationarity(df['ewm_box_pass_diff'])

#Plotting Rolling Statistics
plt.style.use('fivethirtyeight')
ewm_avg_box_rolmean = df['ewm_box_pass_diff'].ewm( halflife=12).mean()
ewm_std_box_rolstd = df['ewm_box_pass_diff'].ewm( halflife=12).std()
plt.plot(df['ewm_box_pass_diff'])
plt.plot(ewm_avg_box_rolmean, label='Rolling Mean')
plt.plot(ewm_std_box_rolstd, c = 'r', label='Rolling Std')
plt.legend(loc='best')
plt.title('Exponential Rolling Mean & Standard Deviation')
plt.show()

"""Differencing
In this technique, we take the difference of the observation at a particular 
instant with that at the previous instant.
First order differencing in Pandas"""


df['box_diff'] = df['box_pass'] - df['box_pass'].shift()
df.head()

plt.plot(df['box_diff'].dropna())

#Plotting Rolling Statistics and Testing ADF
test_stationarity(df['box_diff'])

"""Decomposition
In this approach, both trend and seasonality are modeled separately and the 
remaining part of the series is returned.
"""

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df[['box_pass']]) # DataFrame should be passed inside

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df['box_pass'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

df['resid'] = residual
df.head(14)

#Plotting Rolling Statistics and Testing ADF
test_stationarity(df['resid'])

# Thanks

