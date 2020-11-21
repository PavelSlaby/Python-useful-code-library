"""
LEARNING NOTES 
- based on following tutorial: Python for finance cookbook
"""

#%% Getting data from other sources

import quandl
df_quandl = quandl.get(dataset = 'WIKI/FB', start_date='2019-01-01', end_date='2020-07-10')

import pandas_datareader.data as dtr
dtr.DataReader('AAPL', 'yahoo')

#%% Converting prices to returns
import numpy as np
import yfinance as yf
import quandl
import pandas as pd

df = yf.download('AAPL', start= '2000-01-01', end = '2010-12-31', actions = 'inline')
df = df.rename(columns = {'Adj Close': 'adj_close'})

df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

i = quandl.get(dataset='RATEINF/CPI_USA', start_date='1999-12-01', end_date='2010-12-31')  # inflation in the USA

index1 = pd.date_range(start = '1999-12-31', end = '2010-12-31')
dates = pd.DataFrame(index = index1)

df= dates.join(df.adj_close, how = 'left', ).fillna(method = 'ffill')  # Ffill or forward-fill propagates the last observed non-null value forward until another non-null value is encountered.
df = df.asfreq('M')

i = i.rename(columns = {'Value' : 'cpi'})

df_merged = df.join(i, how = 'left')

df_merged['simple_rtn'] = df_merged.adj_close.pct_change() 
df_merged['inflation_rate'] = df_merged.cpi.pct_change() 

# adjusting returns for inflation 
df_merged['real_rtn'] = ( df_merged.simple_rtn + 1) / (df_merged["inflation_rate"] + 1)  - 1

pd.set_option('display.max_columns', None)

df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

df_vol = df['log_rtn'] 

def realized_vol(x):
    return np.sqrt(np.sum(x**2))

df_rv = df_vol.groupby(pd.Grouper(freq= 'M')).apply(realized_vol)
df_rv.columns
df_rv = pd.DataFrame(df_rv)
df_rv = df_rv.rename(columns = {'log_rtn':'vol'})
df_rv.vol = df_rv.vol * np.sqrt(12)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex = True)
ax[0].plot(df_vol)
ax[0].set(title = 'title 1', ylabel = 'no clue')
ax[1].plot(df_rv)
ax[1].set(title = 'title 2')

# alternatively:
df["log_rtn"].resample('M').mean()     # average monthly return
    

#%% Visualization

df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

df = df.loc[:, ['simple_rtn', 'log_rtn', 'adj_close']]

fig, ax = plt.subplots(3, 1, sharex = True)
df.adj_close.plot(ax = ax[0])
ax[0].set(title = 'aapl time series', ylabel = '$')
df["simple_rtn"].plot(ax = ax[1])
ax[1].set(title = 'simple returns', ylabel = 'returns')
df["log_rtn"].plot(ax = ax[2])
ax[2].set(title = 'log returns', ylabel = ' lgreturns')

#%% Identifiying outliers

df_rolling = df['simple_rtn'].rolling(window = 21).agg(['mean', 'std' ])
df_out = df.join(df_rolling)

def identify_outliers(datset, n =2):
    x = datset['simple_rtn']
    mu = datset['mean']
    sigma = datset['std']
    if (x > mu + n * sigma) or (x < mu - n * sigma ):
        return 1
    else:
        return 0

df_out['outliers'] = df_out.apply(identify_outliers, axis = 1)

df_out[df_out['outliers'] == 1]

df_out.loc[df_out['outliers'] == 1, ['simple_rtn']]
outliers = df_out.loc[df_out['outliers'] == 1, ['simple_rtn']]

fig, ax = plt.subplots()
ax.plot(df_out.index, df_out.simple_rtn, color = 'blue', label = 'Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color = 'red', label = 'outlier')
ax.set_title('AAPL stock returns')
ax.legend(loc = 'lower right')


#%% Investigating stylized facts of asset returns

import pandas_datareader.data as dtr
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt

df_spx  = dtr.DataReader('^GSPC', 'yahoo')
df_spx.head()
df_spx = df_spx.rename(columns = {'Adj Close':'adj_close'})

spx = df_spx
spx['log_rtn'] = np.log(df_spx.adj_close/df_spx.adj_close.shift(1))
spx = pd.DataFrame(spx['log_rtn'])

r_range = np.linspace(min(spx.log_rtn.dropna()), max(spx.log_rtn.dropna()), num = 1000) #Return evenly spaced numbers over a specified interval.
mu = spx.log_rtn.mean()
sigma = spx.log_rtn.std()
norm_pdf = stats.norm.pdf(r_range, mu, sigma)

f'N({mu:.2f}, {sigma**2:.4f})'

# 1] Non-Gaussian distribution of returns -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sns.distplot(spx["log_rtn"], kde=False, norm_hist = True, ax=ax[0])
ax[0].set_title('distribution of S&P 500 returns', fontsize = 16)
ax[0].plot(r_range, norm_pdf, color = 'g', lw = 2, label = f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].legend(loc = 'upper left')
qq = sm.qqplot(spx.log_rtn.values, line= 's', ax=ax[1], color = 'red')
ax[1].set_title('Q-Q plot', fontsize = 16)

'''
Non-Gaussian distribution of returns:
        Negative skewness (third moment): Large negative returns occur more frequently than large positive ones.
        Excess kurtosis (fourth moment) : Large (and small) returns occur more often than expected.

QQ Plot:
If the empirical distribution is
Normal, then the vast majority of the points will lie on the red line. However, we see that
this is not the case, as points on the left side of the plot are more negative (that is, lower
empirical quantiles are smaller) than expected in the case of the Gaussian distribution, as
indicated by the line. This means that the left tail of the returns distribution is heavier than
that of the Gaussian distribution. Analogical conclusions can be drawn about the right tail,
which is heavier than under normality.
'''

# 2] volatility clustering ----------------------------------------------------------------

spx.log_rtn.plot(title = 'S&P 500 log returns')  # volatility clustering

# 3] absence of autocorrelation in retunrs------------------------------------------------
lags = 50
signif_level = 0.05

#autocorrelation function ACF
acf = smt.graphics.plot_acf(spx.log_rtn.dropna(), lags = lags, alpha = signif_level)
pacf = smt.graphics.plot_pacf(spx.log_rtn.dropna(), lags = lags, alpha = signif_level)
# no significant autocorrelation in log return series
# ACF gives us values of auto-correlation with its lagged values
'''PACF:
Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals 
(which remains after removing the effects which are already explained by the earlier lag(s)) 
with the next lag value hence ‘partial’ and not ‘complete’ 
as we remove already found variations before we find the next correlation. 
So if there is any hidden information in the residual which can be modeled by the next lag, 
we might get a good correlation and we will keep that next lag as a feature while modeling. 
Remember while modeling we don’t want to keep too many features which are correlated 
as that can create multicollinearity issues. Hence we need to retain only the relevant features.
'''

# 4] Small and decreasing autocorrelation in squared/absolute returns---------------------------------------------

fig, ax = plt.subplots(2, 1, figsize = (10, 5))
smt.graphics.plot_acf(spx.log_rtn.dropna()**2, lags = lags, alpha = signif_level, ax = ax[0])
ax[0].set(title = 'autocorrelation plots', ylabel = 'squared returns')
smt.graphics.plot_acf((np.abs(spx.log_rtn.dropna())), lags = lags, alpha = signif_level, ax = ax[1])
ax[1].set(title = '', ylabel = 'absolure returns', xlabel = 'lag')

spx = spx.log_rtn.dropna()
spx = pd.DataFrame(spx)

# 5] Leverage effect -------------------------------------------------------

spx['mov_std_252'] = spx['log_rtn'].rolling(window = 252).std()
spx['mov_std_21'] = spx['log_rtn'].rolling(window = 21).std()


fig, ax = plt.subplots(3, 1,figsize = (11, 9))
fig.subplots_adjust(hspace=0.8)
df_spx.adj_close.plot(ax = ax[0])
ax[0].set(title = 'S&P 500 close', ylabel = 'level')
spx.log_rtn.plot(ax = ax[1])
ax[1].set(title = 'log returns', ylabel = 'log returns (%)')
spx['mov_std_252'].plot(ax = ax[2], color = 'r', label = 'moving volatility 252d')
spx['mov_std_21'].plot(ax = ax[2], color = 'g', label = 'moving volatility 21d')
ax[2].set(title = 'moving volatility', ylabel = 'volatility', xlabel = 'date')
ax[2].legend()

# This fact states that most measures of an asset's volatility are negatively correlated with its returns

# VIX
import pandas_datareader.data as pdr

data  = pdr.DataReader(['^GSPC', '^VIX'], data_source = 'yahoo' , start = '1985-01-01', end = '2020-07-20')
data.head()
data.info()
type(data)
data['Adj Close'].info()
type(data['Adj Close'])
type(data['Adj Close']['^VIX']) 

df = data['Adj Close'] 
df = df.rename(columns = {'^GSPC':'sp500', '^VIX':'vix'})
df['log_rtn'] = np.log(df.sp500 / df.sp500.shift(1))
df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
df.dropna(inplace = True)

corr = df.log_rtn.corr(df.vol_rtn)  # correlation

ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, line_kws={'color': 'red'})
ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr:.2f})', ylabel='VIX log returns', xlabel='S&P 500 log returns')

'''
both the negative slope of the regression line and a strong negative correlation between the two series confirm the existence of the leverage effect in the return series.
'''

#%% Technical Analysis

from datetime import datetime
import backtrader as bt

class SmaSignal(bt.Signal):
    params = (('period', 20), )
    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)

data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2018, 1, 1), todate=datetime(2018, 12, 31))

cerebro = bt.Cerebro(stdstats = False)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)

print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.plot(iplot=True, volume=False)


#%% Time Series modelling

import yfinance as yf
import pandas as pd

df = yf.download('GC=F', start= '2000-01-01', end = '2011-12-31')
df = df.rename(columns = {'Adj Close':'price'})
df = pd.DataFrame(df.price)

df.asfreq('M') # najde posledni den v mesici a vezme jeho hodnotu, pokud zadna hodnota neni (napr vikend) doplni NaN, proto lepsi pouzit nasledujici:
df = df.resample('M').last() 

roll = 12

df['rolling_mean'] = df['price'].rolling(roll).mean()
df['rolling_std'] = df['price'].rolling(roll).std()
df.plot(title = 'gold price')

# non-linear growth pattern can be observed -> use the multiplicative model

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df.price, model = 'multiplicative')
seasonal_decompose?

decomposition.plot()
dir(decomposition)
decomposition.resid.plot()
decomposition.trend.plot()
decomposition.seasonal.plot()

#%% Testing for stationarity in time series

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

df = yf.download('GC=F', start= '2000-01-01', end = '2011-12-31')
df = df.rename(columns = {'Adj Close':'price'})
df = pd.DataFrame(df.price)

def adf_test(x):
    stats = ['Test Statistic', 'p-value' , '# of lags', '# of observations']
    adf_test = smt.adfuller(x, autolag = 'AIC') # number of considered lags is automatically selected based on the Akaike Information Criterion (AIC).
    results = pd.Series(adf_test[0:4], index = stats)
    for key, values in adf_test[4].items():
        results[f'Critical Value({key})'] = values
    return results

adf_test(df.price)
# The null hypothesis of the ADF test states that the time series is not stationary. -> we can colnclude that the time series is truly non-stationary

lags = 40
signif_lvl = 0.05 

fig, ax = plt.subplots(2, 1, sharex = True, figsize = (10, 10))
smt.graphics.plot_acf(df.price, lags = lags, alpha = signif_lvl, ax = ax[0])
smt.graphics.plot_pacf(df.price, lags = lags, alpha = signif_lvl, ax = ax[1])

#%%Correcting for stationarity in time series
# - by differcing
# - by natural logarithm

import numpy as np
import pandas as pd
from datetime import date
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import quandl
import yfinance as yf
import matplotlib.pyplot as plt

cpi = quandl.get(dataset='RATEINF/CPI_USA', start_date='1999-12-01', end_date='2010-12-31')  # inflation in the USA

cpi['Inflation'] = cpi['Value']/cpi.iloc[0]['Value']

gold = yf.download('GC=F', start= '1999-12-01', end = '2010-12-31')
gold = gold['Adj Close']
gold = pd.DataFrame(gold)
gold.rename(columns = {'Adj Close': 'price'}, inplace = True)

dates = pd.date_range(start = gold.index[0], end = gold.index[-1] , freq = 'D')
type(dates)
dates = pd.DataFrame(index = dates)

dates1 = dates.join(gold, how = 'left')

gold = dates1.fillna(method = 'ffill')

cpi['Inflation'] = dates.join(cpi['Inflation'], how = 'left')
cpi['Inflation']  = cpi['Inflation'].fillna(method = 'ffill')

gold.iloc[0]
gold['adjusted'] = gold['price']/cpi.loc['2000-08-31']['Inflation']


fig, ax = plt.subplots()
gold['price'].plot(label = 'price')
gold['adjusted'].plot(label = 'adj price')
plt.legend()

#natural logarithm
window = 12

columns = ['prc_log', 'roll_mean_log', 'roll_std_log']
gold['prc_log'] = np.log(gold.price)
gold['roll_mean_log'] = gold.prc_log.rolling(window = window).mean()
gold['roll_std_log'] = gold.prc_log.rolling(window).std()

gold[columns].plot(title = 'Gold Price')
'''
From the preceding plot, we can see that the log transformation did its job, that is,
it made the exponential trend linear.
'''

# differencing
columns = ['prc_log_diff', 'roll_mean_log_diff', 'roll_std_log_diff']

gold.head()
gold['prc_log_diff'] = gold['prc_log'].diff(1)
gold['roll_mean_log_diff'] = gold.prc_log_diff.rolling(window = window).mean()
gold['roll_std_log_diff'] = gold.prc_log_diff.rolling(window).std()

gold[columns].plot(title = 'Gold Price (1st diff)')

'''
The transformed gold prices make the impression of being stationary – the series
oscillates around 0 with more or less constant variance. At least there is no visible
trend.
'''

#%% Modeling time series with ARIMA class models

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns

goog = yf.download('GOOG', start = '2012-01-01', end = '2018-12-31')
goog = goog.resample('W').last().rename(columns = {'Adj Close':'price'}).price

goog_diff = goog.diff().dropna()

fig, ax = plt.subplots(2, sharex = True, figsize = (10, 4))
goog.plot(title = 'google´s stock price', ax =ax[0])
goog_diff.plot(title = '1st diffs', ax =ax[1])

#tests for stationarity
fig, ax = plt.subplots(2, sharex = True, figsize = (10, 4))
smt.graphics.plot_acf(goog, lags = 20, alpha = 0.05, ax = ax[0], title = 'price')
smt.graphics.plot_pacf(goog_diff, lags = 20, alpha = 0.05, ax = ax[1], title = 'diffs')

def adf_test(x):
    stats = ['Test Statistic', 'p-value' , '# of lags', '# of observations']
    adf_test = adfuller(x, autolag = 'AIC') # number of considered lags is automatically selected based on the Akaike Information Criterion (AIC).
    results = pd.Series(adf_test[0:4], index = stats)
    for key, values in adf_test[4].items():
        results[f'Critical Value({key})'] = values
    return results

from statsmodels.tsa.stattools import adfuller

adf_test(goog)
adf_test(goog_diff)

# The results indicate that the differenced prices are stationary.

arima = ARIMA(goog, order = (2,1,1)).fit()
arima.summary()

#function for diagnostics

def arima_diag(resids, n_lags = 40):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    r = resids
    resids = (r - np.nanmean(r)) / np.nanstd(r)
    resids_nonmissing = resids[~(np.isnan(resids))]  ######
    
    sns.lineplot(x = np.arange(len(resids)), y = resids, ax = ax1)
    ax1.set_title('Standardized residuals')
    
    x_lim = (-1.96 * 2, 1.96 *2)
    r_range = np.linspace(x_lim[0], x_lim[1])
    norm_pdf = stats.norm.pdf(r_range)
    sns.distplot(resids_nonmissing, norm_hist = True, hist = True, kde = True, ax = ax2)
    
    ax2.plot(r_range, norm_pdf, 'g', lw= 2, label = 'N(0,1)')
    ax2.set_title('Distribution of standardized residuals')
    ax2.set_xlim(x_lim)
    ax2.legend()
    
    qq = sm.qqplot(resids_nonmissing, line = 's', ax = ax3)
    ax3.set_title('Q-Q plot')
    
    plot_acf(resids, ax = ax4, lags = n_lags, alpha = 0.05)
    ax4.set_title('ACF plot')
    
    return fig

arima_diag(arima.resid)

#%% Forecasting using ARIMA class models
df = yf.download('AAPL', start='2019-01-01', end='2019-03-31', adjusted=True) # data for validating our forecast, "the actual reality"
test = df.resample('W').last().rename(columns={'Adj Close': 'adj_close'}).adj_close
arima # takes values from the preceding section

n = len(test)
arima_f = arima.forecast(n)
arima_f = [pd.DataFrame(arima_f[0], columns = ['prediction']), pd.DataFrame(arima_f[2], columns = ['lower', 'upper'])]

arima_f = pd.concat(arima_f, axis = 1).set_index(test.index)

fig, ax = plt.subplots()
ax = sns.lineplot(data = test, label = 'Actual')
ax.plot(arima_f.prediction, label = 'ARIMA(2,1,1)')
ax.fill_between(arima_f.index, arima_f.upper, arima_f.lower, alpha = 0.3)
ax.set(title = 'stock price actual vs. prediction', xlabel = 'date', ylabel = 'price')
ax.legend(loc = 'lower left')

#%% CAPM

import pandas as pd
import yfinance as yf
import statsmodels.api as sm

stock = 'AMZN'
bench = '^GSPC'
start = '2014-01-01'
end = '2018-12-31'

df = yf.download([stock, bench], start = start, end = end)

x = df['Adj Close'].resample('M').last().pct_change().dropna().rename(columns= {'^GSPC':'SP500'})

#Beta:
x.cov()
cov = x.cov().iloc[0,1]
var = x.SP500.var()
beta = cov / var

Y = x.AMZN
X = sm.add_constant(x.SP500)

model = sm.OLS(Y, X).fit()
model.summary()

# beta is the const in the model so the 1.65

#%% three factor model
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf


import os
os.getcwd()
dir(os)

csv_path = r'F:\_Python\Tutorial replications\Cookbook\F-F_Research_Data_factors_CSV/F-F_Research_Data_Factors.CSV'

factor_df = pd.read_csv(csv_path, skiprows = 3)

factor_df.columns = ['date', 'mkt', 'smb', 'hml', 'rf']

string = ' Annual Factors: January-December '

indices = factor_df.iloc[:, 0] == string
indices[indices == True]

start_annual = factor_df[indices].index[0]

factor_df = factor_df[factor_df.index < start_annual]
factor_df = factor_df.dropna()

factor_df['date'] = pd.to_datetime(factor_df['date'], format='%Y%m').dt.strftime("%Y-%m")
factor_df = factor_df.set_index('date')

asset = 'FB'
start_date= '2013-12-31'
end_date = '2018-12-31'

factor_df = factor_df.loc[start_date:end_date]
factor_df.info()

# convert to numeric

pd.to_numeric(factor_df.mkt)

factor_df = factor_df.apply(pd.to_numeric)
factor_df.info()
factor_df = factor_df/100

fb_df = yf.download(asset, start = start_date, end = end_date, adjusted = True)
y = fb_df['Adj Close'].resample('M').last().pct_change().dropna()
y.index = y.index.strftime('%Y-%m')
y.name = 'rtn'

ff_data = factor_df.join(y)
ff_data['excess_rtn'] = ff_data['rtn'] - ff_data['rf'] 

# estimation of the model


model = smf.ols(formula = 'excess_rtn ~ mkt + smb + hml', data=ff_data).fit()
model.summary()

#%% Implementing the rolling three-factor model on a portfolio of assets

import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web
import numpy as np

stocks = ['AMZN', 'GOOG', 'AAPL', 'MSFT']
w = [0.25, 0.25, 0.25, 0.25]
start_date = '2009-12-31'
end_date = '2018-12-31'

df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date)[0]  # download from fama french

df_three_factor = df_three_factor/100
df_three_factor.index = df_three_factor.index.format()

stocks_df = yf.download(stocks, start_date, end_date, adjusted = True)

stocks_df = stocks_df['Adj Close'].resample('M').last().pct_change().dropna()

stocks_df.index = stocks_df.index.strftime('%Y-%m')  #format dates-------

np.matmul?  # Matrix product of two arrays

stocks_df['portfolio_rtn'] = np.matmul(stocks_df[stocks].values, w)

ff_data = pd.DataFrame(stocks_df.portfolio_rtn).join(df_three_factor)
ff_data['portf_ex_rtn'] = ff_data.portfolio_rtn - ff_data.RF
ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf', 'portf_ex_rtn']

# function for rolling model

def rolling_ff(input_data, formula, window):

    coefs = []
    for start in range(len(input_data) - window + 1):
        end = start + window
        
        ff_model = smf.ols(formula = formula, data = input_data[start:end]).fit()
        coefs.append(ff_model.params)
    
    
    coeffs_df = pd.DataFrame(coefs, index = input_data.index[window - 1:])
    
    return coeffs_df

results = rolling_ff(input_data = ff_data, formula = 'portf_ex_rtn ~ mkt + smb + hml', window = 60)

results.plot()

#%% four- and five-factor models in Python









