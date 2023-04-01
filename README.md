# Paris-G2-Bousselmi-Martin-Siegelin

MACZYNSKI Camille / MARTIN Lucie / MENUET Guillaume / SIEGELIN Adrien

#Introduction: We decided to realize a python code to help us to compare the 3 luxury stocks with the index the CAC40
#We chose various metrics to assess this strong french sector
#We also decided to realize various covariance matrix
#Furthermore, we ploted various graphics such as distribution frequencies and stocks comparison
#Import necessary libraries

import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime

# Define a function to retrieve stock data for a given ticker and date range

def get_stock_data(ticker, start_date, end_date):
  stock_data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
  return stock_data

# Define the list of tickers to retrieve data for, and the date range

tickers = ['KER.PA', 'MC.PA', 'RMS.PA', '^FCHI']
start_date = datetime(2018, 1, 1)
end_date = datetime(2022, 12, 31)

# Retrieve stock data for each ticker and store in a dictionary

stock_data = {}
for ticker in tickers:
  data = get_stock_data(ticker, start_date, end_date)
  stock_data[ticker] = data

# Import numpy for calculating metrics

import numpy as np

# Loop through the stock data and calculate metrics for each stock
for ticker, data in stock_data.items():
  
  # Calculate daily returns and Sharpe ratio
  
  daily_returns = data['Close'].pct_change()
  sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

  # Calculate compound annual growth rate (CAGR)
  
  cagr = ((data['Adj Close'][-1] / data['Adj Close'][0]) ** (1/(len(data)/252))) - 1

  # Print out the metrics for each stock
  
  print('Ticker:', ticker) 
  print('Mean:', data['Close'].mean()) 
  print('Max:', data['Close'].max()) 
  print('Min:', data['Close'].min()) 
  print('Median:', data['Close'].median()) 
  print('Standard Deviation:', data['Close'].std())
  print('Variance:', data['Close'].var())
  print('Sharpe Ratio:', sharpe_ratio)
  print('Annual Compounded Growth Rate (CAGR):', cagr)
  print()

import matplotlib.pyplot as plt

# Create a figure with 2 rows and 2 columns of subplots

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
for i, (ticker, data) in enumerate(stock_data.items()):
  row = i // 2
  col = i % 2
  ax = axes[row, col]
  ax.plot(data['Close'], label=ticker)
  ax.set_title(ticker)
  ax.legend()
plt.tight_layout()

# Create another figure with 2 rows and 2 columns of subplots

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
for i, (ticker, data) in enumerate(stock_data.items()):
  row = i // 2
  col = i % 2
  ax = axes[row, col]
  
  # Plot a histogram of the closing prices for the current stock on the selected subplot
  
  ax.hist(data['Close'], bins=50, density=True)
  ax.set_title(ticker)
plt.tight_layout()

# Download historical stock data from YFinance for the specified tickers and time period

data = yf.download(tickers, start="2018-01-01", end="2022-12-31") 

# Extract the adjusted closing prices from the downloaded data

adj_close = data["Adj Close"] 

# Create a line plot of the adjusted closing prices for all tickers on the same plot

adj_close.plot(figsize=(10, 6)) 
plt.legend(tickers) 
plt.title("Stock prices") 
plt.xlabel("Date") 
plt.ylabel("Price")
plt.show()

import statsmodels.api as sm
import pandas as pd

# OLS Regression between Kering and CAC40
# Selecting the stock data for Kering and CAC40

kering = stock_data['KER.PA']
cac40 = stock_data['^FCHI']

# Creating a dataframe with the closing prices of Kering and CAC40

df = pd.concat([kering['Close'], cac40['Close']], axis=1)
df.columns = ['KER.PA', '^FCHI']
df = df.dropna()
X = df['^FCHI']
y = df['KER.PA']

# Adding a constant term to the independent variable X for the regression model

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Printing the summary statistics of the regression model

print(model.summary())

# Creating a scatter plot of the data points and the regression line

fig, ax = plt.subplots()
ax.scatter(X.iloc[:,1], y)
ax.plot(X.iloc[:,1], model.predict(), color='red')
ax.set_xlabel('^FCHI')
ax.set_ylabel('KER.PA')
plt.show()

# OLS Regression between LVMH and CAC40
# Selecting the stock data for LVMH and CAC40

LVMH = stock_data['MC.PA']
cac40 = stock_data['^FCHI']

# Creating a dataframe with the closing prices of LVMH and CAC40

df = pd.concat([LVMH['Close'], cac40['Close']], axis=1)
df.columns = ['MC.PA', '^FCHI']
df = df.dropna()

# Assigning the independent variable X and the dependent variable y

X = df['^FCHI']
y = df['MC.PA']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Printing the summary statistics of the regression model

print(model.summary())

# Creating a scatter plot of the data points and the regression line

fig, ax = plt.subplots()
ax.scatter(X.iloc[:,1], y)
ax.plot(X.iloc[:,1], model.predict(), color='green')
ax.set_xlabel('^FCHI')
ax.set_ylabel('MC.PA')
plt.show()

# OLS Regression between Hermès and CAC40
# Selecting the stock data for Hermès and CAC40

Hermès = stock_data['RMS.PA']
cac40 = stock_data['^FCHI']

# Creating a dataframe with the closing prices of Hermès and CAC40

df = pd.concat([Hermès['Close'], cac40['Close']], axis=1)

# Naming the columns of the dataframe

df.columns = ['RMS.PA', '^FCHI']
df = df.dropna()
X = df['^FCHI']
y = df['RMS.PA']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Printing the summary statistics of the regression

print(model.summary())

# Creating a scatter plot of the data points and the regression line

fig, ax = plt.subplots()
ax.scatter(X.iloc[:,1], y)
ax.plot(X.iloc[:,1], model.predict(), color='orange')
ax.set_xlabel('^FCHI')
ax.set_ylabel('RMS.PA')
plt.show()
