#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# In[15]:


# Load dataset
df = pd.read_csv('/Users/admin/Downloads/Money price time-series.csv', parse_dates=['DATE'],index_col='DATE')
df.rename(columns={'IPG2211A2N': 'Money_price'}, inplace=True)


# In[16]:


df


# In[5]:


# Plot raw data
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Money_price'], label='Money_price')
plt.title('Money_price over time')
plt.xlabel('Date')
plt.ylabel('Production')
plt.grid(True)
plt.legend()
plt.show()


# In[17]:


# Feature Engineering: Lag, Rolling, Expanding
df['Lag_1'] = df['Money_price'].shift(1)
df['Rolling_Mean_12'] = df['Money_price'].rolling(window=12).mean()
df['Expanding_Mean'] = df['Money_price'].expanding().mean()


# In[18]:


df


# In[20]:


# Check for the stationary usinf ADF Test
def adf_test(df):
    result = adfuller(df)
    print("ADF Statistic:", result[0])
    print("p-values:", result[1])
    if result[1] <= 0.05:
        print("Data is Stationary")
    else:
        print("Data is not Stationary")


# In[21]:


adf_test(df['Money_price'])


# In[11]:


# Differencing to remove trend and seasonality if there is seasonality
df['Money_price_diff'] = df['Money_price'].diff().dropna()
df['Money_price_diff'] = df['Money_price'].diff(12).dropna()


# In[23]:


# plot ACF and PACF to identify parameters
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Money_price'].dropna(), ax=axes[0])
plot_pacf(df['Money_price'].dropna(), ax=axes[1])
plt.show()


# In[25]:


# Decomposition
decomposition = seasonal_decompose(df['Money_price'], model='additive', period=12)
decomposition.plot()
plt.tight_layout()
plt.show()


# In[27]:


# Differencing
# First order
df['Diff_1'] = df['Money_price'].diff()
# Seasonal differencing
df['Seasonal_Diff_12'] = df['Money_price'].diff(12)


# In[28]:


# Train-test split
train_size = int(len(df) * 0.8)
train, test = df['Money_price'][0:train_size], df['Money_price'][train_size:]


# In[41]:


#walk forward validation
history = train.tolist()
predictions = []
for t in test:
    model = AutoReg(history, lags = 7)
    model_fit = model.fit()
    
    y_pred = model_fit.predict(start=len(history), end=len(history))[0]
    predictions.append(y_pred)
    
    history.append(t)


# In[42]:


rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Walk-Forward Validation RMSE: {rmse:.4f}')


# In[44]:


# plot actual vs predicted values
plt.figure(figsize=(10,5))
plt.plot(test, label='Actual Production', marker='o')
plt.plot(predictions, label='Predicted Production', marker='x', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('AR Model - Walk Forward Validation')
plt.legend()
plt.show()


# In[45]:


# Random walk
random_walk = np.random.normal(0, 1, len(df)).cumsum()
plt.plot(random_walk)
plt.title('Random Walk')
plt.show()


# In[12]:


# AR model
model_ar = AutoReg(train, lags=12).fit()
predictions = model_ar.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)


# In[13]:


# Forecast plot
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predicted', linestyle='--')
plt.legend()
plt.title('AR Model Forecast')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




