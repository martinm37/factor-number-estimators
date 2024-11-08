


import os
import time
import datetime
import pandas as pd
import numpy as np
from numpy.random import randint

from src.utils.paths import get_results_path,get_data_path



stock_number = 150


# loading
data_price = pd.read_csv(os.path.join(get_data_path(),"SP500_constituents_stock_prices.csv"))
data_market_cap = pd.read_csv(os.path.join(get_data_path(),"SP500_constituents_market_capitalization.csv"))

# time conversion
data_price["UTCTIME"] = pd.to_datetime(data_price["UTCTIME"], format="%d/%m/%Y")
data_market_cap["UTCTIME"] = pd.to_datetime(data_market_cap["UTCTIME"], format="%d/%m/%Y")

# cutting to 10 years length - 2014 to 2024

start_date = datetime.datetime(2013,12,31)
end_date = datetime.datetime(2024,1,1)
data_price = data_price[(data_price["UTCTIME"] >= start_date) & (data_price["UTCTIME"] < end_date)]

start_date = datetime.datetime(2014,1,1)
end_date = datetime.datetime(2024,1,1)
data_market_cap = data_market_cap[(data_market_cap["UTCTIME"] >= start_date) & (data_market_cap["UTCTIME"] < end_date)]


# dropping the NA values
data_price = data_price.dropna(axis = "columns",how = "any") # drops all columns which contain at least one NA value
data_market_cap = data_market_cap.dropna(axis = "columns",how = "any") # drops all columns which contain at least one NA value


# we start in the last date of 2013. as we want to have daily return for the first day of 2014



# calculating daily returns - V2
data_daily_returns = data_price.iloc[1:,:].copy()


time_lenght = len(data_daily_returns)

column_names = data_daily_returns.columns.values.tolist()

#columns_names = data_daily_returns.columns
column_names.remove("UTCTIME")

# flushing
for column_name in column_names:
    data_daily_returns[column_name] = np.zeros(time_lenght) # flushing


# creating 2 price df

data_price_t0 = data_price.iloc[:-1,:].copy()
data_price_t0_np = data_price_t0.iloc[:,1:]
data_price_t0_np = data_price_t0_np.to_numpy()

data_price_t1 = data_price.iloc[1:,:].copy()
data_price_t1_np = data_price_t1.iloc[:,1:]
data_price_t1_np = data_price_t1_np.to_numpy()


daily_returns_np = (data_price_t1_np - data_price_t0_np) / data_price_t0_np * 100 # !!!!!!!!!

# storing in the pandas df
data_daily_returns.iloc[:,1:] = daily_returns_np

# removing the zero rows

dr_column_names = data_daily_returns.columns.values.tolist()

dr_column_names_stocks = dr_column_names.copy()

dr_column_names_stocks.remove("UTCTIME")


data_daily_returns = data_daily_returns.loc[~(data_daily_returns[dr_column_names_stocks] == 0.0).all(axis=1)]
# dropping all rows for which all stocks are zero on the given day





"""
1) need to redo the names of columns
2) need to create an ordering based on market cap
- for this i will use the simplest - order in the first day of 2014
3) keep just the daily returns of the 150 top stocks

"""

# amaziing somebody has already done it xddd

data_market_cap.columns = data_market_cap.columns.str.replace(" - MARKET VALUE","")

mc_column_names = data_market_cap.columns.values.tolist()

# check if the conversion was successful

hmm = mc_column_names == dr_column_names


# need to drop time columns before sorting

data_market_cap = data_market_cap.drop("UTCTIME",axis=1)

sort_label = data_market_cap.index[0]
# we sort by the first day

data_market_cap_sort = data_market_cap.sort_values(by = sort_label,axis = "columns",ascending = False)


data_market_cap_remove = data_market_cap_sort.iloc[:,stock_number:]


# selected_stocks = data_market_cap_sort.columns.values.tolist()
# #selected_stocks.remove("UTCTIME")
# all_stocks = data_daily_returns.columns.values.tolist()

# stocks_to_drop = list(set(all_stocks) - set(selected_stocks))

stocks_to_drop = data_market_cap_remove.columns.values.tolist()

"""
selecting daily returns
"""

data_daily_returns_selected = data_daily_returns.drop(columns = stocks_to_drop)


data_daily_returns_selected.to_csv(os.path.join(get_data_path(),"SP500_400_daily_returns.csv"),index = False)

