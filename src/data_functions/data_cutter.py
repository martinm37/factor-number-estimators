

import os
import time
import datetime
import pandas as pd
import numpy as np
from numpy.random import randint


from src.data_functions.data_transform_functions import daily_returns_fun, correlations_list, max_correlation
from src.utils.paths import get_results_path,get_data_path

"""
hyperparameters
******************************
"""
# selecting time window

start_date_data_price = datetime.datetime(2009,12,31)
end_date_data_price = datetime.datetime(2024,1,6)

start_date_market_cap = start_date_data_price + datetime.timedelta(days=1) # we start one day later, as in the creation of retruns we lost one day
end_date_market_cap = end_date_data_price


# selecting the amount of stocks we want to have
stock_number = 400

"""
******************************
"""


# loading data
data_price = pd.read_csv(os.path.join(get_data_path(),"SP500_constituents_stock_prices.csv"))
data_market_cap = pd.read_csv(os.path.join(get_data_path(),"SP500_constituents_market_capitalization.csv"))

# time conversion
data_price["UTCTIME"] = pd.to_datetime(data_price["UTCTIME"], format="%d/%m/%Y")
data_market_cap["UTCTIME"] = pd.to_datetime(data_market_cap["UTCTIME"], format="%d/%m/%Y")

# cutting to the desired time length:
data_price = data_price[(data_price["UTCTIME"] >= start_date_data_price) & (data_price["UTCTIME"] < end_date_data_price)]
data_market_cap = data_market_cap[(data_market_cap["UTCTIME"] >= start_date_market_cap) & (data_market_cap["UTCTIME"] < end_date_market_cap)]

# dropping the NA values - stocks which are not avalibal=ble for the whole of the SELECTED!!!! period -> NOT the original time period !!!!
data_price = data_price.dropna(axis = "columns",how = "any") # drops all columns which contain at least one NA value
data_market_cap = data_market_cap.dropna(axis = "columns",how = "any") # drops all columns which contain at least one NA value



# calculating the daily returns
data_daily_returns = daily_returns_fun(data_price)

# removing the entries which are overly correlated
# ------------------------------------------------
cutoff = 0.85
corr_list = correlations_list(data_daily_returns, cutoff =cutoff)

corr_list_mc = [s + " - MARKET VALUE" for s in corr_list]

data_daily_returns = data_daily_returns.drop(columns = corr_list)
data_market_cap = data_market_cap.drop(columns = corr_list_mc)






"""
sorting the daily returns by the market capitalization
"""

dr_column_names = data_daily_returns.columns.values.tolist()

data_market_cap.columns = data_market_cap.columns.str.replace(" - MARKET VALUE","") # removing the unneccessary string

mc_column_names = data_market_cap.columns.values.tolist()

# check if the conversion was successful

assert mc_column_names == dr_column_names, "column names are not equal!!!"



# need to drop time column before sorting
data_market_cap = data_market_cap.drop("UTCTIME",axis=1)

sort_label = data_market_cap.index[0]
# we sort by the first day of our desired period

"""
here I will sort the data by the 1st Jan 2010, for the both datasets,
the 10 year one and the 14 year one, they have to contain the same stocks and this is more logical
"""

data_market_cap_sort = data_market_cap.sort_values(by = sort_label,axis = "columns",ascending = False)




data_market_cap_remove = data_market_cap_sort.iloc[:,stock_number:]


stocks_to_drop = data_market_cap_remove.columns.values.tolist()

"""
selecting daily returns
"""

data_daily_returns_selected = data_daily_returns.drop(columns = stocks_to_drop)


# corr check
maxcorr = max_correlation(data_daily_returns_selected)
assert maxcorr < cutoff, f"correlation is too high!!: {maxcorr}"

years = 14

data_daily_returns_selected.to_csv(os.path.join(get_data_path(),f"SP500_{stock_number}_daily_returns_years_{years}.csv"),index = False)

"""
cutting down to 10 years:
"""

start_date_final = datetime.datetime(2010,1,1)
end_date_final = datetime.datetime(2020,1,6)

data_daily_returns_selected = data_daily_returns_selected[(data_daily_returns_selected["UTCTIME"] >= start_date_final)
                                                          & (data_daily_returns_selected["UTCTIME"] < end_date_final)]


# corr check
maxcorr = max_correlation(data_daily_returns_selected)
assert maxcorr < cutoff, f"correlation is too high!!: {maxcorr}"

years = 10

data_daily_returns_selected.to_csv(os.path.join(get_data_path(),f"SP500_{stock_number}_daily_returns_years_{years}.csv"),index = False)






