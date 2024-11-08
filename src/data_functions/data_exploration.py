
import os
import time
import pandas as pd
import numpy as np

from src.utils.paths import get_data_path

N = 400
years = 14

t0_optim = time.time()


# importing data
daily_returns_df = pd.read_csv(os.path.join(get_data_path(),f"SP500_{N}_daily_returns_years_{years}.csv"))

daily_returns_df = daily_returns_df.iloc[:,1:] # removing time column

# variance statistics
# -------------------
var_vector = daily_returns_df.std()

print(f"N, years: ({N}, {years}) minimum std: {var_vector.min()}")
print(f"N, years: ({N}, {years}) maximum std: {var_vector.max()}")


# standard deviation statistic
# ----------------------------
pairwise_correlations = daily_returns_df.corr()
pairwise_correlations_np = pairwise_correlations.to_numpy()

np.fill_diagonal(pairwise_correlations_np, 0)

print(f"N, years: ({N}, {years}) minimum pairwise correlation: {np.min(pairwise_correlations_np,axis=None)}")

print(f"N, years: ({N}, {years}) maximum pairwise correlation: {np.max(pairwise_correlations_np,axis=None)}")


cutoff = 0.85

selection = pairwise_correlations[(pairwise_correlations >= cutoff)]
selection = selection.dropna(axis = "columns",how = "all")
selection = selection.dropna(axis = "rows",how = "all")



rows, cols = selection.shape


for i in range(rows):
    for j in range(cols):
        if j > i:
            selection.iloc[i,j] = pd.NA






dict_list = []

for i in range(rows):
    for j in range(cols):
        if not np.isnan(selection.iloc[i,j]):
            result_dict = {
                "first": selection.index[i],
                "second": selection.columns[j],
                "cross correlation": selection.iloc[i,j]
            }
            dict_list.append(result_dict)

result_df = pd.DataFrame(dict_list)


corr_list = selection.columns.to_list()



result_df.to_csv(os.path.join(get_data_path(), f"individual_pairwise_correlations_{N}_years_{years}_cutoff_{cutoff}.csv"),index=False)

selection.to_csv(os.path.join(get_data_path(), f"sig_pairwise_correlations_{N}_years_{years}_cutoff_{cutoff}.csv"))

#
#
#
#
#
#
#
#
# data_daily_returns = data_daily_returns.loc[~(data_daily_returns[dr_column_names_stocks] == 0.0).all(axis=1)]
#
# pairwise_correlations_09 = pairwise_correlations[pairwise_correlations >= 0.9]
#
# pairwise_correlations_09.to_csv(os.path.join(get_data_path(), f"pairwise_correlations_0.9_{N}_years_{years}.csv"))
#
#
print("hello there xddddd")



