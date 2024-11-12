


import os
import pandas as pd

from src.utils.paths import get_results_path

N = 400
years = 14
k_max = 20
k_min = 0


results_df = pd.read_csv(os.path.join(get_results_path(),f"summary_stats_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}.csv"),index_col=0)

#results_df.to_latex()

#print(results_df.style.format(precision=2).hide(axis="index").to_latex())
print(results_df.style.format(precision=2).to_latex())