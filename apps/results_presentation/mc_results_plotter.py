
"""
code for 3D results_presentation of the results
"""



# importing packages
# ------------------
import numpy as np
import pandas as pd

import src.graphs.monte_carlo_graphs as rg




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



"""
loading data

"""
# # %%
# data_iid = pd.read_csv("results_iid_T_large.csv",index_col=None)
# # %%
# iid_fig = rg.setting_plot_T_growing(data_iid,"i.i.d. data")

# # %%
# iid_fig.savefig("iid_T_growing.eps",bbox_inches='tight')



data_Ncorr = pd.read_csv("results_Ncorr_T_large.csv",index_col=None)

Ncorr_T_large_fig = rg.setting_plot_T_growing(data_Ncorr,"Cross-sectional correlations")

Ncorr_T_large_fig.savefig("Ncorr_T_growing.eps",bbox_inches='tight')


# data_Tcorr = pd.read_csv("results_Tcorr_T_large.csv",index_col=None)

# Tcorr_T_large_fig = rg.setting_plot_T_growing(data_Tcorr,"Temporal correlations")

# Tcorr_T_large_fig.savefig("Tcorr_T_growing.eps",bbox_inches='tight')


data_NTcorr = pd.read_csv("results_NTcorr_T_large.csv",index_col=None)

NTcorr_T_large_fig = rg.setting_plot_T_growing(data_NTcorr,"Cross-sectional and temporal correlations")

NTcorr_T_large_fig.savefig("NTcorr_T_growing.eps",bbox_inches='tight')





""""
looking at increasing N

"""


# data_iid = pd.read_csv("results_iid_N_large.csv",index_col=None)

# iid_fig = rg.setting_plot_N_growing(data_iid,"i.i.d. data")

# iid_fig.savefig("iid_N_growing.eps",bbox_inches='tight')



data_Ncorr = pd.read_csv("results_Ncorr_N_large.csv",index_col=None)

Ncorr_N_large_fig = rg.setting_plot_N_growing(data_Ncorr,"Cross-sectional correlations")

Ncorr_N_large_fig.savefig("Ncorr_N_growing.eps",bbox_inches='tight')



# data_Tcorr = pd.read_csv("results_Tcorr_N_large.csv",index_col=None)

# Tcorr_N_large_fig = rg.setting_plot_N_growing(data_Tcorr,"Temporal correlations")

# Tcorr_N_large_fig.savefig("Tcorr_N_growing.eps",bbox_inches='tight')



data_NTcorr = pd.read_csv("results_NTcorr_N_large.csv",index_col=None)

NTcorr_N_large_fig = rg.setting_plot_N_growing(data_NTcorr,"Cross-sectional and temporal correlations")

NTcorr_N_large_fig.savefig("NTcorr_N_growing.eps",bbox_inches='tight')



""""
looking at varying beta across Ncorr

"""


data_beta_var = pd.read_csv("results_Ncorr_beta_var.csv",index_col=None)

beta_var_fig = rg.setting_plot_beta_var(data_beta_var,"varying beta")

beta_var_fig.savefig("beta_var.eps",bbox_inches='tight')



""""
looking at varying rho across Tcorr

"""


data_rho_var = pd.read_csv("results_Tcorr_rho_var.csv",index_col=None)



rho_var_fig = rg.setting_plot_rho_var(data_rho_var,"varying rho")

rho_var_fig.savefig("rho_var.eps",bbox_inches='tight')







