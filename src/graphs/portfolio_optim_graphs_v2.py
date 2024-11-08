

import numpy as np
import matplotlib.pyplot as plt




def optim_subplots_v2(data,title,ylim,stepsize):

    fig, axs = plt.subplots(4, 2, figsize=(6.4 * 1.5, 4.8), layout="constrained")

    #fig.suptitle(title, fontsize=15) no suptitle - export to latex

    axs[0, 0].plot(data["UTCTIME"],data["BN_IC1"])
    axs[0, 0].set_ylabel("BN_IC1")
    axs[0, 0].set_ylim(ylim)
    axs[0, 0].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[0, 1].plot(data["UTCTIME"],data["BN_BIC3"])
    axs[0, 1].set_ylabel("BN_BIC3")
    axs[0, 1].set_ylim(ylim)
    axs[0, 1].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[1, 0].plot(data["UTCTIME"],data["ON_ED"])
    axs[1, 0].set_ylabel("ON_ED")
    axs[1, 0].set_ylim(ylim)
    axs[1, 0].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[1, 1].plot(data["UTCTIME"],data["AH_ER"])
    axs[1, 1].set_ylabel("AH_ER")
    axs[1, 1].set_ylim(ylim)
    axs[1, 1].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[2, 0].plot(data["UTCTIME"],data["AH_GR"])
    axs[2, 0].set_ylabel("AH_GR")
    axs[2, 0].set_ylim(ylim)
    axs[2, 0].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[2, 1].plot(data["UTCTIME"],data["WC_TKCV"])
    axs[2, 1].set_ylabel("WC_TKCV")
    axs[2, 1].set_ylim(ylim)
    axs[2, 1].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[3, 0].plot(data["UTCTIME"],data["dummy"])
    axs[3, 0].set_ylabel("K = k_max")
    axs[3, 0].set_ylim(ylim)
    axs[3, 0].set_yticks(np.arange(ylim[0], ylim[1], step=stepsize))

    axs[3, 1].set_axis_off()

    return fig







def optim_subplots_triple_v2(data1, data2, data3, title):

    # finding values for the y axis range, to make it same for all plots
    min_1 = data1.iloc[:,1:].to_numpy().min() # first column is time
    min_2 = data3.iloc[:,1:].to_numpy().min() # first column is time
    min_d = min(min_1, min_2) - 0.5

    max_1 = data1.iloc[:,1:].to_numpy().max()
    max_2 = data3.iloc[:,1:].to_numpy().max()
    max_d = max(max_1, max_2) + 0.5

    fig, axs = plt.subplots(4, 2, figsize=(6.4 * 1.5, 4.8), layout="constrained")

    #fig.suptitle(title, fontsize=15) no suptitle - export to latex

    axs[0, 0].plot(data1["UTCTIME"],data1["BN_IC1"],color = "tab:blue")
    axs[0, 0].plot(data1["UTCTIME"],data2["BN_IC1"], color="tab:orange")
    axs[0, 0].plot(data1["UTCTIME"],data3["BN_IC1"],color = "tab:blue")
    axs[0, 0].set_ylabel("BN_IC1")
    axs[0, 0].set_ylim([min_d,max_d])

    axs[0, 1].plot(data1["UTCTIME"],data1["BN_BIC3"],color = "tab:blue")
    axs[0, 1].plot(data1["UTCTIME"],data2["BN_BIC3"], color="tab:orange")
    axs[0, 1].plot(data1["UTCTIME"],data3["BN_BIC3"],color = "tab:blue")
    axs[0, 1].set_ylabel("BN_BIC3")
    axs[0, 1].set_ylim([min_d, max_d])

    axs[1, 0].plot(data1["UTCTIME"],data1["ON_ED"],color = "tab:blue")
    axs[1, 0].plot(data1["UTCTIME"],data2["ON_ED"], color="tab:orange")
    axs[1, 0].plot(data1["UTCTIME"],data3["ON_ED"],color = "tab:blue")
    axs[1, 0].set_ylabel("ON_ED")
    axs[1, 0].set_ylim([min_d, max_d])

    axs[1, 1].plot(data1["UTCTIME"],data1["AH_ER"], color="tab:blue")
    axs[1, 1].plot(data1["UTCTIME"],data2["AH_ER"], color="tab:orange")
    axs[1, 1].plot(data1["UTCTIME"],data3["AH_ER"], color="tab:blue")
    axs[1, 1].set_ylabel("AH_ER")
    axs[1, 1].set_ylim([min_d, max_d])

    axs[2, 0].plot(data1["UTCTIME"],data1["AH_GR"],color = "tab:blue")
    axs[2, 0].plot(data1["UTCTIME"],data2["AH_GR"], color="tab:orange")
    axs[2, 0].plot(data1["UTCTIME"],data3["AH_GR"],color = "tab:blue")
    axs[2, 0].set_ylabel("AH_GR")
    axs[2, 0].set_ylim([min_d, max_d])

    axs[2, 1].plot(data1["UTCTIME"],data1["WC_TKCV"],color = "tab:blue")
    axs[2, 1].plot(data1["UTCTIME"],data2["WC_TKCV"],color = "tab:orange")
    axs[2, 1].plot(data1["UTCTIME"],data3["WC_TKCV"],color = "tab:blue")
    axs[2, 1].set_ylabel("WC_TKCV")
    axs[2, 1].set_ylim([min_d, max_d])

    axs[3, 0].plot(data1["UTCTIME"],data1["dummy"],color = "tab:blue")
    axs[3, 0].plot(data1["UTCTIME"],data2["dummy"],color = "tab:orange")
    axs[3, 0].plot(data1["UTCTIME"],data3["dummy"],color = "tab:blue")
    axs[3, 0].set_ylabel("K = k_max")
    axs[3, 0].set_ylim([min_d, max_d])

    axs[3, 1].set_axis_off()


    return fig


















































def optim_subplots_double(data,data2,title):

    # finding values for the y axis range, to make it same for all plots
    min_1 = data.to_numpy().min()
    min_2 = data2.to_numpy().min()
    min_d = min(min_1, min_2) - 0.5

    max_1 = data.to_numpy().max()
    max_2 = data2.to_numpy().max()
    max_d = max(max_1, max_2) + 0.5

    fig, axs = plt.subplots(3, 2, figsize=(6.4 * 1.5, 4.8), layout="constrained")

    fig.suptitle(title, fontsize=15)

    axs[0, 0].plot(data["ER"])
    axs[0, 0].plot(data2["ER"])
    axs[0, 0].set_ylabel("ER")
    axs[0, 0].set_ylim([min_d,max_d])

    axs[0, 1].plot(data["GR"])
    axs[0, 1].plot(data2["GR"])
    axs[0, 1].set_ylabel("GR")
    axs[0, 1].set_ylim([min_d, max_d])

    axs[1, 0].plot(data["ED"])
    axs[1, 0].plot(data2["ED"])
    axs[1, 0].set_ylabel("ED")
    axs[1, 0].set_ylim([min_d, max_d])

    axs[1, 1].axis('off')

    axs[2, 0].plot(data["IC1"])
    axs[2, 0].plot(data2["IC1"])
    axs[2, 0].set_ylabel("IC1")
    axs[2, 0].set_ylim([min_d, max_d])

    axs[2, 1].plot(data["BIC3"])
    axs[2, 1].plot(data2["BIC3"])
    axs[2, 1].set_ylabel("BIC3")
    axs[2, 1].set_ylim([min_d, max_d])

    return fig












def policy_comparison_plot(date, policy_type, time_vector, zeros_vector,
                           price, charge_policy, charge_nocontrol,
                           energy_policy, energy_nocontrol,
                           cost_policy, cost_nocontrol):
    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 1.5, 4.8), layout="constrained")
    # fig, axs = plt.subplots(3,2,layout="constrained")

    fig.suptitle(date + " " + policy_type + " vs. baseline", fontsize=15)

    hrz = 9  # len(price)

    price = price * 1000
    time_vector = time_vector + 1
    time_vector = time_vector / 4

    axs[0, 0].plot(time_vector, price, "r")
    axs[0, 0].plot(time_vector, zeros_vector, "k--")
    axs[0, 0].set_ylabel("energy price [€/MWh]")
    axs[0, 0].set_xlabel("time [hour]")
    axs[0, 0].set_xticks(np.arange(1, hrz, step=1))

    axs[1, 0].plot(time_vector, charge_policy)
    axs[1, 0].plot(time_vector, charge_nocontrol)
    axs[1, 0].plot(time_vector, zeros_vector, "k--")
    axs[1, 0].set_ylabel("charge [kW]")
    axs[1, 0].set_xlabel("time [hour]")
    axs[1, 0].set_xticks(np.arange(1, hrz, step=1))

    axs[0, 1].plot(time_vector, energy_policy, label="optimal control")
    axs[0, 1].plot(time_vector, energy_nocontrol, label="baseline")
    axs[0, 1].plot(time_vector, zeros_vector, "k--")
    axs[0, 1].set_ylabel("energy in battery [kWh]")
    axs[0, 1].set_xlabel("time [hour]")
    axs[0, 1].set_xticks(np.arange(1, hrz, step=1))
    axs[0, 1].legend(bbox_to_anchor=(1.00, 0.4))

    axs[1, 1].plot(time_vector, cost_policy.cumsum())
    axs[1, 1].plot(time_vector, cost_nocontrol.cumsum())
    axs[1, 1].plot(time_vector, zeros_vector, "k--")
    axs[1, 1].set_ylabel("cumulative cost [€]")
    axs[1, 1].set_xlabel("time [hour]")
    axs[1, 1].set_xticks(np.arange(1, hrz, step=1))

    # profit = cost_nocontrol - cost_policy
    #
    # axs[1, 1].plot(time_vector, profit.cumsum(),"k")
    # axs[1, 1].plot(time_vector, zeros_vector, "k--")
    # axs[1, 1].set_ylabel("cumulative profit [€]")
    # axs[1, 1].set_xlabel("time [0.25h]")
    # axs[1, 1].set_xticks(np.arange(0, hrz, step=10))

    # plt.show()

    return fig

