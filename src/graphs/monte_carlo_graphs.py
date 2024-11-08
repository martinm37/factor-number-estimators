
"""
code for 3D results_presentation of the results
"""



# importing packages
# ------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
try to rework the N and T part using dictionaries
- just N and T lists are inserted into the
parent function

and this functions takes the lengths of these lists,
and then iteratively slices the data and puts the
slices into dictionaries, which are then 
subsequently plotted

"""


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------





def estimator_subplot_T_growing(plot_axis,data,estimator):

    """
    function for plotting a subplot
    for a particular estimator    
    """

    # N part
    # **************************

    data_N = data[estimator].to_numpy().reshape(-1,1)
    N1 = data["$N$"].to_numpy().reshape(-1,1) 
    T1 = data["$T$"].to_numpy().reshape(-1,1) 
    data_N = np.concatenate((N1,T1,data_N),axis=1)

    data_N_slice1 = data_N[0:5]
    data_N_slice2 = data_N[5:10]
    data_N_slice3 = data_N[10:15]
    data_N_slice4 = data_N[15:20]
    data_N_slice5 = data_N[20:25]

    # T part
    # **************************


    data_T_full = data.sort_values(["$T$","$N$"])

    data_T = data_T_full[estimator].to_numpy().reshape(-1,1)

    N2 = data_T_full["$N$"].to_numpy().reshape(-1,1)
    T2 = data_T_full["$T$"].to_numpy().reshape(-1,1)

    data_T = np.concatenate((N2,T2,data_T),axis=1)

    data_T_slice1 = data_T[0:5]
    data_T_slice2 = data_T[5:10]
    data_T_slice3 = data_T[10:15]
    data_T_slice4 = data_T[15:20]
    data_T_slice5 = data_T[20:25]

    # plotting

    for data_slice in [data_N_slice1,data_N_slice2,data_N_slice3,data_N_slice4,data_N_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)
        plot_axis.set_xticks(N2.flatten())
        plot_axis.set_yticks(T2.flatten())
        plot_axis.tick_params(axis='both', which='major', labelsize=15)

    for data_slice in [data_T_slice1,data_T_slice2,data_T_slice3,data_T_slice4,data_T_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)


    




def estimator_subplot_N_growing(plot_axis,data,estimator):

    """
    function for plotting a subplot
    for a particular estimator    
    """


    # T part
    # **************************

    data_T = data[estimator].to_numpy().reshape(-1,1)

    N2 = data["$N$"].to_numpy().reshape(-1,1)
    T2 = data["$T$"].to_numpy().reshape(-1,1)

    data_T = np.concatenate((N2,T2,data_T),axis=1)

    data_T_slice1 = data_T[0:5]
    data_T_slice2 = data_T[5:10]
    data_T_slice3 = data_T[10:15]
    data_T_slice4 = data_T[15:20]
    data_T_slice5 = data_T[20:25]

    # N part
    # **************************

    data_N_full = data.sort_values(["$N$","$T$"])

    data_N = data_N_full[estimator].to_numpy().reshape(-1,1)

    N2 = data_N_full["$N$"].to_numpy().reshape(-1,1)
    T2 = data_N_full["$T$"].to_numpy().reshape(-1,1)

    data_N = np.concatenate((N2,T2,data_N),axis=1)

    data_N_slice1 = data_N[0:5]
    data_N_slice2 = data_N[5:10]
    data_N_slice3 = data_N[10:15]
    data_N_slice4 = data_N[15:20]
    data_N_slice5 = data_N[20:25]


    for data_slice in [data_T_slice1,data_T_slice2,data_T_slice3,data_T_slice4,data_T_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)
        plot_axis.set_xticks(N2.flatten())
        plot_axis.set_yticks(T2.flatten())
        plot_axis.tick_params(axis='both', which='major', labelsize=15)


    for data_slice in [data_N_slice1,data_N_slice2,data_N_slice3,data_N_slice4,data_N_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)







def estimator_subplot_beta_var(plot_axis,data,estimator):

    """
    function for plotting a subplot
    for a particular estimator    
    """

    # beta part
    # **************************


    data_beta = data[estimator].to_numpy().reshape(-1,1)

    N2 = data["$beta$"].to_numpy().reshape(-1,1)
    T2 = data["$T$"].to_numpy().reshape(-1,1)

    data_beta = np.concatenate((N2,T2,data_beta),axis=1)

    data_beta_slice1 = data_beta[0:5]
    data_beta_slice2 = data_beta[5:10]
    data_beta_slice3 = data_beta[10:15]
    data_beta_slice4 = data_beta[15:20]


    # T part
    # **************************


    data_T_full = data.sort_values(["$T$","$beta$"])

    data_T = data_T_full[estimator].to_numpy().reshape(-1,1)

    N2 = data_T_full["$beta$"].to_numpy().reshape(-1,1)
    T2 = data_T_full["$T$"].to_numpy().reshape(-1,1)

    data_T = np.concatenate((N2,T2,data_T),axis=1)

    # data_T_slice1 = data_T[0:5]
    # data_T_slice2 = data_T[5:10]
    # data_T_slice3 = data_T[10:15]
    # data_T_slice4 = data_T[15:20]

    data_T_slice1 = data_T[0:4]
    data_T_slice2 = data_T[4:8]
    data_T_slice3 = data_T[8:12]
    data_T_slice4 = data_T[12:16]
    data_T_slice5 = data_T[16:20]

    # plotting

    for data_slice in [data_beta_slice1,data_beta_slice2,data_beta_slice3,data_beta_slice4]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)
        plot_axis.set_xticks(N2.flatten())
        plot_axis.set_yticks(T2.flatten())
        plot_axis.tick_params(axis='both', which='major', labelsize=15)

    for data_slice in [data_T_slice1,data_T_slice2,data_T_slice3,data_T_slice4,data_T_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)




def estimator_subplot_rho_var(plot_axis,data,estimator):

    """
    function for plotting a subplot
    for a particular estimator    
    """

    # beta part
    # **************************


    data_beta = data[estimator].to_numpy().reshape(-1,1)

    N2 = data["$rho$"].to_numpy().reshape(-1,1)
    T2 = data["$T$"].to_numpy().reshape(-1,1)

    data_beta = np.concatenate((N2,T2,data_beta),axis=1)

    data_beta_slice1 = data_beta[0:5]
    data_beta_slice2 = data_beta[5:10]
    data_beta_slice3 = data_beta[10:15]
    data_beta_slice4 = data_beta[15:20]


    # T part
    # **************************


    data_T_full = data.sort_values(["$T$","$rho$"])

    data_T = data_T_full[estimator].to_numpy().reshape(-1,1)

    N2 = data_T_full["$rho$"].to_numpy().reshape(-1,1)
    T2 = data_T_full["$T$"].to_numpy().reshape(-1,1)

    data_T = np.concatenate((N2,T2,data_T),axis=1)

    # data_T_slice1 = data_T[0:5]
    # data_T_slice2 = data_T[5:10]
    # data_T_slice3 = data_T[10:15]
    # data_T_slice4 = data_T[15:20]

    data_T_slice1 = data_T[0:4]
    data_T_slice2 = data_T[4:8]
    data_T_slice3 = data_T[8:12]
    data_T_slice4 = data_T[12:16]
    data_T_slice5 = data_T[16:20]

    # plotting

    for data_slice in [data_beta_slice1,data_beta_slice2,data_beta_slice3,data_beta_slice4]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)
        plot_axis.set_xticks(N2.flatten())
        plot_axis.set_yticks(T2.flatten())
        plot_axis.tick_params(axis='both', which='major', labelsize=15)

    for data_slice in [data_T_slice1,data_T_slice2,data_T_slice3,data_T_slice4,data_T_slice5]:
        x, y, z = data_slice[:,0],data_slice[:,1],data_slice[:,2]
        plot_axis.plot(x,y,z,c='tab:blue', linewidth=3)
    



def setting_plot_T_growing(data_df,title):

    fig = plt.figure(figsize=(15, 10),layout="constrained")

    fig.suptitle(title,fontsize = 30)

    subplot_font_size = 25
    axes_label_font_size = 25

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    #ax.set_title("IC_1", size=subplot_font_size)
    ax.set_title("IC_1",fontweight="bold", size=subplot_font_size)
    #ax.title.set_text("IC_1")
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$IC_1$")


    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.set_title("PC_1",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$PC_1$")

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.set_title("BIC_3",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$BIC_3$")

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.set_title("ER",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$ER$")

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.set_title("GR",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$GR$")

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.set_title("ED",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_T_growing(ax,data_df,"$ED$")


    return fig


def setting_plot_beta_var(data_df,title):

    fig = plt.figure(figsize=(15, 10),layout="constrained")

    fig.suptitle(title,fontsize = 30)

    subplot_font_size = 25
    axes_label_font_size = 25

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    #ax.set_title("IC_1", size=subplot_font_size)
    ax.set_title("IC_1",fontweight="bold", size=subplot_font_size)
    #ax.title.set_text("IC_1")
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$IC_1$")


    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.set_title("PC_1",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$PC_1$")

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.set_title("BIC_3",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$BIC_3$")

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.set_title("ER",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$ER$")

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.set_title("GR",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$GR$")

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.set_title("ED",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\beta$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_beta_var(ax,data_df,"$ED$")


    return fig



def setting_plot_rho_var(data_df,title):

    fig = plt.figure(figsize=(15, 10),layout="constrained")

    fig.suptitle(title,fontsize = 30)

    subplot_font_size = 25
    axes_label_font_size = 25

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    #ax.set_title("IC_1", size=subplot_font_size)
    ax.set_title("IC_1",fontweight="bold", size=subplot_font_size)
    #ax.title.set_text("IC_1")
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$IC_1$")


    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.set_title("PC_1",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$PC_1$")

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.set_title("BIC_3",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$BIC_3$")

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.set_title("ER",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$ER$")

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.set_title("GR",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$GR$")

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.set_title("ED",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel(r"$\rho$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_rho_var(ax,data_df,"$ED$")


    return fig



def setting_plot_N_growing(data_df,title):

    fig = plt.figure(figsize=(15, 10),layout="constrained")

    fig.suptitle(title,fontsize = 30)

    subplot_font_size = 25
    axes_label_font_size = 25

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    #ax.set_title("IC_1", size=subplot_font_size)
    ax.set_title("IC_1",fontweight="bold", size=subplot_font_size)
    #ax.title.set_text("IC_1")
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$IC_1$")


    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.set_title("PC_1",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$PC_1$")

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.set_title("BIC_3",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$BIC_3$")

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.set_title("ER",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$ER$")

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.set_title("GR",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$GR$")

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.set_title("ED",fontweight="bold", size=subplot_font_size)
    ax.set_xlabel("$N$", fontsize=axes_label_font_size)
    ax.set_ylabel("$T$", fontsize=axes_label_font_size)
    ax.set_zlim([0.0, 8.0])
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    estimator_subplot_N_growing(ax,data_df,"$ED$")


    return fig


