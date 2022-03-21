from turtle import color, width
import matplotlib.pyplot as plt 
import os
import pandas as pd
from pandas.core.algorithms import mode
import json
import seaborn as sns
from matplotlib import ticker


def plotAndSaveBarplot(df, x_column, y_column, y_axis_label ,hue_column, ncol, file_name, fig_save_dir):
    """This method should accept a dataframe, x axis column, y axis column name, hue column name, legend label count per row, fileName, figure saving location, filter column name value dictionary

    Args:
        result_dataframe (_type_): None
    """
    
    df1 = df 
    # df1 = df.drop(df[df['name'] == 'STOCHASTIC'].index) 
    sns.set_context(rc = {'patch.linewidth': 0.0})
    fig_dims = (13, 2)
    fig, (ax1) = plt.subplots(figsize=fig_dims)
    plt.rcParams.update({'font.family':'Times New Roman'})

    ax1 = sns.barplot(ax= ax1, x=x_column,y=y_column, hue=hue_column, data = df1, palette = 'deep')
    ax1.set(ylabel=y_axis_label) 
    y1lim = df1[y_column].max()
    # ax.set_ylim(0,100)
    if y_column=="fps_per_w_per_area" or y_column=="fps_per_w" or y_column == 'fps':
        ax1.ticklabel_format(axis='y', style='sci',  scilimits=(3,6))
    ax1.xaxis.label.set_visible(False)
    ax1.xaxis.label.set_fontsize(14)
    ax1.yaxis.label.set_fontsize(14)
    ax1.xaxis.label.set_fontweight('bold')
    ax1.yaxis.label.set_fontweight('bold')
    t = ax1.yaxis.get_offset_text()
    t.set_x(-0.055)
    t.fontweight = 'bold'
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    for label in labels: label.set_fontweight('bold')
    ax1.tick_params(axis= 'y', labelsize = 15)
    ax1.tick_params(axis = 'x', labelsize = 15)
    sns.move_legend(
        ax1, "lower center",
        bbox_to_anchor=(.5, 1), ncol=ncol, title=None,handletextpad=0.4, columnspacing = 0.5, handlelength =1.7 , prop = {'weight':'bold','size':'14'}, borderaxespad= 0, framealpha=0)
    # plt.xticks(rotation = 45)
    plt.rcParams.update({'font.family':'Times New Roman'})
    plt.savefig(fig_save_dir+file_name,bbox_inches='tight' )
    plt.show()
        
    



df1= pd.read_csv('Result/ACC_FOUR_BIT/Analog_Metrics.csv')
df2 = pd.read_csv('Result/ACC_FOUR_BIT/Stochastic_BER9_Metrics.csv')
df3 = pd.read_csv('Result/ACC_FOUR_BIT/Stochastic_BER3_Metrics.csv')
# * Dataframe to be ploted 
df =  pd.concat([df1, df2, df3], ignore_index=True)
# * Metrics to be plotted and saved as seperate figures in for dic with {'metric_column_name': 'Y axis label'}
parameters_label= {'fps':"FPS", 'fps_per_w_per_area':'$FPS/W/mm^2$' ,"fps_per_w":"FPS/W"}
# parameters_label= {'fps':'$FPS$'}
# parameters_label= {}
#* filters: column values to be removed from the plotting dataframe
for metric in parameters_label:
        fileName = parameters_label[metric].replace(".csv","").replace(" ","_").replace("/","_").replace("$",'_')+'.png'
        plotAndSaveBarplot(df,'Model_Name',metric,parameters_label[metric],'name', 5,fileName ,'Plots/ACC_FOUR_BIT/')


