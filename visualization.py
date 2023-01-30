from turtle import color, width
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import seaborn as sns
from matplotlib import ticker
from scipy import stats


def plotAndSaveBarplot(df, x_column, y_column, y_axis_label, hue_column, ncol, file_name, fig_save_dir):
    """This method should accept a dataframe, x axis column, y axis column name, hue column name, legend label count per row, fileName, figure saving location, filter column name value dictionary

    Args:
        result_dataframe (_type_): None
    """

    df1 = df
    # df1 = df.drop(df[df['name'] == 'STOCHASTIC'].index)
    sns.set_context(rc={'patch.linewidth': 0.0})
    fig_dims = (13, 2)
    fig, (ax1) = plt.subplots(figsize=fig_dims)
    plt.rcParams.update({'font.family': 'Times New Roman'})

    ax1 = sns.barplot(ax=ax1, x=x_column, y=y_column,
                      hue=hue_column, data=df1, palette='deep')
    # for container in ax1.containers:
    #     ax1.bar_label(container)
    ax1.set(ylabel=y_axis_label)
    y1lim = df1[y_column].max()
    # ax.set_ylim(0,100)
    if y_column == "fps_per_w_per_area" or y_column == "fps_per_w" or y_column == "total_latency":
        ax1.ticklabel_format(axis='y', style='sci',  scilimits=(3, 6))
    else:
        ax1.set_yscale("log")
    ax1.xaxis.label.set_visible(False)
    ax1.xaxis.label.set_fontsize(14)
    ax1.yaxis.label.set_fontsize(14)
    ax1.xaxis.label.set_fontweight('bold')
    ax1.yaxis.label.set_fontweight('bold')
    # ax1.set_ylim([0, 7E-3]) # latency
    # ax1.set_ylim([0, 9E-6]) # FPS/W
    # ax1.set_ylim([0, 9E-12]) # FPS/W/mm2
    t = ax1.yaxis.get_offset_text()
    t.set_x(-0.01)
    # t.set_y(0.05)
    t.fontweight = 'bold'
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    for label in labels:
        label.set_fontweight('bold')
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    sns.move_legend(
        ax1, "lower center",
        bbox_to_anchor=(.5, 1.14), ncol=ncol, title=None, handletextpad=0.4, columnspacing=0.5, handlelength=1.7, prop={'weight': 'bold', 'size': '14'}, borderaxespad=0, framealpha=0)
    # plt.xticks(rotation = 45)
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.savefig(fig_save_dir+file_name, bbox_inches='tight')
    plt.show()


def calGmeanDF(df, cal_gmean_col):
    accelerator_types = list(df['name'].unique())
    for accelerator in accelerator_types:
        print(accelerator)
        gmean_row = {}
        df_acc = df[df['name'] == accelerator]
        gmean_row['name'] = accelerator
        gmean_row['Model_Name'] = 'Gmean'
        gmean_row['config'] = 'NA'
        for col in cal_gmean_col:
            gmean_row[col] = stats.gmean(df_acc[col])
        df = df.append(gmean_row, ignore_index=True)
    return df


acc_precision = 'ACC_SIXTEEN_BIT'
# cal_gmean_col = ['hardware_utilization','total_latency','fps','total_dynamic_energy','total_static_power','fps_per_w','area','fps_per_w_per_area']
# df1= pd.read_csv('Result/'+acc_precision+'/Analog_Metrics.csv')
# df1 = calGmeanDF(df1,cal_gmean_col)
# df2 = pd.read_csv('Result/'+acc_precision+'/STOCHASTIC_BER_9.csv')
# df2 = calGmeanDF(df2,cal_gmean_col)
# df3 = pd.read_csv('Result/'+acc_precision+'/STOCHASTIC_BER_3.csv')

# * Dataframe to be ploted
# df =  pd.concat([df1, df2, df3], ignore_index=True)
# df.to_csv('Result/'+acc_precision+'/Combined.csv')

df = pd.read_csv('Result/ISQED/Vis_Test.csv')
cal_gmean_col = ['fps']
df = calGmeanDF(df, cal_gmean_col)
df = df.drop(df[df['Model_Name'] == 'VGG16'].index)
df = df.drop(df[df['Model_Name'] == 'DenseNet121'].index)

# * Metrics to be plotted and saved as seperate figures in for dic with {'metric_column_name': 'Y axis label'}
# parameters_label= {'fps':"FPS (Log Scale)", 'fps_per_w_per_area':'$FPS/W/mm^2$' ,"fps_per_w":"FPS/W"}
parameters_label = {'fps':"FPS (Log Scale)"}
# parameters_label= {}
# * filters: column values to be removed from the plotting dataframe
for metric in parameters_label:
    fileName = parameters_label[metric].replace(".csv", "").replace(
        " ", "_").replace("/", "_").replace("$", '_')+'.png'
    plotAndSaveBarplot(df, 'Model_Name', metric,
                       parameters_label[metric], 'name', 5, fileName, 'Plots/ISQED/')

# * The below code gives the descriptive information on which accelerator is better than other accelerators and by how much

df_descriptive = df[df['Model_Name'] == 'Gmean'].reset_index(drop=True)
print(df_descriptive)
parameter_column = 'fps'
findmax_query = parameter_column+"=="+parameter_column+".max()"
max_row = df_descriptive.query(findmax_query)
print("Max Row", max_row)
max_tpc_name = max_row['name'].values[0]
max_fps = max_row[parameter_column]

for idx in df_descriptive.index:
    # print("MAX achieved", max_fps)
    # print("Compared with", )
    increment = max_fps/df_descriptive[parameter_column][idx]
    increment = increment.values[0]
    print("The accelerator " + str(max_tpc_name) + " achieves "+str(increment) +
          "x times better "+parameter_column+" than "+str(df_descriptive['name'][idx]))

df_descriptive = df_descriptive.drop(
    df_descriptive[df_descriptive['name'] == max_tpc_name].index)

print("Details of second best accelerator")
# print(df_descriptive)
max_row = df_descriptive.query(findmax_query)
max_tpc_name = max_row['name'].values[0]
max_fps = max_row[parameter_column]

for idx in df_descriptive.index:
    # print("MAX achieved", max_fps)
    # print("Compared with", )
    increment = max_fps/df_descriptive[parameter_column][idx]
    increment = increment.values[0]
    print("The accelerator " + str(max_tpc_name) + " achieves "+str(increment) +
          "x times better "+parameter_column+" than "+str(df_descriptive['name'][idx]))


# print(df_descriptive.iloc[df['fps'].argmax()])
