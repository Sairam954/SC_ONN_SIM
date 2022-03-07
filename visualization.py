import matplotlib.pyplot as plt 
import os
import pandas as pd
from pandas.core.algorithms import mode
import json
import seaborn as sns

result_df_depthwise = pd.read_csv('Result/Depthwise/all_models_metrics.csv')
result_df_pointwise = pd.read_csv('Result/Pointwise/all_models_metrics.csv')
result_df_complete = pd.read_csv('Result/Complete/all_models_metrics.csv')
result_df_large = pd.read_csv('Result/Complete/largemodelsmetrics.csv')
result_df_small = pd.read_csv('Result/Complete/smallmodelsmetrics.csv')
result_area_equivalent = pd.read_csv('Result/Area_Equivalent/all_model_metrics.csv')

# result_df_complete_sample = pd.read_csv('Result/Sample/all_models_metrics.csv')

# print(result_df)

parameters_label= {'total_latency':'Latency(S)', 'fps':"FPS", "area":'Area','fps_per_w_per_area':'FPS/W/mm2' ,"fps_per_w":"FPS/W","total_dynamic_energy":"Data Dependent Energy(J)","total_static_power":"Non Data Dependent Power(W)","hardware_utilization":"TPC Utilization"}



_result_df = result_df_small

# _result_df = _result_df.drop(_result_df[_result_df.Size == "large"].index)

# # # #*Group 1
# _result_df = result_df.drop(result_df[result_df.Model_Name == "Xception.csv"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Model_Name == "EfficientNet_B7.csv"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Model_Name == "ResNet50.csv"].index)

# #*Group 2
# _result_df = result_df.drop(result_df[result_df.Model_Name == "ShuffleNet_V2.csv"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Model_Name == "NASNetMobile.csv"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Model_Name == "MobileNet_V2.csv"].index)


# y = 'fps'
# y = 'fps_per_w'
# y= 'fps_per_w_per_area'
# y= 'area'
# y = 'total_static_power'
# y = 'total_dynamic_energy'
y = 'hardware_utilization'

# fig, axes = plt.subplots(1,2,sharey=True)
# ax = sns.barplot( x="name",y="fps", hue="Model_Name", data = result_df_depthwise)  

sns.set(font_scale = 3)
ax = sns.barplot( x="Model_Name",y=y, hue="name", data = _result_df) 
ax.set(xlabel='NN Models', ylabel=parameters_label[y]) 

plt.legend(bbox_to_anchor=(0.6, 1), loc='upper center',markerscale=0.1, ncol = 9, frameon= False, handletextpad=0.1)
plt.setp(ax.get_legend().get_texts(), fontsize='10') 
plt.savefig('Plots/Sample.png')
plt.show()