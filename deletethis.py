from PerformanceMetrics.metrics import Metrics
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

metrics = Metrics()


# CROSS_LIGHT = [{ELEMENT_SIZE:10,ELEMENT_COUNT:20,UNITS_COUNT:20,RECONFIG:[],SUPPORTED_LAYER_LIST:["convolution"]},{ELEMENT_SIZE:15,ELEMENT_COUNT:30,UNITS_COUNT:20,RECONFIG:[],SUPPORTED_LAYER_LIST:["inner_product"]}]
# HOLY_LIGHT = [{ELEMENT_SIZE:128,ELEMENT_COUNT:128,UNITS_COUNT:8,RECONFIG:[]}]
# RECONFIG_NINE = [{ELEMENT_SIZE:78,ELEMENT_COUNT:75,UNITS_COUNT:14, RECONFIG:[9]}]
# RECONFIG_TWOFIVE = [{ELEMENT_SIZE:50,ELEMENT_COUNT:5,UNITS_COUNT:100, RECONFIG:[9,25]}]
# RECONFIG_NINE_SIXTEEN = [{ELEMENT_SIZE:78,ELEMENT_COUNT:70,UNITS_COUNT:14, RECONFIG:[9,16]}]
# RECONFIG_NINE_SIXTEEN_TWENTY_FIVE = [{ELEMENT_SIZE:78,ELEMENT_COUNT:54,UNITS_COUNT:22, RECONFIG:[4,9,16,25]}]
# RECONFIG_NINE_SIXTEEN_TWENTY_FIVE_MAM = [{ELEMENT_SIZE:128,ELEMENT_COUNT:13,UNITS_COUNT:78, RECONFIG:[4,9,16,25]}]
# cross_light_area = metrics.get_total_area("AMM",20,0,10,20,0,0) + metrics.get_total_area("AMM",20,0,15,30,0,0)
# print("Cross Light ", cross_light_area)
# holy_light_area = metrics.get_total_area("MAM",8,0,128,128,0,0)
# print("Holy Light Area ", holy_light_area)
# reconfig_mam = metrics.get_total_area("AMM",22,0,78,54,0,0)
# print("Reconfig MAM ",reconfig_mam)
# reconfig_amm = metrics.get_total_area("MAM",78,0,128,13,0,0)
# print("Reconfig AMM ",reconfig_amm)
# amm_base = metrics.get_total_area("AMM",14,0,78,78,0,0)
# print("AMM Base ", amm_base)
_result_df = pd.read_csv('Result/fpswarea.csv')
ax = sns.barplot(x="Architecture",y="Value", hue="Neural_Network", data = _result_df)     
ax.set(xlabel='VDP Architecture', ylabel='FPS/W/Area')
plt.show()