import matplotlib.pyplot as plt 
import os
import pandas as pd
from pandas.core.algorithms import mode
import json
import seaborn as sns
#'total_latency','total_dynamic_energy','total_static_power','fps','area','edp','fps_per_w'
# parameters = ['total_latency','total_dynamic_energy','total_static_power','fps','area','fps_per_w']

AREA = {
        'AMM_BASE':151.56,
        'Crosslight':29.736,
        'Holylight':216.122,
        'Reconfig_4_9_16_25':416.83,
        'Reconfig_4_9_16_25_MAM':292.55
        }

parameters = ['fps','fps_per_w']
parameters_label= {'total_latency':'Latency(S)', 'fps':"FPS", "area":'Area', "fps_per_w":"FPS/W/Area","total_dynamic_energy":"Data Dependent Energy(J)","total_static_power":"Non Data Dependent Energy(J)","hardware_utilization":"TPC Utilization"}
_result_directory = 'Result/'
_result_directory_sub_folder = [x[0] for x in os.walk(_result_directory)]
_result_directory_sub_folder = _result_directory_sub_folder[1:]
_data = {}
_result_df = pd.DataFrame(columns = ['Parameter','Architecture','Neural_Network','Value'])
for parameter in parameters:
    
    _data[parameter] = {}
    for architecture in _result_directory_sub_folder: 
        _nnmodels = os.listdir(architecture)    
        _data[parameter][architecture.replace(_result_directory,'')] = {}
        print("Architecture :",architecture)
        architecture_values = []
        for nn_model in _nnmodels:
            print("Neural Network ", nn_model)
            try:
                model_df = pd.read_csv(architecture+'/'+nn_model)
                _data[parameter][architecture.replace(_result_directory,'')][nn_model.replace('Arch.csv','')] = model_df[parameter][0]
                _result = {}
                _result['Parameter'] = parameter
                _result['Architecture'] = architecture.replace(_result_directory,'')
                
                _result['Neural_Network'] = nn_model.replace('Arch.csv','').replace('.csv','')
                _result['Value'] =  model_df[parameter][0]
                print(_result)
                _result_df = _result_df.append(_result, ignore_index = True)
                print(_result_df)
                # print("Area ===>", AREA(_result['Architecture']))
                # _result['Value'] =  (model_df[parameter][0])/ (AREA[_result['Architecture']])
            except Exception as error:
                _result['Parameter'] = ""
                _result['Architecture'] = ""
                _result['Neural_Network'] = ""
                _result['Value'] =  ""
                
    print(json.dumps(_data,indent=4))
    


# _result_df['edp'] = _result_df.apply(lambda row:(row['fps_per_w']*row['total_latency']),axis=1)  
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Hybrid"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Sample"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Architecture == "CrossLight"].index)
_result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "AATEST"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "ResNet50"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "OverFeatFast"].index)
# _result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "VGG16"].index)
_result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "DenseNet"].index)
_result_df = _result_df.drop(_result_df[_result_df.Neural_Network == "Alexnet"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Reconfig25"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Reconfig_9"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Reconfig_9_16"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Reconfig_9_16_25"].index)
_result_df = _result_df.drop(_result_df[_result_df.Architecture == "Reconfig_4_9_16"].index)
print(_result_df.to_csv('Result/parameter_result.csv'))

# performance_metrics = sns.load_dataset(_result_df)
plt.figure()
ax = sns.barplot(x="Architecture",y="Value", hue="Neural_Network", data = _result_df)     
ax.set(xlabel='TPC Architecture', ylabel=parameters_label[parameter])
plt.show()