import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


SOMA = range(5,100,3)
B = [2,4,8,16]
S = 0.1e-3
supported_list = []

for Z in range(1,200,1): 
    for bit_precision in B:
        for soma in SOMA:
            supported = {}
            analog_levels = np.power(2,(bit_precision+np.log2(Z)))
            available_power = analog_levels*soma*1e-6
    
            if (available_power/Z)<=(1e-3-S):
                # print("Supported")
                supported['Z'] = Z
                supported['SOMA'] = soma
                supported['Precision'] = bit_precision
                supported['Valid'] = 'Supported'
            else:
                supported['Z'] = Z
                supported['SOMA'] = soma
                supported['Precision'] = bit_precision
                supported['Valid'] = 'Not Supported'
            supported_list.append(supported)
result = pd.DataFrame(supported_list)
result = result[result['Precision']==4]

ax = sns.scatterplot(data=result,x='SOMA',y='Z',hue='Valid',palette=['green','red'])
# ax.set(xlabel='4 Bit Precision', ylabel='Supported Z')
plt.title("4 Bit Precision")
plt.show()
# 
# result.to_csv('analysis.csv')
# print(supported_list)            