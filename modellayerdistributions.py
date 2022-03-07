import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

cnnModelDirectory = "./CNNModels/"
modelList =  [f for f in listdir(cnnModelDirectory) if isfile(join(cnnModelDirectory, f))]
model_distribution_list = []
for modelName in modelList:
    model_distribution = {}
    model_df = pd.read_csv(cnnModelDirectory+modelName)
    model_name = modelName
    depthwiseCount = model_df[model_df.name == 'DepthWiseConv'].shape[0]
    pointwiseCount = model_df[model_df.name == 'PointWiseConv'].shape[0]
    model_distribution['Type'] = 'DepthWise'
    model_distribution['Model_Name'] = model_name
    model_distribution['Count'] = depthwiseCount
    model_distribution_list.append(model_distribution)
    model_distribution = {}
    model_distribution['Type'] = 'PointWise'
    model_distribution['Model_Name'] = model_name
    model_distribution['Count'] = pointwiseCount
    
    model_distribution_list.append(model_distribution)
distribution_df = pd.DataFrame(model_distribution_list)
ax = sns.barplot( x="Model_Name",y='Count', hue="Type", data = distribution_df) 

plt.show()

print(distribution_df)

