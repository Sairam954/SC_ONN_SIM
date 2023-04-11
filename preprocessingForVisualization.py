from platform import architecture
import pandas as pd



preprocessFileList = ['BatchSize_1_IS_PCA.csv', 'BatchSize_1_OS_PCA.csv', 'BatchSize_1_WS_PCA.csv']
newDataframe = []
for file in preprocessFileList:
    df = pd.read_csv('Result/HEANA/'+file)
    
    df.to_csv('Result/HEANA/'+file, index=False)

    for idx in df.index:
        architectureName = df['name'][idx]
        model_name = df['Model_Name'][idx]
        datarate = df['Datarate'][idx]
        dataflow = df['Dataflow'][idx]
        architectureName = architectureName.split('_')[0]
        architectureName = architectureName+"_PCA"
        if dataflow == 'Output-Stationary':
            architectureName = architectureName+"_OS"
        elif dataflow == 'Input-Stationary':
            architectureName = architectureName+"_IS"
        elif dataflow == 'Weight-Stationary':
            architectureName = architectureName+"_WS"
        
        if datarate == 1:
            architectureName = architectureName+"(1 Gs/s)"
        elif datarate == 5:
            architectureName = architectureName+"(5 Gs/s)"
        elif datarate == 10 and dataflow!="Output-Stationary":
             architectureName = architectureName+"(10 Gs/s)"
        elif datarate == 10 and dataflow=="Output-Stationary":
              architectureName = architectureName+"(1 Gs/s)"
        elif datarate == 50:
            architectureName = architectureName+"(5 Gs/s)"
        elif datarate == 100:
            architectureName = architectureName+"(10 Gs/s)"
            
        row = {'name':architectureName, 'model_name':model_name, 'total_latency':df['total_latency'][idx], 'fps':df['fps'][idx], 'fps_per_w':df['fps_per_w'][idx]}
        newDataframe.append(row)
new_df = pd.DataFrame(newDataframe)
new_df.to_csv('Result/HEANA/Preprocessed_HEANA_BatchSize_256.csv', index=False)