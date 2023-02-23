import pandas as pd
from os.path import isfile, join
from os import listdir
import os
import numpy as np
from PerformanceMetrics.metrics import Metrics
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

metrics = Metrics()
print('AMM', metrics.get_total_area('AMM',1, 31, 31,'ANALOG'))
print('MAM', metrics.get_total_area('MAM',1, 44, 44,'ANALOG'))
print('MMA', metrics.get_total_area('MMA',1, 83, 83,'ANALOG'))

# def im2col(X,conv1, stride, pad):
#     # Padding
#     X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
#     X = X_padded
#     new_height = int((X.shape[2]+(2*pad)-(conv1.shape[2]))/stride)+1
#     new_width =  int((X.shape[3]+(2*pad)-(conv1.shape[3]))/stride)+1
#     im2col_vector = np.zeros((X.shape[1]*conv1.shape[2]*conv1.shape[3],new_width*new_height*X.shape[0]))
#     c = 0
#     for position in range(X.shape[0]):

#         image_position = X[position,:,:,:]
#         for height in range(0,image_position.shape[1],stride):
#             image_rectangle = image_position[:,height:height+conv1.shape[2],:]
#             if image_rectangle.shape[1]<conv1.shape[2]:
#                 continue
#             else:
#                 for width in range(0,image_rectangle.shape[2],stride):
#                     image_square = image_rectangle[:,:,width:width+conv1.shape[3]]
#                     if image_square.shape[2]<conv1.shape[3]:
#                         continue
#                     else:
#                         im2col_vector[:,c:c+1]=image_square.reshape(-1,1)
#                         c = c+1         
            
#     return(im2col_vector)
# # * Input model files column headers constants
# LAYER_TYPE = "name"
# MODEL_NAME = "model_name"
# KERNEL_DEPTH = "kernel_depth"
# KERNEL_HEIGHT = "kernel_height"
# KERNEL_WIDTH = "kernel_width"
# TENSOR_COUNT = "tensor_count"
# INPUT_SHAPE = "input_shape"
# OUTPUT_SHAPE = "output_shape"
# TENSOR_SHAPE = "tensor_shape"
# INPUT_HEIGHT = "input_height"
# INPUT_WIDTH = "input_width"
# INPUT_DEPTH = "input_depth"
# OUTPUT_HEIGHT = "output_height"
# OUTPUT_WIDTH = "output_width"
# OUTPUT_DEPTH = "output_depth"


# cnnModelDirectory = "./CNNModels/"
# modelList = [f for f in listdir(
#     cnnModelDirectory) if isfile(join(cnnModelDirectory, f))]
# modelList = ['GoogLeNet.csv','ResNet50.csv','MobileNet_V2.csv', 'ShuffleNet_V2.csv']
# max_capacitor_count = 0
# row_set = set()
# for modelName in modelList:

#     nnModel = pd.read_csv(cnnModelDirectory+modelName)
#     nnModel = nnModel.astype({"model_name": str, 'name': str, 'kernel_depth': int, 'kernel_height': int, 'kernel_width': int,	'tensor_count': int, 'input_shape': str,
#                                 'output_shape': str, 'tensor_shape': str,	'input_height': int,	'input_width': int, 'input_depth': int, 'output_height': int, 'output_width': int, 'output_depth': int})

#     # # * filter specific layers for debugging
#     # nnModel = nnModel.drop(nnModel[nnModel.name == "DepthWiseConv"].index)
#     # nnModel = nnModel.drop(nnModel[nnModel.name == "Conv2D"].index)
#     # nnModel = nnModel.drop(nnModel[nnModel.name == "PointWiseConv"].index)
#     # nnModel = nnModel.drop(nnModel[nnModel.name == "Dense"].index)
#     # nnModel = nnModel.drop(nnModel[nnModel.name == "MaxPooling2D"].index)
#     print("Model Name: ", modelName)
#     total_latency = []
#     vdp_ops = []
#     vdp_sizes = []
#     for idx in nnModel.index:
#         layer_type = nnModel[LAYER_TYPE][idx]
#         model_name = nnModel[MODEL_NAME][idx]
#         kernel_depth = nnModel[KERNEL_DEPTH][idx]
#         kernel_width = nnModel[KERNEL_WIDTH][idx]
#         kernel_height = nnModel[KERNEL_HEIGHT][idx]
#         tensor_count = nnModel[TENSOR_COUNT][idx]
#         input_shape = nnModel[INPUT_SHAPE][idx]
#         output_shape = nnModel[OUTPUT_SHAPE][idx]
#         tensor_shape = nnModel[TENSOR_SHAPE][idx]
#         input_height = nnModel[INPUT_HEIGHT][idx]
#         input_width = nnModel[INPUT_WIDTH][idx]
#         input_depth = nnModel[INPUT_DEPTH][idx]
#         output_height = nnModel[OUTPUT_HEIGHT][idx]
#         output_width = nnModel[OUTPUT_WIDTH][idx]
#         output_depth = nnModel[OUTPUT_DEPTH][idx]
#         conv1 = np.random.randn(tensor_count,kernel_height,kernel_width,kernel_depth) 
#         X = np.random.randn(1,input_height,input_width,input_depth)
#         # print(X)
#         stride = 1
#         # Toeplitz matrix
#         X_im2col = im2col(X=X,conv1=conv1,pad=0,stride=1)
#         conv1_col = conv1.reshape(conv1.shape[0],-1)
#         capacitor_count = X_im2col.shape[1]
#         row_set.add(X_im2col.shape[0])
#         if capacitor_count > max_capacitor_count:
#             max_capacitor_count = capacitor_count
#         no_of_vdp_ops = output_height*output_depth*output_width
#         no_of_vdp_ops_toepltiz = tensor_count*X_im2col.shape[1] 
#         print(no_of_vdp_ops==no_of_vdp_ops_toepltiz)
# print("Max Capacitor Count: ", max_capacitor_count)
# print("Row Combination: ", row_set)


