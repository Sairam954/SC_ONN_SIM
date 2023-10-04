

# SC_ONN_SIM (Stochastic Computing Optical Neural Network Simulator)

This is a transaction-level, event-driven python-based simulator for evaluation of stochastic computing based optical neural network accelerators for various quantized Convolutional Neural Network models.  

### ArXiv Preprint
https://arxiv.org/abs/2302.07036



### Installation and Execution

    git clone https://github.com/Sairam954/SC_ONN_SIM.git
    python main.py

### Bibtex

Please cite us if you use SC_ONN_SIM

```bash
@inproceedings{DBLP:conf/ipps/VatsavaiKTSH23,
  author       = {Sairam Sri Vatsavai and
                  Venkata Sai Praneeth Karempudi and
                  Ishan G. Thakkar and
                  Sayed Ahmad Salehi and
                  Jeffrey Todd Hastings},
  title        = {{SCONNA:} {A} Stochastic Computing Based Optical Accelerator for Ultra-Fast,
                  Energy-Efficient Inference of Integer-Quantized CNNs},
  booktitle    = {{IEEE} International Parallel and Distributed Processing Symposium,
                  {IPDPS} 2023, St. Petersburg, FL, USA, May 15-19, 2023},
  pages        = {546--556},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/IPDPS54959.2023.00061},
  doi          = {10.1109/IPDPS54959.2023.00061},
  timestamp    = {Tue, 25 Jul 2023 16:27:14 +0200},
  biburl       = {https://dblp.org/rec/conf/ipps/VatsavaiKTSH23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

### Video Tutorial
https://youtu.be/fjjq4jv_iCk

### Accelerator Configuration 

The accelerator configuration can be provided in main.py file. The configuration dictionary looks like below:
``` bash
ACCELERATOR = [
{
    ELEMENT_SIZE: 176, # The supported dot product size of the processing unit, generally equal to number of wavelengths multiplexed in weight bank/activation bank 
    ELEMENT_COUNT: 176, # Number of parallel dot products that can be performed by one processing unit, generally equal to the number of output waveguides in a processing unit  
    UNITS_COUNT: 512, # Number of processing unit present in an accelerator
    RECONFIG: [], # Useful if the processing unit element size can be reconfigured according to the convolution computation need
    VDP_TYPE: "AMM", # More information abour vector dot product can be found in our paper ([https://ieeexplore.ieee.org/abstract/document/9852767]
    NAME: "SCONNA", # Name of the accelerator 
    ACC_TYPE: "STOCHASTIC", # Accelerator Type for example, ANALOG, STOCHASTIC accelerator
    PRECISION: 8, # The bit precision supported  by the accelerator, this value along with ***accelerator_required_precision*** determines whether bit-slicing needs to be implemented
    BITRATE: 30, # The bit rate of the accelerator 
}
]
```
### SCONNA  Accelerator
The below image shows SCONNA accelerator processing unit.
![image](https://user-images.githubusercontent.com/23030293/217599962-935aec5f-b3b9-4f99-93c9-ab83fb8de7a2.png)



### SC_ONN_Simulator Project Structure 
``` bash
├── CNN_Inference - *Contains Files Necessary for Stochastic Computing based CNN inference *
│   ├── 8_bit_MUL.zip - * Contains a CSV with values corresponding stochastic bit stream for all combination of 8 bit integers and their multiplication result
│   ├── helperfunction.py - *Accuracy Evaluation functions*
│   ├── inference.py - *Inference Evaluation Script*
│   ├── sc_conv.py - *Convolution layer performing stochastic computing based convolutions with the help of look up table*
│   ├── sc_linear.py - *Convolution layer performing stochastic computing based convolutions with the help of look up table*
│   └── sc_lut.npy - *Stochastic Computing based look up table*
├── CNNModels - *Folder contains various CNN models available for performing simulations.
│   ├── DenseNet121.csv
│   ├── GoogLeNet.csv
│   ├── Inception_V3.csv
│   ├── MobileNet_V2.csv
│   ├── ResNet50.csv
│   ├── Sample
│   ├── ShuffleNet_V2.csv
│   ├── VGG16.csv
│   └── VGG19.csv
├── constants.py
├── Controller - *This contains the logic for scheduling the convolutions and corresponding dot product operations on to the accelerator hardware*
│   │   controller.py
│   ├── controller.py
│   └── __init__.py
├── Exceptions - *Accelerator Custom Exceptions*
│   └── AcceleratorExceptions.py
├── Hardware - *Different classes corresponding to the accelerator*
│   ├── Accelerator.py
│   ├── Accumulator_TIA.py
│   ├── Activation.py
│   ├── ADC.py
│   ├── Adder.py
│   ├── BtoS.py
│   ├── bus.py
│   ├── DAC.py
│   ├── eDram.py
│   ├── __init__.py
│   ├── io_interface.py
│   ├── MRR.py
│   ├── MRRVDP.py
│   ├── PD.py
│   ├── Pheripheral.py
│   ├── Pool.py
│   ├── router.py
│   ├── Serializer.py
│   ├── stochastic_MRRVDP.py
│   ├── TIA.py
│   ├── vdpelement.py
│   └── VDP.py
├── __init__.py
├── main.py
├── PerformanceMetrics
│   └── metrics.py - *Class to calculate various peformance metrics like FPS, FPS/W and FPS/W/mm2*
├── Plots 
│   ├── _Area(mm^2)_.png
│   ├── Area.png
│   ├── FPS.png
│   ├── _FPS_W_mm^2_.png
│   ├── FPS_W.png
│   ├── IPDPS
│   ├── Sample
│   ├── Sample.png
│   └── TPC_Utilization.png
├── projectstruct.txt
├── README.md
├── Result
│   ├── IPDPS
│   └── Sample
├── StochasticADCError.py - *ADC error calculation*
├── StochasticGateAnalysis.py -*Test the stochastic compututation versus conventional computation of operations like addition, substraction and multiplication* 
├── utils - *Stochastic Computing Utils*
│   ├── ADC.py
│   ├── modelmetrics.py
│   ├── SCONNALayers.py
│   ├── SCONNAOps.py
│   ├── SCONNAUtils.py
│   ├── UnarySimLayers.py
│   └── UnarySimUtils.py
└── visualization.py  

```

### Simulation Result CSV:
After the simulations are completed, the results are stored in the form of a csv file containing information as shown below :

![image](https://user-images.githubusercontent.com/23030293/217608492-74183454-d00c-4ccd-863e-359491f6c367.png)

The performance metrics are calculated by using PeformanceMetrics/metrics.py, currently it provides the above values. Users can change the file to reflect their accelerator components energy and power parameters.  

### Evaluation Visualization:
The visualization.py can take the generated simulation csv and plot barplot for the results. It also prints useful information in the console about the top two accelerators. 
![Result](https://user-images.githubusercontent.com/23030293/217608960-0cdc3c7e-abe6-4f8e-8ee9-6cab53dcdf8d.png)


Simulation Results Analysis: 
```bash
The accelerator SCONNA achieves 1.0x times better fps than SCONNA
The accelerator SCONNA achieves 66.56219184599667x times better fps than MAM (HOLYLIGHT [6])
The accelerator SCONNA achieves 146.4465119184254x times better fps than AMM (DEAPCNN [8])
Details of second best accelerator
The accelerator MAM (HOLYLIGHT [6]) achieves 1.0x times better fps than MAM (HOLYLIGHT [6])
The accelerator MAM (HOLYLIGHT [6]) achieves 2.200145575993879x times better fps than AMM (DEAPCNN [8])


```
### Device Level Simulations:

Please refer this repository for device level simulation files: https://github.com/uky-UCAT/MRR-PEOLG

### Accuracy Drop Estimation
To evaluate accuracy drop due to ADC error in SCONNA, we modified the PyTorch quantized conv and linear modules at torch/ao/nn/quantized/ to introduce error during the forward pass of CNN inference
We follow the steps recommended by the community of PyTorch Forum to introduce error : https://discuss.pytorch.org/t/adding-an-offset-to-qint8-tensor/164846
Note: Below steps are only applicable to 8-bit integer quantized CNNs and for SCONNA evalautions we use pre-trained 8-bit integer quantized CNNs from TorchVision 
Example of how linear layer forward method is updated to introduce error:
```bash
error = 0.001
output = torch.ops.quantized.linear(x, self._packed_params._packed_params, self.scale, self.zero_point)
output = torch.quantize_per_tensor(output.dequantize(), output.q_scale()*(1+error), output.q_zero_point(), output.dtype)
return output
```




