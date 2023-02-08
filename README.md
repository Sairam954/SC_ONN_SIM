

# SC_ONN_SIM (Stochastic Computing Optical Neural Network Simulator)

This is a transaction-level, event-driven python-based simulator for evaluation of stochastic computing based optical neural network accelerators for various quantized Convolutional Neural Network models.  

### Installation and Execution

    git clone https://github.com/Sairam954/SC_ONN_SIM.git
    python main.py

### Bibtex

Please cite us if you use SC_ONN_SIM

```bash
@INPROCEEDINGS{SairamIPDPS2023,
  author =       {Sairam Sri Vatsavai, Venkata Sai Praneeth Karempudi, and Ishan Thakkar},
  title =        {SCONNA: A Stochastic Computing Based Optical
Accelerator for Ultra-Fast, Energy-Efficient
Inference of Integer-Quantized CNNs},
  booktitle =    {2023 International Parallel and Distributed Processing Symposium (IPDPS)}, 
  year =         {2023},
  volume =       {},
  number =       {},
  pages =        {},
}
```

### Video Tutorial
https://www.youtube.com/watch?v=X6yifdEB7xU

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
│   constants.py
│   main.py - *Runs the simulator and allows users to change the inputs according to the accelerator* 
│   README.md
│   requirements.txt
│   visualization.py - *Plots the performance metrics like FPS, FPS/W etc of various accelerators on single barplot and also provides information of the best performing accelerator* 
│   __init__.py
│
| *Script to generate model files ->(https://github.com/Sairam954/CNN_Model_Layer_Information_Generator)*
├───CNNModels - *Folder contains various CNN models available for performing simulations. 
│   │   DenseNet121.csv
│   │   GoogLeNet.csv
│   │   Inception_V3.csv
│   │   MobileNet_V2.csv
│   │   ResNet18.csv
│   │   ResNet50.csv
│   │   ShuffleNet_V2.csv
│   │   VGG-small.csv
│   │   VGG16.csv
│   │   VGG19.csv
│   │
│   └───Sample
│           ResNet50.csv
│
├───Controller - *This contains the logic for scheduling the convolutions and corresponding dot product operations on to the accelerator hardware*
│   │   controller.py
│   │   __init__.py
│   │
│   └───__pycache__
│           controller.cpython-310.pyc
│           controller.cpython-38.pyc
│           __init__.cpython-310.pyc
│           __init__.cpython-38.pyc
│
├───Exceptions - *Accelerator Custom Exceptions*
│   │   AcceleratorExceptions.py
│   │
│   └───__pycache__
│           AcceleratorExceptions.cpython-310.pyc
│           AcceleratorExceptions.cpython-38.pyc
│
├───Hardware - *Different classes corresponding to the accelerator*
│   │   Accelerator.py
│   │   Accumulator_TIA.py
│   │   Activation.py
│   │   ADC.py
│   │   Adder.py
│   │   BtoS.py
│   │   bus.py
│   │   DAC.py
│   │   eDram.py
│   │   io_interface.py
│   │   LightBulbVDP.py
│   │   MRR.py
│   │   MRRVDP.py
│   │   PD.py
│   │   Pheripheral.py
│   │   Pool.py
│   │   RobinVDP.py
│   │   router.py
│   │   Serializer.py
│   │   stochastic_MRRVDP.py
│   │   TIA.py
│   │   VCSEL.py
│   │   VDP.py
│   │   vdpelement.py
│   │   __init__.py  
│
└───PerformanceMetrics
│   │   metrics.py - *Class to calculate various peformance metrics like FPS, FPS/W and FPS/W/mm2*
│ 
│
├───Plots - *Folder containing the plots produced by visualization.py*
│   ├───ISQED
│   │       FPS_(Log_Scale).png
│   │
│   └───Sample
├───Result
│   └───ISQED - *Simulation Result of various Binary Neural Network Accelerator*
│           

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





