B_ONNSIM (BINARY Optical Neural Network Simulator)

``` bash
C:.
│   .gitignore
│   constants.py
│   main.py - Runs the simulator and allows users to change the inputs according to the accelerator 
│   README.md
│   requirements.txt
│   visualization.py - Plots the performance metrics like FPS, FPS/W etc of various accelerators on single barplot and also provides information of the best performing accelerator 
│   __init__.py
│
├───.vscode
│       launch.json
│       settings.json
│
├───CNNModels - Folder contains various CNN models available for performing simulations
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
├───Controller - This contains the logic for scheduling the convolutions and corresponding dot product operations on to the accelerator hardware
│   │   controller.py
│   │   __init__.py
│   │
│   └───__pycache__
│           controller.cpython-310.pyc
│           controller.cpython-38.pyc
│           __init__.cpython-310.pyc
│           __init__.cpython-38.pyc
│
├───Exceptions - Accelerator Custom Exceptions
│   │   AcceleratorExceptions.py
│   │
│   └───__pycache__
│           AcceleratorExceptions.cpython-310.pyc
│           AcceleratorExceptions.cpython-38.pyc
│
├───Hardware - Different classes corresponding to th
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
│   │   metrics.py - Class to calculate various peformance metrics like FPS, FPS/W and FPS/W/mm2
│ 
│
├───Plots - Folder containing the plots produced by visualization.py
│   ├───ISQED
│   │       FPS_(Log_Scale).png
│   │
│   └───Sample
├───Result
│   └───ISQED - Simulation Result of various Binary Neural Network Accelerator
│           LIGHTBULB_All.csv
│           OXBNN_50_ALL.csv
│           OXBNN_5_ALL.csv
│           ROBIN_EO_All.csv
│           ROBIN_PO_All.csv
│           Vis_Test.csv
...
