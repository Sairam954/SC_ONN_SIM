import pandas as pd
from utils.ADC import *
from os import listdir
from os.path import isfile, join

""" The Summary of this Code
calculate the error between the decimal output and the adc output of stochastic gate 
    1. read the vin value of each combination
    2. send it to adc with Vref and Vin get the adc output
    3. Find the difference betweeen Decimal output and adc_output 
    Update:
    The Vin based on the observation shows an almost equivalent increment for increase in number of 1s on the stochastic output
    Therefore, scaling the Vin depending on the Vin for zero number of 1s in the stochastic output. Similarly, Vref is also obtained based on the Vin(0) and operation
    Vin(x) where x is the number of 1s in stochastic output bitstream
    For Addtion:
    Vin(0)/BaseValue = 3.1 mV, Increment_Voltage (Vinc) = 0.222mv ,  Vin(x) = x*Vinc + Vin(0), and Vref = 2**(bitwidth+1)*Vinc+Vin(0)
    For Substraction:
    Vin(0)/BaseValue = 4.1 mV, Increment_Voltage (Vinc) = 0.222mv ,  Vin(x) = x*Vinc + Vin(0), and Vref = 2**(bitwidth)*Vinc+Vin(0)
    For Multiplication:
    Vin(0)/BaseValue = 3.2 mV, Increment_Voltage (Vinc) = 0.222mv ,  Vin(x) = x*Vinc + Vin(0), and Vref = 2**(2*bitwidth)*Vinc+Vin(0)     
"""

PCAResultDir = "results/PCAResult/"
GateErrorDir = "results/GateError/"
ErrorMetricsDir = "results/ErrorMetrics/"
pcaFiles = [f for f in listdir(PCAResultDir) if isfile(join(PCAResultDir, f))]
errorMetricsList = []
amplification = True  # This is to amplify the output of stochastic multiplier by 2**N
for fileName in pcaFiles:
    print("Processing File Name :", fileName)
    result = pd.read_csv(PCAResultDir + fileName)
    fileName = fileName.replace(".csv", "")
    fileNameTokens = fileName.split("_")
    bitwidth = int(fileNameTokens[0])
    operation = fileNameTokens[2]
    print("Bitwidth :", bitwidth)
    print("Operation :", operation)
    # * Currently running from SUB 3 bit
    if bitwidth == 4 and operation == "MUL":

        if operation == "MUL":
            Vin_0 = 338.69  # Voltage at the PCA when output stochastic bitstream has 0 number of 1s
            Vinc = 0.222  # Incremental voltage for increase in number of 1s by each 1
            Vref = 2 ** (2 * bitwidth) * Vinc + Vin_0 - Vin_0
            result["SC_OPT"] = result["SC_OPT"] * (2**bitwidth)
            Vin = result["SC_OPT"] * Vinc + Vin_0 - Vin_0
            result["Vin"] = Vin
        elif operation == "SUB":
            Vin_0 = 608.45  # Voltage at the PCA when output stochastic bitstream has 0 number of 1s
            Vinc = 0.22  # Incremental voltage for increase in number of 1s by each 1
            Vref = 2 ** (bitwidth) * Vinc + Vin_0 - Vin_0
            Vin = result["SC_OPT"] * Vinc + Vin_0 - Vin_0
            result["Vin"] = Vin
        elif operation == "ADD":
            Vin_0 = 957.8  # Voltage at the PCA when output stochastic bitstream has 0 number of 1s
            Vinc = 0.22  # Incremental voltage for increase in number of 1s by each 1
            Vref = 2 ** (bitwidth + 1) * Vinc + Vin_0 - Vin_0
            Vin = result["SC_OPT"] * Vinc + Vin_0 - Vin_0
            result["Vin"] = Vin
        else:
            print("Operation Not yet supported")

        adcOutput = result.apply(
            lambda row: get_adc_output(
                row["Vin"], Vref, bitwidth=bitwidth, operation=operation
            ),
            axis=1,
        )
        error = abs(result["Decimal_OPT"] - adcOutput)
        stochasticError = abs(result["Decimal_OPT"] - result["SC_OPT"])
        result["ADC_output"] = adcOutput
        result["ADC_ERROR"] = error
        result["SC_ERROR"] = stochasticError
        # print("Error Array :", error)
        # print("Average Error :", error.mean())
        # print("Standard Deviation :", error.std())
        result.to_csv(
            GateErrorDir + str(bitwidth) + "_bit_AMP_" + operation + ".csv", index=False
        )

        errorMetrics = {}
        errorMetrics["Bitwidth"] = bitwidth
        errorMetrics["Operation"] = operation
        errorMetrics["ADC_ERR_MEAN"] = error.mean()
        errorMetrics["ADC_ERR_STD"] = error.std()
        errorMetrics["SC_ERR_MEAN"] = stochasticError.mean()
        errorMetrics["SC_ERR_STD"] = stochasticError.std()
        errorMetricsList.append(errorMetrics)

errorMetricsDf = pd.DataFrame(errorMetricsList)
errorMetricsDf.to_csv(
    "results/ErrorMetrics/Stochastic_Error_Metrics_Amp_8bit.csv", index=False
)
