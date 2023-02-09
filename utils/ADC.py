import numpy as np
import pandas as pd


def get_adc_output(Vin, Vref, bitwidth, operation="ADD"):
    """This method implements a flash based ADC, it takes voltage values  Vin,Vref, bitwitdth and operation.
    The resolution needed by the adc is dependent on the stochastic operation performed
    SUB : (Bitwidth)
    ADD : (Bitwidth+1)
    MUL : (2*Bitwidth)
    Args:
        Vin (_type_): The input voltage to the ADC
        Vref (_type_): Vref is the reference voltage which is determined based on the operation
        bitwidth (_type_): The bit length of the binary values
        operation (str, optional): _description_. Defaults to "ADD". Can supports "SUB" and "MUL". "DIV" will be added in future

    Returns:
        adc_output: binary value corresponding to the Vin
    """
    voltage_divide_points = []
    comparator_states = []
    adc_output = 0
    if operation == "MUL":
        resolution = 2 * bitwidth
    elif operation == "SUB":
        resolution = bitwidth
    elif operation == "ADD":
        resolution = bitwidth + 1
    else:
        print("Operation Not yet supported")
    for i in range(2**resolution - 1):
        voltage_divide_points.append((i + 1) * Vref / (2**resolution))
    # print("Voltage Divide Points ", voltage_divide_points)
    for v in voltage_divide_points:
        if Vin >= v:
            comparator_states.append(1)
        else:
            comparator_states.append(0)
    # print("Comparator States :", comparator_states)

    comparator_idx = len(comparator_states)
    while comparator_idx >= 1:
        if comparator_states[comparator_idx - 1] > 0:
            adc_output = comparator_idx
            return adc_output
        comparator_idx = comparator_idx - 1
    return 0


# # ! Test the adc working
# Vin = 6
# Vref = 6
# bitwidths = [3, 4, 5, 8, 9]
# operations = ["ADD", "SUB", "MUL"]

# adc_output = get_adc_output(
#     Vin=Vin, Vref=Vref, bitwidth=bitwidths[3], operation=operations[1]
# )
# print("ADC Output :", adc_output)
# input = pd.read_csv("AND.csv")
# print(input.columns)
# Vin = input["Vin"]
# print(Vin)
# max_output_idx = input["Y"].idxmax()
# print("Max Output :", max_output_idx)
# Vref = Vin[max_output_idx]
# print(Vref)
# # adc_output = get_adc_output(3.2, Vref)
# # print(adc_output)
# adc_output = input.apply(lambda row: get_adc_output(row[0], Vref), axis=1)
# print(adc_output)
# error = abs(input["Y"] - adc_output)
# print("Error Array :", error)
# print("Average Error :", error.mean())
# print("Standard Deviation :", error.std())
