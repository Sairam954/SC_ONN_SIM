from utils.SCONNAOps import (
    countOneInBS,
    stochasticMUL,
    stochasticADD,
    stochasticSUB,
)
from utils.SCONNAUtils import *
import itertools
from itertools import permutations
import numpy as np
import pandas as pd

""" Summary of this Code
The purpose of this code is to test the stochastic compututation versus conventional computation of operations like addition, substraction and multiplication 
1. It will accept binary bitwidth
2. For the bitwidth find all the combination of operand_1 and operand_2 that are possible. 
For example, for bitwidth = 2, the possible binary values are [0,1,2,3] where max value is 2**(bitwidth)-1. and
the possible (operand_1, operand_2) combinations : [0,0], [0,1],[0,2],[0,3], [1,0], [1,1] etc
3. After generating these combinations, depending on the operation being tested find the output value in binary which are true outputs
For example, if the operation is ADD for a combination such as (2,1) -> result = 3, similarly for all combinations and all operations
4. Convert all the combinations of binary values into stochastic bitstreams depending on the operation
5. Perform all operations in stochastic domain for each stochastic bitstream combinations
6. Find the result of the operation obtained in stochastic domain 
"""


bitwidth = 8
operand1_range = torch.arange(0, 2**bitwidth)
operand2_range = torch.arange(0, 2**bitwidth)
operation = "MUL"


result_list = []
for operand1 in operand1_range:
    for operand2 in operand2_range:
        result = {}
        # * Binary output computation
        if operation == "MUL":
            output = operand1 * operand2
        elif operation == "SUB":
            output = abs(operand1 - operand2)
        elif operation == "ADD":
            output = operand1 + operand2
        else:
            print("Operation Not yet supported")
        # * Stochastic Ouput computation

        if operation == "MUL":
            operand1_sc = getDecimalToUnary(operand1, bitwidth)
            operand2_sc = getDecimalToUnaryMul(operand2, bitwidth)
            output_sc_bs = stochasticMUL(operand1_sc, operand2_sc)
        elif operation == "SUB":
            operand1_sc = getDecimalToUnary(operand1, bitwidth)
            operand2_sc = getDecimalToUnary(operand2, bitwidth)
            output_sc_bs = stochasticSUB(operand1_sc, operand2_sc)
        elif operation == "ADD":
            operand1_sc = getDecimalToUnary(operand1, bitwidth + 1)
            operand2_sc = getDecimalToUnaryADD(operand2, bitwidth)
            output_sc_bs = stochasticADD(operand1_sc, operand2_sc)
        else:
            print("Operation Not yet supported")
        output_sc = countOneInBS(output_sc_bs)
        print("Decimal Operand 1 ", operand1)
        print("Decimal Operand 2 ", operand2)
        # print("Decimal Output ", output)
        # print("Stochastic Operand 1 :", operand1_sc)
        # print("Stochastic Operand 2 :", operand2_sc)
        # print("Stochastic Output :", output_sc)
        result["Decimal_OP1"] = operand1.numpy()
        result["Decimal_OP2"] = operand2.numpy()
        result["Decimal_OPT"] = output.numpy()
        result["SC_OP1"] = operand1_sc.numpy()
        result["SC_OP2"] = operand2_sc.numpy()
        result["SC_OPT_BS"] = output_sc_bs.numpy()
        result["SC_OPT"] = output_sc.numpy()
        result_list.append(result)
resultDf = pd.DataFrame(result_list)
fileName = str(bitwidth) + "_bit_" + operation + ".csv"
resultDf.to_csv("results/" + fileName, index=False)
