import torch


def stochasticMUL(operand1, operand2):
    """This method takes in two tensors which represent stochastic bitstreams.
    The bitstreams are muliplied which is equivalent to bitwise AND in stochastic domain.
    Args:
        operand1 (_type_): First Operand
        operand2 (_type_): Second Operand
    """

    result = torch.logical_and(operand1, operand2).long()

    return result


def stochasticADD(operand1, operand2):
    """
    This method takes in two tensors which represent stochastic bitstreams.
    The bitstreams are added which is equivalent to bitwise OR in stochastic domain.
    Args:
        operand1 (_type_): First Operand Stochastic Bitstream
        operand2 (_type_): Second Operand Stochastic Bitstream which is negatively correlated to the operand1

    Returns:
        _type_: _description_
    """

    result = torch.logical_or(operand1, operand2).long()

    return result


def stochasticSUB(operand1, operand2):
    """
    This method takes in two tensors which represent stochastic bitstreams.
    The bitstreams are substracted which is equivalent to bitwise XOR in stochastic domain.
    Args:
        operand1 (_type_): First Operand Stochastic Bitstream
        operand2 (_type_): Second Operand Stochastic Bitstream which is positively correlated to the operand1

    Returns:
        _type_: _description_
    """

    result = torch.logical_xor(operand1, operand2).long()

    return result


def countOneInBS(bitstream):
    """Given a stochastic bitstream, should caculate the number of 1 in the bitstream. This only works for unary coded bitstream

    Args:
        bitstream (_type_): stochastic bit stream as tensor ex [1,0,1,0,1,1]

    Returns:
        _type_: count of 1 in the bitstream tensor
    """
    numOfOnes = torch.count_nonzero(bitstream)
    return numOfOnes


# #! Test Stochastic Multiplication
# Operand_1 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
# Operand_2 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1])
# stochastic_output = stochasticMUL(Operand_1, Operand_2)
# print("Multiplication :", countOneInBS(stochastic_output))


# #! Test Stochastic Substraction
# Operand_1 = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
# Operand_2 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
# stochastic_output = stochasticSUB(Operand_1, Operand_2)
# print("Substraction :", countOneInBS(stochastic_output))


# #! Test Stochastic ADD
# Operand_1 = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Operand_2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
# stochastic_output = stochasticADD(Operand_1, Operand_2)
# print("Addition :", countOneInBS(stochastic_output))
