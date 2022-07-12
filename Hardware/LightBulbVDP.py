import numpy as np
from Exceptions.AcceleratorExceptions import VDPElementException, VDPException

from Hardware.VDP import VDP
from Hardware.vdpelement import VDPElement
from constants import *
import logging as logging

logger = logging.getLogger("__MRR_VDP__")
logger.setLevel(logging.INFO)


class LightBulb_VDP(VDP):
    
    """
    This represents the VDP element of Light Bulb architecture for binary neural network acceleration
    
    """
    
    