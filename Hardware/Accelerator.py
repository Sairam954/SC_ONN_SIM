from Exceptions.AcceleratorExceptions import AcceleratorException
from Hardware.VDP import VDP
from Hardware.Pheripheral import Pheripheral
import math

import logging as logging

logger = logging.getLogger("__Accelerator__")
logger.setLevel(logging.INFO)


class Accelerator:
    
    def __init__(self):
        self.vdp_units_list=[]
        self.vdp_units_count= 0
        self.pheripherals = []
        self.is_hybrid = False
        self.vdp_element_sizes = set()
        self.pheripherals = {}
        self.acc_type = 'ANALOG'
        self.psum_writes = 0
        self.psum_reads = 0
        self.cache_size_per_element = math.ceil((1*1024*8)/4) # 1kb per VDP element with 4 bits per word 
        self.cache_size = 0
        self.cache_reads = 0
        self.cache_writes = 0
    
    def set_vdp_type(self,vdp_type):
        self.vdp_type = vdp_type
    
    def set_acc_type(self,acc_type):
        self.acc_type = acc_type
    
    def set_cache_size(self,cache_size):
        self.cache_size = cache_size
    
    def add_vdp(self,vdp):
        """[summary]

        Args:
            vdp ([type]): [description]

        Raises:
            TypeError: [description]
        """
        if not isinstance(vdp,VDP):
            raise AcceleratorException(f"Object should be of type VDP Class")
        self.vdp_units_list.append(vdp)
        self.vdp_element_sizes.add(vdp.vdp_element_list[0].element_size)
        if len(self.vdp_element_sizes)>1:
            self.is_hybrid = True
        self.vdp_units_count+=1
    def add_pheripheral(self,name,pheripheral):
        
        if not isinstance(pheripheral,Pheripheral):
            raise AcceleratorException(f"Object should be of type Pheripheral Sub Class")
        self.pheripherals[name]=pheripheral
    
    # * need to reset accelerator before each layer
    def reset(self):
        for vdp in self.vdp_units_list:
            vdp.reset()
        for pheripheral in self.pheripherals.keys():
            self.pheripherals[pheripheral].reset()
