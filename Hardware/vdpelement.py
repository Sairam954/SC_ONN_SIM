from Exceptions.AcceleratorExceptions import VDPElementException
import sys
sys.path.append(".")

import math
from constants import *

class VDPElement:
    
    def  __init__(self,element_size,reconfigurable_to_element_sizes=[], auto_reconfigurable = False, precision = 4):
        self.element_size = element_size
        self.reconfigurable = False
        self.precision = precision
        self.min_reconfig_size = 4
        if not auto_reconfigurable:
            self.set_reconfig_size_map(reconfigurable_to_element_sizes)
        else:
            self.set_reconfig_size_map_auto() 
            
    
    def set_reconfig_size_map_auto(self):
        self.reconfigurable_to_element_sizes = []
        self.reconfigurable_subelement_map = {}
        lowest_element_size = int(math.ceil(math.sqrt(self.min_reconfig_size)))
        highest_element_size = int(math.sqrt(self.element_size))
        reconfigurable_to_element_sizes = list(element**2 for element in range(lowest_element_size,highest_element_size+1)) 
        reconfigurable_to_element_sizes.append(self.element_size)
        for element_size in reconfigurable_to_element_sizes:
            self.reconfigurable_subelement_map[element_size] = int(self.element_size/element_size)
        self.reconfigurable_to_element_sizes = reconfigurable_to_element_sizes
        # print(self.reconfigurable_subelement_map)
    
    
    def set_reconfig_size_map(self,reconfigurable_to_element_sizes):
        self.reconfigurable_to_element_sizes = []
        self.reconfigurable_subelement_map = {}
        if len(reconfigurable_to_element_sizes)>0:
            self.reconfigurable = True
            for reconfig_size in reconfigurable_to_element_sizes:
                if self.element_size<reconfig_size:    
                    raise VDPElementException("The VDP Element size must be greater than reconfig size")
                else:
                    self.reconfigurable_to_element_sizes.append(reconfigurable_to_element_sizes)    
                    self.reconfigurable_subelement_map[reconfig_size] = int(self.element_size/reconfig_size)
            self.reconfigurable_subelement_map[self.element_size] = 1
            self.reconfigurable_to_element_sizes.append(self.element_size)
        else:
            self.reconfigurable_to_element_sizes.append(self.element_size)    
            self.reconfigurable_subelement_map[self.element_size] = 1
    def perform_convo_count(self,kernel_size):
        """[The method returns for the given kernel_size = 9,25 etc how many such operations can this vdp element perform ]

        Args:
            kernel_size ([type]): [description]

        Raises:
            VDPElementException: [description]

        Returns:
            [type]: [description]
        """
        try:
            # print(self.reconfigurable_subelement_map)
            return self.reconfigurable_subelement_map[kernel_size]
        except KeyError:
            if kernel_size>self.element_size:
                raise VDPElementException("Cannot Map the Kernel Size to VDP directly please decompose")
            else:
                # print("Should have been here")
                # print(self.reconfigurable_subelement_map)
                return self.reconfigurable_subelement_map[self.element_size]
    
    def get_utilized_rings(self,kernel_size):
        
        return self.element_size-self.perform_convo_count(kernel_size)*kernel_size
       
    def __str__(self):
        return "Element_Size :"+self.element_size.__str__()+" Reconfigurable :"+self.reconfigurable.__str__()+" Reconfig Map :"+str(self.reconfigurable_subelement_map)