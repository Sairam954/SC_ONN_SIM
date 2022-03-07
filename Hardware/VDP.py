import numpy as np
from abc import ABC,abstractmethod
class VDP(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]

    Returns:
        [type]: [description]
    """
   
    def __init__(self,elements_count,vdp_type) -> None:
        self.elements_count = elements_count
        self.vdp_type = vdp_type
        self.start_time = 0
        self.end_time = 0
        
    @abstractmethod  
    def set_vdp_latency(self,):
        pass
    @abstractmethod 
    def add_vdp_element(self,):
        pass
    @abstractmethod 
    def add_vdp_element_list(self,):
        pass
    def __str__(self) -> str:
        return "Element Size :" + self.element_size.__str__() + " Elements Count :"+self.elements_count.__str__() +" Element Type "+self.vdp_type.__str__() 
    
    