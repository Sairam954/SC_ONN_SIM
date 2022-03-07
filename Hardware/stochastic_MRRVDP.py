import numpy as np
from Exceptions.AcceleratorExceptions import VDPElementException, VDPException

from Hardware.VDP import VDP
from Hardware.vdpelement import VDPElement
from constants import *
import logging as logging

logger = logging.getLogger("__MRR_VDP__")
logger.setLevel(logging.INFO)


class Stocastic_MRRVDP(VDP):
    """ This class is to define Stochastic based MRR VDP unit that performs actions in the digital domain. MRR can be used to perform AND operation in digital domain.

    Args:
        VDP ([type]): [description]

    Raises:
        VDPException: [description]
        VDPElementException: [description]

    Returns:
        [type]: [description]
    """
    
    def __init__(self,ring_radius,pitch,vdp_type, supported_layer_list = []) -> None:
        self.ring_radius = ring_radius
        self.pitch = pitch
        self.vdp_type = vdp_type
        self.start_time = 0
        self.end_time = 0 
        self.vdp_element_list = []  
        self.br = 50*1e9 #* Bit rate is defined in BPS which defines the latency of the VDP unit in digital domain
        self.pheripheral_latency = 4.68e-9
        self.calls_count = 0
        # * other latency is from HolyLight clock 1.28GHz => 1 cycle ADC, 1 cycle DRAM, 2 cycle S_A, 1 cycle Activation and 1 cycle writeback => 6 cycle*78.125ns  
        self.layer_supported = LAYERS_SUPPORTED # * the supported layers by a vdp element by default are set to convolution and inner_product(fc)
        if supported_layer_list:
            self.set_supported_layers(supported_layer_list)

    def set_supported_layers(self,layerList):
        self.layer_supported = []
        for layer in layerList:
            self.layer_supported.append(layer)      
    
    def does_support_layer(self,layer_name):
        if layer_name in self.layer_supported:
            return True
        else:
            return False
    
    def set_vdp_latency(self)-> float:
        self.vdp_element_list.sort(key= lambda vdp_element:vdp_element.element_size, reverse=True)
        if self.vdp_type == 'AMM':
            distance = self.vdp_element_list[0].element_size*(2*np.pi*self.ring_radius+self.pitch)
            self.prop_latency = distance/(3e8)
            self.mod_latency = (2**self.vdp_element_list[0].precision)/self.br
            self.latency = self.prop_latency +self.mod_latency+self.pheripheral_latency
            return self.latency
        else:
            raise VDPException('The latency calculation for specified type is not supported')
    
    def add_vdp_element(self,vdp_element):
        if not isinstance(vdp_element,VDPElement):
            raise VDPElementException("The element should be a class VDP element")
        self.vdp_element_list.append(vdp_element)
    def get_vdp_element_reconfig_sizes(self):
        return self.vdp_element_list[0].reconfigurable_to_element_sizes[0]
            
    def add_vdp_element_list(self,vdp_element_list):
        self.vdp_element_list = vdp_element_list
    
    def get_element_count(self)->int:
        return len(self.vdp_element_list)
    
    # * Aggregation using mux and splitting user splitter MRR are not including the calculations
    def get_utilized_idle_rings_convo(self,element_convo_count,kernel_size,element_size):
        no_of_comb_switches = 0
        no_of_used_comb_switches = 0
        total_vdp_mrr = 0
        reconfig_sizes = self.get_vdp_element_reconfig_sizes()
        # print(reconfig_sizes)
        if isinstance(reconfig_sizes,list):
            no_of_comb_switches_per_element = 0
            for re_size in reconfig_sizes:
                no_of_comb_switches_per_element += int(element_size/re_size)*2
            # print('No of comb switches per element', no_of_comb_switches_per_element)
            no_of_comb_switches = no_of_comb_switches_per_element*self.get_element_count()
    
        # print("Reconfig Sizes :",reconfig_sizes)
        # print("Element Convo Count :", element_convo_count)
        if element_convo_count>1:
            no_of_used_comb_switches = self.get_element_count()*int(element_size/kernel_size)*2 
            # print('No of utilized comb switches',no_of_used_comb_switches)       
        if self.vdp_type == "AMM":
            total_vdp_mrr = self.get_element_count()*(element_size*2)+no_of_comb_switches  
            utilized_rings = element_convo_count*(kernel_size*2)*self.get_element_count()+no_of_used_comb_switches
            idle_rings = total_vdp_mrr - utilized_rings
        # * Need to evaluate the utilization for the  MAM more accurately
        if self.vdp_type == "MAM":
            total_vdp_mrr = ((self.get_element_count()*element_size)+no_of_comb_switches)+element_size
            utilized_rings = element_convo_count*kernel_size*self.get_element_count()+element_size
            idle_rings = total_vdp_mrr-utilized_rings    
        # print("No of Comb Switches", no_of_comb_switches)
        # print("No of used comb switches", no_of_used_comb_switches)
        return {"utilized_rings":utilized_rings,"idle_rings":idle_rings}
    
    # todo utilization funtion for fc now its directly in controller logic 
    
    
    def __str__(self) -> str:
        
        return " Elements Count :"+str(self.get_element_count()) +" Element Type "+self.vdp_type.__str__() 
    
    def reset(self):
        self.start_time = 0
        self.end_time = 0 