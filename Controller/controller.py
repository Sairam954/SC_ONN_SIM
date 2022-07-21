from Exceptions.AcceleratorExceptions import VDPElementException
import math
import numpy as np

class Controller:
    
    # Todo need to intialize properties related to the controller 
    def __init__(self):
        self.utilized_rings = np.uint64(0)
        self.idle_rings = np.uint64(0)
    
    def get_channel_latency(self,accelerator,channels,convolutions_per_channel,kernel_size):
        total_latency = 0
        for channel in range(channels):
            total_latency += self.get_convolution_latency(accelerator,convolutions_per_channel,kernel_size)
            print(total_latency)
        return total_latency
    
    def get_partial_convolution_latency(self,clock,clock_increment,accelerator,partial_convolutions,kernel_size):
        """ This method is to perform convolution of vdp unit which cannot perform a single convolution operation even after decomposition
            This method thus calculate the latency by updating the clock, the partial convolution input will be always have kernel size equivalent to kernel
            size equivalent to vdp unit size. 

        Args:
            clock ([type]): [description]
            clock_increment ([type]): [description]
            accelerator ([type]): [description]
            partial_convolutions ([type]): [description]
            kernel_size ([type]): [description]

        Returns:
            [type]: [description]
        """
        ZERO = 0
        LAST = -1 
        ADDER = "adder"
        UTILIZED_RINGS = "utilized_rings"
        IDLE_RINGS = "idle_rings"
        cycle = 0
        completed_layer = False
        # print("Partial Sum Convolution")
        while clock>=0:
            vdp_no = 0
            for vdp in accelerator.vdp_units_list:
                # print('VDP Number ', vdp_no)
                # print("VDP End Time: ", vdp.end_time)
                # print(" Supported Layers ",vdp.layer_supported)
                if vdp.end_time <= clock:
                    # print("VDP unit Available Vdp No ", vdp_no)
                    vdp.start_time = clock
                    vdp.end_time = clock+vdp.latency
                    vdp.calls_count +=1
                    vdpelement = vdp.vdp_element_list[ZERO]
                    vdp_convo_count = 0
                    
                    # * element convo count contains the value of number of kernel size convo performed by vdp for reconfig it can be greater than one
                    try:
                        element_convo_count = vdpelement.perform_convo_count(kernel_size)
                    # * This situation arises in hybrid model where VDP element size is not constant
                    #* IF the element size are say 15 and 10. Due to large VDP size partial sum is needed then it is decomposed if it is decomposed based on 15
                    #* then partial sum is divided according to the largere and the smaller vdo element cannot perform this operation.
                    #* Have tried to 
                    except Exception as error:
                        # print("This VDP cannot process this element size")
                        vdp.end_time = clock
                        element_convo_count = 0
                    vdp_convo_count = element_convo_count*vdp.get_element_count()
                    # print("VDP Convolution Count",vdp_convo_count)
                    partial_convolutions = partial_convolutions-vdp_convo_count
                    # *  AMM has array of Weight and Input so 2 + element size to represent the rings in the input WDM mux
                    vdp_mrr_utiliz = vdp.get_utilized_idle_rings_convo(element_convo_count,kernel_size,vdpelement.element_size)
                    self.utilized_rings += vdp_mrr_utiliz[UTILIZED_RINGS]
                    self.idle_rings += vdp_mrr_utiliz[IDLE_RINGS]
                        # print("Utilized Rings :", self.utilized_rings)
                        # print("Idle Rings :",self.idle_rings)   
                    # print("--------------Partial Convolutions Left-------------------", partial_convolutions)  
                    if partial_convolutions <= 0:
                        completed_layer=True
                        # print("************Partial convolutions Completed****************",partial_convolutions)
                        break  
                else:
                    # print("VDP Unit Unavailable VDP NO:",vdp_no) 
                    pass
                if completed_layer:
                    break
                vdp_no += 1 
            if completed_layer:
                break
            cycle+=1
            # print('partial cycle', cycle)
            clock=clock+clock_increment  
            # print('Clock', clock)
        return clock,accelerator
    def  get_convolution_latency(self, accelerator, convolutions, kernel_size):
        """[  Function has to give the latency taken by the given accelerator to perform stated counvolutions with mentioned kernel size
        ]

        Args:
            accelerator ([Hardware.Accelerator]): [Accelerator for performing the convolutions]
            convolution_count ([type]): [No of convolutions to be performed by the accelerator]
            kernel_size ([type]): [size of the convolution]
            
        Returns:
            [float]: [returns the latency required by the accelerator to perform all the convolutions]
        """
        ELEMENT_SIZE = 'element_size'
        ELEMENT_COUNT = 'element_count'
        UNITS_COUNT = 'units_count'
        RECONFIG = "reconfig"
        ZERO = 0
        LAST = -1 
        ADDER = "adder"
        UTILIZED_RINGS = "utilized_rings"
        IDLE_RINGS = "idle_rings"
        PCA_DKV_LIMIT = 14
        clock = 0
        clock_increment = accelerator.vdp_units_list[ZERO].latency
        # print('Convolutions to be completed ', convolutions)
        # print('Clock Increment', clock_increment)
        # return 
        # if accelerator.is_hybrid:
        #     clock_increment = abs(accelerator.vdp_units_list[ZERO].latency- accelerator.vdp_units_list[LAST].latency)
        # print("Is Hybrid :", accelerator.is_hybrid)
        completed_layer = False
        cycle = 0
        
        while clock>=0:
            # print("Cycle =>",cycle)
            # print("Clock =>",clock)
            vdp_no = 0
            partial_sum_list = []
            accelerator.pheripherals[ADDER].controller(clock)
            for vdp in accelerator.vdp_units_list:
                # print("VDP End Time: ", vdp.end_time)
                # print('VDP Number ', vdp_no)
                # print('Clock ', clock)s
                # print('Vdpe end time', vdp.end_time)
                if vdp.end_time <= clock:
                    # print("VDP unit Available Vdp No ", vdp_no)
                    vdp.start_time = clock
                    vdp.end_time = clock+vdp.latency
                    vdp.calls_count +=1
                    vdpelement = vdp.vdp_element_list[ZERO]
                    vdp_convo_count = 0
                    try:
                        # print("Kernel Size :",kernel_size)
                        # * element convo count contains the value of number of kernel size convo performed by vdp for reconfig it can be greater than one
                        element_convo_count = vdpelement.perform_convo_count(kernel_size)
                        vdp_convo_count = element_convo_count*vdp.get_element_count()
                        # print("Element VDP Count",element_convo_count)
                        convolutions = convolutions-vdp_convo_count
                        
                        # *  AMM has array of Weight and Input so 2 + element size to represent the rings in the input WDM mux
                        vdp_mrr_utiliz = vdp.get_utilized_idle_rings_convo(element_convo_count,kernel_size,vdpelement.element_size)
                        self.utilized_rings += vdp_mrr_utiliz[UTILIZED_RINGS]
                        self.idle_rings += vdp_mrr_utiliz[IDLE_RINGS]
                        # print("Utilized Rings :", self.utilized_rings)
                        # print("Idle Rings :",self.idle_rings)
                    except(VDPElementException):
                        # print("Need to Decompose Kernel")
                        
                        decomposed_kernel_size = vdpelement.reconfigurable_to_element_sizes[LAST]
                        # print("Kernel Size",kernel_size)
                        # print("Decomposed Kernel Size",decomposed_kernel_size)
                        # print("VDP Element Size ", vdpelement.element_size)
                        # print("VDP Element Count ",vdp.get_element_count())
                        decomposed_kernel_count = math.ceil(kernel_size/decomposed_kernel_size)
                        # print("Decomposed Kernel Count ",decomposed_kernel_count)
                        element_convo_count = vdpelement.perform_convo_count(decomposed_kernel_size)
                        # print("VDPE Convolution Count ", element_convo_count)
                        vdp_convo_count = int((element_convo_count*vdp.get_element_count())/(decomposed_kernel_count))
                        # print("VDP Convolution Count",vdp_convo_count)
                        # * one use case that was missed while performing this logic was what to do when a single convolution can not be performed 
                        # * on one vdp unit even with kernel decomposition, in this case the partial convo gets divided into multiple vdps 
                        # * method to solve this thing you are sending creating a seperate method to perform these partial convolution
                        # * First calculate the number of partial convo   
                        if vdp_convo_count == 0:
                            # * need to distribute the convolution on to various vdp units as single vdp cannot perform 
                            # print('DKV distrubted across various VDPEs')
                            partial_convolutions = decomposed_kernel_count
                            vdp.end_time = clock # * this is to make this vdp unit also available for operation
                            clock, accelerator = self.get_partial_convolution_latency(clock,clock_increment,accelerator,partial_convolutions,decomposed_kernel_size)

                            vdp_convo_count = 1 # * since we are calculating the time taken to perform partial convo count of single convo
                            # * Substract this vdp utilization as they are already taken care in partial convolution latency calculation
                            vdp_mrr_utiliz = vdp.get_utilized_idle_rings_convo(element_convo_count,decomposed_kernel_size,vdpelement.element_size)
                            self.utilized_rings += vdp_mrr_utiliz[UTILIZED_RINGS]
                            self.idle_rings += vdp_mrr_utiliz[IDLE_RINGS]    
                        vdp_mrr_utiliz = vdp.get_utilized_idle_rings_convo(element_convo_count,decomposed_kernel_size,vdpelement.element_size)
                        self.utilized_rings += vdp_mrr_utiliz[UTILIZED_RINGS]
                        self.idle_rings += vdp_mrr_utiliz[IDLE_RINGS]
                        # print("Utilized Rings :", self.utilized_rings)
                        # print("Idle Rings :",self.idle_rings)
                        convolutions = convolutions-vdp_convo_count
                        # * Sceduling of partial sum request and updating convolution latency 
                        if accelerator.acc_type != 'ONNA':
                            partial_sum_latency = accelerator.pheripherals[ADDER].get_request_latency(decomposed_kernel_count)
                        else: 
                            required_precision = 20
                            # print('Decomposed Kernel ', decomposed_kernel_count)
                            # print('Batch of VDPs accumulated ', math.floor(decomposed_kernel_count/(PCA_DKV_LIMIT*(2**required_precision))) )
                            if (math.floor(decomposed_kernel_count/PCA_DKV_LIMIT))>1:
                                partial_sum_latency = accelerator.pheripherals[ADDER].get_request_latency(math.floor(decomposed_kernel_count/PCA_DKV_LIMIT))
                                # print('Partial Sum Latency',partial_sum_latency)
                            else:
                                partial_sum_latency = 0 
                                # print('Partial Sum Latency',partial_sum_latency)
                        vdp.end_time = vdp.end_time + partial_sum_latency
                    if convolutions <= 0:
                        completed_layer=True
                        # print("************Convolutions Completed****************",convolutions)
                        break   
                    
                    # print("Convolutions Left :", convolutions)
                   
                else:
                    pass
                    # print("VDP Unit Unavailable VDP NO:",vdp_no)
               
                if completed_layer:
                    break
                vdp_no += 1
            # print("Convolutions Left :", convolutions)
           
            if completed_layer:
                break
            cycle+=1
        
            clock=clock+clock_increment
        # print('Conv Latency', clock)
        # print('PSum Latency', accelerator.pheripherals[ADDER].get_waiting_list_latency())   
        clock = clock + accelerator.pheripherals[ADDER].get_waiting_list_latency()
        # print('Clock', clock)
        accelerator.pheripherals[ADDER].reset()
        return clock 
    #* the below method is not need in version 2
    def get_fully_connected_latency(self,accelerator, fully_connected_dp):
        
        ELEMENT_SIZE = 'element_size'
        ELEMENT_COUNT = 'element_count'
        UNITS_COUNT = 'units_count'
        RECONFIG = "reconfig"
        ZERO = 0
        LAST = -1 
        ADDER = "adder"
        completed_layer = False
        cycle = 0
        clock = 0
        clock_increment = accelerator.vdp_units_list[ZERO].latency
        if accelerator.is_hybrid:
            clock_increment = abs(accelerator.vdp_units_list[ZERO].latency- accelerator.vdp_units_list[LAST].latency)
        # print(clock_increment)
         
        while fully_connected_dp>0:
            vdp_no = 0
            accelerator.pheripherals[ADDER].controller(clock)
            for vdp in accelerator.vdp_units_list:
                # print("VDP End Time: ", vdp.end_time)
                if vdp.end_time <= clock and vdp.does_support_layer("inner_product"):
                    # print("VDP unit Available Vdp No ", vdp_no)
                    vdp.start_time = clock
                    vdp.end_time = clock+vdp.latency
                    vdp.calls_count +=1
                    vdpelement = vdp.vdp_element_list[ZERO]
                    vdp_fc_count = 0
                    
                    element_fc_count = vdpelement.element_size
                    vdp_fc_count = element_fc_count*vdp.get_element_count()
                    # * for AMM architecture no of rings in VDP unit (element_size + element_count*(element_size)*2)
                    self.utilized_rings += (2*vdp_fc_count + vdpelement.element_size)
                    
                    # print("Utilized Rings :",self.utilized_rings)
                    
                    # print("VDP FC Count",vdp_fc_count)
                    fully_connected_dp = fully_connected_dp-vdp_fc_count
                    if vdp_fc_count>fully_connected_dp:
                        self.idle_rings += (vdp_fc_count-fully_connected_dp)*2
                        
                    # print("Idle Rings :", self.idle_rings)
                    partial_sum_latency = accelerator.pheripherals[ADDER].get_request_latency(1)
                    vdp.end_time = vdp.end_time + partial_sum_latency
                    # print("Partial Sum latency ", partial_sum_latency)     
                    if fully_connected_dp <= 0:
                        completed_layer=True
                        self.utilized_rings+= fully_connected_dp
                        print("************Fully Connected Completed****************",fully_connected_dp)
                        break   
                    
                   
                    
                else:
                    pass
                    # print("VDP Unit Unavailable VDP NO:",vdp_no)
                vdp_no += 1
                if completed_layer:
                    break
            clock=clock+clock_increment
            # print("Fully Connected Left", fully_connected_dp)
            if completed_layer:
                    break
            cycle+=1
                
            
        clock = clock + accelerator.pheripherals[ADDER].get_waiting_list_latency()
        accelerator.pheripherals[ADDER].reset()
        # print('Latency', clock)
        # print('Cycle', cycle)
        return clock
    
