
from Exceptions.AcceleratorExceptions import AdderException
from Hardware.Pheripheral import Pheripheral

import math
import logging as logging

logger = logging.getLogger("__Adder__")
logger.setLevel(logging.INFO)
 

class Adder(Pheripheral):
    
    def __init__(self):
        self.no_of_parallel_requests = 128
        self.power = 0.05*1e-3
        self.energy = 1.79e-9
        self.latency = 3.125*1e-9
        self.area = None
        self.request_queue = self.no_of_parallel_requests
        self.waiting_queue = 0
        self.start_time = 0
        self.end_time = 0
    # *Request Queue working => a kernel size is more than vdp element size 
    # *it will decomposed to multiple vdp elements and
    # *then the vdp unit will check the request queue of adder 
    # *if its available end time of vdp is equal to latencyt of adder + vdp end time 
    # *if queue is not available end time of vdp is equal to 
    # *len(waiting_queue)/(no_of_parallel_requests) times latency and add the vdp requests to the waiting queue 
    # *when ever clock is greater than end time 
    # *check waiting queue - if empty make request queue empty
    # *else add waiting queue to the request queue
    # *define a controller for adder which is called a every cycle to perform above operations
    def reset(self):
        self.start_time = 0
        self.end_time = 0
        self.waiting_queue = 0
        self.request_queue = 0
    
    def controller(self,clock):
        if clock>= self.end_time:
            # print("Adder is available for operations")
            self.start_time = clock
            self.end_time = clock+self.latency
            self.request_queue = self.no_of_parallel_requests
            if self.waiting_queue >= self.request_queue:
                self.waiting_queue = self.waiting_queue - self.request_queue
                self.request_queue = 0
            else:
                self.request_queue = self.request_queue-self.waiting_queue
                self.waiting_queue = 0
        else:
            pass
            # print("Adder is busy for operations")
    
      
    def get_request_latency(self,request_count):
        """[It takes the no of partial sum request and returns the latency required by adder to complete themselves]

        Args:
            request_count ([type]): [description]
        """
        # print("Partial Sum Request Count :", request_count)
        # print("Request Queue ", self.request_queue)
        # print("Waiting Queue ", self.waiting_queue)
        if self.request_queue !=0 and self.waiting_queue > 0:
            raise AdderException("Something is wrong with this controller check it ")
        
        if self.request_queue>0:
            if self.request_queue<request_count:
                request_count =request_count - self.request_queue
                self.request_queue = 0
                self.waiting_queue = self.waiting_queue+request_count
                adder_clock_required = 1
            else:
                self.request_queue = self.request_queue- request_count
                adder_clock_required = 1
                
        else:
            self.waiting_queue= self.waiting_queue+request_count
            adder_clock_required = math.ceil(self.waiting_queue/self.no_of_parallel_requests)
        # print("Adder Clocks required ",adder_clock_required)
        return self.latency*adder_clock_required
    
    def get_waiting_list_latency(self):
        adder_clock_required = math.ceil(self.waiting_queue/self.no_of_parallel_requests)
        # print("Adder Clocks required ",adder_clock_required)
        return self.latency*adder_clock_required