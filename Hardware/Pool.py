
from Exceptions.AcceleratorExceptions import AdderException
from Hardware.Pheripheral import Pheripheral

import math 

class Pool(Pheripheral):
    
    def __init__(self):
        self.no_of_parallel_requests = 16
        self.power = 0.4*1e-3
        self.energy = 0
        self.latency = 3.125*1e-9
        self.area = None
        self.request_queue = self.no_of_parallel_requests
        self.waiting_queue = 0
        self.start_time = 0
        self.end_time = 0
    
    def reset(self):
        self.start_time = 0
        self.end_time = 0
        self.waiting_queue = 0
        self.request_queue = 0
    
    def controller(self,clock):
        if clock>= self.end_time:
            # print("Pooling unit is available for operations")
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
            # print("Pooling unit is busy for operations")
            pass
    
    # todo : 1 refactor the adder name to pool 2: Better way of dealing the boiler plate code in Adder and Pool
    def get_request_latency(self,request_count):
        """[It takes the no of Pooling request and returns the latency required by adder to complete themselves]

        Args:
            request_count ([type]): [description]
        """
        # print("Pooling Request Count :", request_count)
        # print("Request Queue ", self.request_queue)
        # print("Waiting Queue ", self.waiting_queue)
        if self.request_queue !=0 and self.waiting_queue > 0:
            raise AdderException("Something is wrong with this controller check it ")
        
        if self.request_queue>0:
            if self.request_queue<request_count:
                request_count = request_count - self.request_queue
                self.request_queue = 0
                self.waiting_queue = self.waiting_queue+request_count
                adder_clock_required = 2
            else:
                self.request_queue = self.request_queue- request_count
                adder_clock_required = 1
                
        else:
            self.waiting_queue= self.waiting_queue+request_count
            adder_clock_required = math.ceil(self.waiting_queue/self.no_of_parallel_requests)
        # print("Pooling Clocks required ",adder_clock_required)
        return self.latency*adder_clock_required
    
    def get_waiting_list_latency(self):
        adder_clock_required = math.ceil(self.waiting_queue/self.no_of_parallel_requests)
        # print("Pooling Clocks required ",adder_clock_required)
        return self.latency*adder_clock_required