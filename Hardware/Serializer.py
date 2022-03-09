from Hardware.Pheripheral import Pheripheral


class Serializer(Pheripheral):
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 0.0025
        self.energy = 0.04*1e-12 #! need to find the energy of this unit 
        self.latency = 0 
        self.area = 5.9*1e-3 #* Taked from proteus Section 5-A) 1
        # * Each MRR has a dedicated Serializer so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0