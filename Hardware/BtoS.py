from Hardware.Pheripheral import Pheripheral


class BtoS(Pheripheral):
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 6e-5
        self.energy = 0 #! need to find the energy of this unit 
        self.latency = 0 # * from holylight 1 Cycle for ADC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = None
        # * Each MRR has a dedicated ADC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0