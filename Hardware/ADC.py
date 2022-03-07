from Hardware.Pheripheral import Pheripheral


class ADC(Pheripheral):
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 2e-3
        self.energy = 49.7e-9
        self.latency = 0.78e-9 # * from holylight 1 Cycle for ADC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = None
        # * Each MRR has a dedicated ADC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0