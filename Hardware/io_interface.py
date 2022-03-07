class IOInterface:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 0.0244e-3
        self.energy = 0  # todo find the io interface energy
        self.latency = 0.78e-9 # * from holylight 1 Cycle for ADC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = None
        # * Each MRR has a dedicated ADC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0 # todo also figure out how many times it will be called 