class PD:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 2.8e-3
        self.energy = 0.122e-12
        self.latency = None # todo find the latency of the PD
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0