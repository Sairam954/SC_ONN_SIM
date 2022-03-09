class TIA:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 0.017 # * taken from excel scheet 
        self.energy = 0.21e-12
        self.latency = None  # todo find the latency of the TIA element 
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0