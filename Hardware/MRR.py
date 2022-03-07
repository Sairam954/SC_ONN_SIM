class MRR:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power_to = 1.375e-3 # thermo optic tuning power
        self.power_eo = 1.6e-6 # electro optic tuning
        self.energy = 5e-14
        self.latency = None # todo find the latency of the PD
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0 