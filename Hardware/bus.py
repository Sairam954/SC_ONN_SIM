class Bus:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 7e-3
        self.energy = 0 # todo need to find this value
        self.latency = None
        self.area = None
        # * Each MRR has a dedicated ADC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0 # todo need to know how this is incremented