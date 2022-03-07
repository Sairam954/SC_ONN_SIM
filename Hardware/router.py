class Router:
    
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 42e-3
        self.energy = 9.911412e-3
        self.latency = None # todo find the latency of the router
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0
    