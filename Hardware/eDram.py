class EDram:
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 41.4e-3
        self.energy = 0.0192e-9
        self.latency = 0.78e-9*2 # * from holylight 2 Cycle for eDram and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0
