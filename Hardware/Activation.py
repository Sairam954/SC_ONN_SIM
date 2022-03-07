class Activation:
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 0.52e-3
        self.energy = 0 # todo find the energy of activation unit
        self.latency = 0.78e-9 # * from holylight 1 Cycle for activation and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0