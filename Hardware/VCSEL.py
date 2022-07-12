
class VCSEL:
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 0.66e-3 # * from paper Efficient Hybrid Integration of Long Wavelength VCSELs on Silicon Photonic Circuits
        self.energy = 13.2e-12
        self.latency = None  # todo find the latency of the TIA element 
        self.area = None
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0