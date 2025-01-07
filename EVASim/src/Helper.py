import time

class Helper:
    def init(self):
        self.start = 0
        self.end = 0
        
    def set_timer(self):
        self.start = time.perf_counter()
    
    def end_timer(self, task):
        self.end = time.perf_counter()
        print('(Time elapsed(s) in {}: {:10.6f}sec)'.format(task, self.end-self.start))